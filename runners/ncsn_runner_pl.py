import argparse
import contextlib
import logging
import numpy as np
import os
import platform
import pytorch_lightning as pl
import time
import torch

from PIL import Image
from torch._C import memory_format
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm, trange

from . import fast_collate, get_hooks
from datasets import data_transform, get_dataset, inverse_data_transform
from losses import get_optimizer
from losses.dsm import anneal_dsm_score_estimation_gennorm
from models import (anneal_Langevin_dynamics,
                    anneal_Langevin_dynamics_inpainting,
                    anneal_Langevin_dynamics_interpolation,
                    get_sigmas)
from models.ema import EMAHelper
from models.ncsn import CondRefineNetDilated
from models.ncsnv2 import NCSNv2, NCSNv2Deeper, NCSNv2Deepest


__all__ = ['NCSNRunner']


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


def get_model(config: argparse.Namespace) -> None:
    if config.data.dataset == 'MNIST' or config.data.dataset == 'FashionMNIST':
        return CondRefineNetDilated(config)
    elif config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config)


class NCSNRunner(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, config: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.config = config

        os.makedirs(os.path.join(args.log_path, 'samples'), exist_ok=True)
        self.beta = self.args.beta
        self.sigmas = get_sigmas(self.config)

        # Data
        dataset, test_dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                pin_memory=True, num_workers=self.config.data.num_workers, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=self.config.data.num_workers, drop_last=True, collate_fn=collate_fn)
        test_iter = iter(test_loader)

        self.model = get_model(self.config)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model)
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            model.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        # Logging
        tb_logger = self.config.tb_logger
        hook, test_hook, tb_hook, test_tb_hook = get_hooks(tb_logger, self.config, self.sigmas, step=step)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return get_optimizer(self.config, self.model.parameters())

    def get_noise(self, x, used_sigmas):
        if self.args.beta == 2.0:
            noise = torch.randn_like(x) * used_sigmas
            target = - 1 / (used_sigmas ** 2) * noise
        else:
            alpha = 2 ** 0.5
            gamma = np.random.gamma(shape=1+1/self.beta, scale=2**(self.beta/2), size=x.shape)
            delta = alpha * gamma ** (1 / self.beta) / (2 ** 0.5)
            gn_samples = (2 * np.random.rand(*x.shape) - 1) * delta

            noise = torch.tensor(gn_samples).float().to(x.device) * used_sigmas
            constant = - self.beta / (used_sigmas * 2.0 ** 0.5) ** self.beta
            target = constant * torch.sign(noise) * torch.abs(noise) ** (self.beta - 1)

        return noise, target

    def weighted_mse(self, scores, targets, used_sigmas):
        return 0.5 * ((scores - targets) ** 2).sum(dim=(1, 2, 3)) * used_sigmas.squeeze() ** self.config.training.anneal_power

    def step(self, x, model):
        labels = torch.randint(0, len(self.sigmas), (x.shape[0],), device=x.device)
        used_sigmas = self.sigmas[labels].view(x.shape[0], *([1] * len(x.shape[1:])))
        noise, targets = self.get_noise(x, used_sigmas=used_sigmas)
        x_noised = x + noise

        if self.config.model.variational:
            scores, kl = model(x_noised, labels)
        else:
            scores = model(x_noised, labels)

        loss = self.weighted_mse(scores, targets, used_sigmas=used_sigmas)
        if self.hook is not None:
            self.hook(loss, labels)
        loss = loss.mean(dim=0)
        if self.config.model.variational:
            loss += kl['loss']
            if self.training:
                self.log('kl/loss', kl['loss'])
                self.log()

        return loss

    def training_step(self, batch, batch_idx):
        t_start = time.time()
        loss = self.step(data_transform(batch[0]), self.model)
        self.log('iter_per_s', 1 / (time.time() - t_start))
        self.log('loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.config.model.ema:
            test_model = self.ema_helper.ema_copy(self.model)
        else:
            test_model = self.model
        test_model.eval()

        loss = self.step(data_transform(batch[0]), test_model)
        self.log('test_loss', loss)

        return loss

    def train(self) -> None:
        if step % 200 == 0:
            tb_logger.add_histogram('kl/log_var', kl['logvar'], global_step=step)

            model.train()

        if step % self.config.training.snapshot_freq == 0:
            states = [model.state_dict(), optimizer.state_dict(), epoch, step]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())

            torch.save(states, os.path.join(self.args.log_path, f'checkpoint_{step}.pth'))
            torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

            if self.config.training.snapshot_sampling:
                if self.config.model.ema:
                    test_model = ema_helper.ema_copy(model)
                else:
                    test_model = model
                test_model.eval()

                init_samples = torch.rand(36, self.config.data.channels,
                                        self.config.data.image_size, self.config.data.image_size,
                                        device=self.config.device)
                init_samples = data_transform(self.config, init_samples)
                all_samples = anneal_Langevin_dynamics(init_samples, test_model, sigmas.cpu().numpy(),
                                                        self.config.sampling.n_steps_each,
                                                        self.config.sampling.step_lr,
                                                        final_only=True, verbose=True,
                                                        denoise=self.config.sampling.denoise)
                sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                self.config.data.image_size,
                                                self.config.data.image_size)
                sample = inverse_data_transform(self.config, sample)

                image_grid = make_grid(sample, 5)
                save_image(image_grid,
                            os.path.join(self.args.log_sample_path, f'image_grid_{step}.png'))
                torch.save(sample, os.path.join(self.args.log_sample_path, f'samples_{step}.pth'))

                del test_model
                del all_samples

            model.train()
