import argparse
import contextlib
import logging
import numpy as np
import os
import platform
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
from models import ald, ald_inpaint, ald_interp, get_sigmas, vald
from models.ema import EMAHelper
from models.ncsn import CondRefineNetDilated
from models.ncsnv2 import NCSNv2, NCSNv2Deeper, NCSNv2Deepest


__all__ = ['NCSNRunner']


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


def get_model(config: argparse.Namespace) -> None:
    if config.data.dataset == 'MNIST' or config.data.dataset == 'FashionMNIST':
        # return CondRefineNetDilated(config)
        return NCSNv2(config)
    elif config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config)


class NCSNRunner():
    def __init__(self, args: argparse.Namespace, config: argparse.Namespace) -> None:
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def format_memory(self) -> None:
        if self.config.training.channels_last:
            self.memory_format = torch.channels_last
            print('\nWARNING: Channels last transforms are not implemented.\n')
        else:
            self.memory_format = torch.contiguous_format
        if not platform.system() == 'Windows':
            self.collate_fn = lambda b: fast_collate(b, memory_format=self.memory_format)
        else:
            self.collate_fn = None

    def train(self) -> None:
        self.format_memory()

        # Data
        shape = (self.config.training.batch_size,
                 self.config.data.channels,
                 self.config.data.image_size,
                 self.config.data.image_size)
        dataset, test_dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                pin_memory=True, num_workers=self.config.data.num_workers,
                                collate_fn=self.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=self.config.data.num_workers, drop_last=True,
                                 collate_fn=self.collate_fn)
        test_iter = iter(test_loader)

        # Model
        model = get_model(self.config)
        model.to(self.config.device).to(memory_format=self.memory_format)
        model = torch.nn.DataParallel(model)

        # Optimizer
        optimizer = get_optimizer(self.config, model.parameters())
        if self.config.training.amp:
            scaler = torch.cuda.amp.GradScaler()

        # Hyperparams
        beta = self.args.beta
        sigmas = get_sigmas(self.config)

        # Training options
        start_epoch = 0
        step = 0
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
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
        hook, test_hook, tb_hook, test_tb_hook = get_hooks(tb_logger, self.config, sigmas, step=step)

        # Training
        with trange(self.config.training.n_iters-step, desc=f"Training progress") as pbar:
            for epoch in range(start_epoch, self.config.training.n_epochs):
                for _, (X, _) in enumerate(dataloader):
                    t_start = time.time()
                    step += 1

                    X = X.to(self.config.device, memory_format=self.memory_format)
                    X = data_transform(self.config, X)

                    labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                    used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))

                    if beta == 2.0:
                        noise = torch.randn_like(X) * used_sigmas
                        target = - 1 / (used_sigmas ** 2) * noise
                    else:
                        alpha = 2 ** 0.5
                        gamma = np.random.gamma(shape=1+1/beta, scale=2**(beta/2), size=X.shape)
                        delta = alpha * gamma ** (1 / beta) / (2 ** 0.5)
                        gn_samples = (2 * np.random.rand(*X.shape) - 1) * delta

                        noise = torch.tensor(gn_samples).float().to(X.device) * used_sigmas
                        constant = - beta / (used_sigmas * 2.0 ** 0.5) ** beta
                        target = constant * torch.sign(noise) * torch.abs(noise) ** (beta - 1)

                    X_noised = X + noise

                    with torch.cuda.amp.autocast() if self.config.training.amp else dummy_context_mgr() as _:
                        if self.config.model.variational:
                            scores, kl = model(X_noised, labels)
                        else:
                            scores = model(X_noised, labels)
                        loss = 0.5 * ((scores - target) ** 2).sum(dim=(1, 2, 3)) * used_sigmas.squeeze() ** self.config.training.anneal_power
                        if hook is not None:
                            hook(loss, labels)
                        loss = loss.mean(dim=0)
                        if self.config.model.variational:
                            loss += kl['loss']

                    for p in model.parameters():
                        p.grad = None
                    if self.config.training.amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    if self.config.model.ema:
                        ema_helper.update(model)

                    pbar.set_postfix(loss=loss.item())
                    pbar.update()

                    if step >= self.config.training.n_iters:
                        return 0

                    if step % 200 == 0:
                        tb_logger.add_scalar('iter_per_s', 1/(time.time() - t_start), global_step=step)
                        tb_logger.add_scalar('loss', loss, global_step=step)
                        if self.config.model.variational:
                            tb_logger.add_scalar('kl/loss', kl['loss'], global_step=step)
                            tb_logger.add_histogram('kl/log_var', kl['log_var'], global_step=step)
                        tb_hook()

                        if self.config.model.ema:
                            test_model = ema_helper.ema_copy(model)
                        else:
                            test_model = model
                        test_model.eval()

                        try:
                            X_test, _ = next(test_iter)
                        except StopIteration:
                            test_iter = iter(test_loader)
                            X_test, _ = next(test_iter)
                        X_test = X_test.to(self.config.device, memory_format=self.memory_format)
                        X_test = data_transform(self.config, X_test)

                        with torch.no_grad():
                            test_dsm_loss = anneal_dsm_score_estimation_gennorm(
                                model, X, sigmas, self.args.beta, None, self.config.training.anneal_power,
                                hook=test_hook, var=self.config.model.variational)
                            tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
                            test_tb_hook()
                            del test_model

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

                            init_samples = torch.rand(36, *shape[1:], device=self.config.device)
                            init_samples = data_transform(self.config, init_samples)
                            all_samples = ald(init_samples, test_model, sigmas.cpu().numpy(),
                                              self.config.sampling.n_steps_each,
                                              self.config.sampling.step_lr,
                                              verbose=True,
                                              denoise=self.config.sampling.denoise)
                            sample = all_samples[-1].view(all_samples[-1].shape[0], *shape[1:])
                            sample = inverse_data_transform(self.config, sample)

                            image_grid = make_grid(sample, 5)
                            save_image(image_grid,
                                       os.path.join(self.args.log_sample_path, f'image_grid_{step}.png'))

                            del test_model
                            del all_samples

                        model.train()

    def sample(self) -> None:
        self.format_memory()

        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'),
                                map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path,
                                             f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        model = get_model(self.config)
        model.to(self.config.device).to(memory_format=self.memory_format)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        dataset, _ = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                pin_memory=True, num_workers=4)

        shape = (self.config.sampling.batch_size,
                 self.config.data.channels,
                 self.config.data.image_size,
                 self.config.data.image_size)

        if not self.config.sampling.fid:
            if self.config.sampling.data_init:
                data_iter = iter(dataloader)
                init_samples, _ = next(data_iter)
            else:
                init_samples = torch.rand(*shape, device=self.config.device)
            init_samples = init_samples.to(self.config.device).to(memory_format=self.memory_format)
            init_samples = data_transform(self.config, init_samples)
            if self.config.sampling.data_init:
                init_samples += sigmas_th[0] * torch.randn_like(init_samples)

            if self.config.sampling.inpainting:
                refer_images, _ = next(data_iter)
                refer_images = refer_images.to(self.config.device)
                width = int(np.sqrt(self.config.sampling.batch_size))
                init_samples = torch.rand(width, width, *shape[1:], device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                all_samples = ald_inpaint(init_samples, refer_images[:width, ...], model,
                                          sigmas,
                                          self.config.data.image_size,
                                          self.config.sampling.n_steps_each,
                                          self.config.sampling.step_lr)
                refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1,
                                                                                                     *refer_images.shape[
                                                                                                      1:])
                save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'), nrow=width)

            else:
                if self.config.sampling.interpolation:
                    all_samples = ald_interp(init_samples, model, sigmas,
                                             self.config.sampling.n_interpolations,
                                             self.config.sampling.n_steps_each,
                                             self.config.sampling.step_lr,
                                             verbose=True)
                else:
                    fn = vald if self.config.model.variational else ald
                    all_samples = fn(init_samples, model, sigmas,
                                     self.config.sampling.n_steps_each,
                                     self.config.sampling.step_lr,
                                     verbose=True,
                                     denoise=self.config.sampling.denoise)

            imgs = []
            for i, sample in tqdm(enumerate(all_samples), total=len(all_samples), desc="Saving samples"):
                sample = sample.view(sample.shape[0], *shape[1:])
                sample = inverse_data_transform(self.config, sample)

                if i % 10 == 0:
                    image_grid = make_grid(sample, 5)
                    save_image(image_grid, os.path.join(self.args.image_folder, f'image_grid_{i}.png'))
                    img = torch.clamp(255 * image_grid + 0.5, min=0, max=255)
                    imgs.append(Image.fromarray(img.permute(1, 2, 0).to('cpu', torch.uint8).numpy()))

            imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"),
                         save_all=True, append_images=imgs[1:], duration=1, loop=0)

        else:
            n_rounds = self.config.sampling.num_samples4fid // self.config.sampling.batch_size
            if self.config.sampling.data_init:
                dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                        pin_memory=True, num_workers=4)
                data_iter = iter(dataloader)

            img_id = 0
            for _ in tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
                if self.config.sampling.data_init:
                    try:
                        init_samples, _ = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        init_samples, _ = next(data_iter)
                else:
                    init_samples = torch.rand(*shape, device=self.config.device)
                init_samples = init_samples.to(self.config.device).to(memory_format=memory_format)
                init_samples = data_transform(self.config, init_samples)
                if self.config.sampling.data_init:
                    init_samples += sigmas_th[0] * torch.randn_like(init_samples)

                fn = vald if self.config.model.variational else ald
                all_samples = fn(init_samples, model, sigmas,
                                 self.config.sampling.n_steps_each,
                                 self.config.sampling.step_lr,
                                 verbose=True,
                                 denoise=self.config.sampling.denoise)

                for img in all_samples[-1]:
                    img = inverse_data_transform(self.config, img)
                    save_image(img, os.path.join(self.args.image_folder, f'image_{img_id}.png'))
                    img_id += 1

    def test(self) -> None:
        self.format_memory()

        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'),
                                map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path,
                                             f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        model = get_model(self.config)
        model.to(self.config.device).to(memory_format=self.memory_format)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        sigmas = get_sigmas(self.config)

        _, test_dataset = get_dataset(self.args, self.config)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.test.batch_size, shuffle=True,
                                     pin_memory=True, num_workers=self.config.data.num_workers, drop_last=False)

        verbose = False
        for ckpt in range(self.config.test.begin_ckpt, self.config.test.end_ckpt + 1, 5000):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                model.load_state_dict(states[0])

            if self.config.model.variational:
                model.train()
            else:
                model.eval()

            mean_loss = 0.
            for i, (x, _) in enumerate(test_dataloader):
                x = x.to(self.config.device).to(memory_format=memory_format)
                x = data_transform(self.config, x)

                with torch.no_grad():
                    test_loss = anneal_dsm_score_estimation_gennorm(model, x, sigmas, self.args.beta,
                                                                    None, self.config.training.anneal_power, None)
                    if verbose:
                        logging.info(f"step: {i}, test_loss: {test_loss.item()}")
                    mean_loss += test_loss.item()

            logging.info(f"ckpt: {ckpt}, average test loss: {mean_loss / i}")
