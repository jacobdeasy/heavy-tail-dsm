import logging
import numpy as np
import os
import torch

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from losses import get_optimizer
from losses.dsm import anneal_dsm_score_estimation, anneal_dsm_score_estimation_gennorm
from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from models.ncsn import NCSN, NCSNdeeper
from datasets import data_transform, get_dataset, inverse_data_transform
from models import (anneal_Langevin_dynamics,
                    anneal_Langevin_dynamics_inpainting,
                    anneal_Langevin_dynamics_interpolation)
from models import get_sigmas
from models.ema import EMAHelper

__all__ = ['NCSNRunner']


def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)


class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def train(self):
        dataset, test_dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = get_model(self.config)

        score = torch.nn.DataParallel(score)
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        sigmas = get_sigmas(self.config)

        if self.config.training.log_all_sigmas:
            test_loss_per_sigma = [None for _ in range(len(sigmas))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar(f'test_loss_sigma_{i}', test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(len(sigmas)):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(len(sigmas)):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar(f'test_loss_sigma_{i}', test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for _, (X, _) in enumerate(dataloader):
                score.train()
                step += 1

                X = X.to(self.config.device)
                X = data_transform(self.config, X)

                if self.args.beta == 2.0:
                    loss = anneal_dsm_score_estimation(score, X, sigmas, None,
                                                       self.config.training.anneal_power,
                                                       hook)
                else:
                    loss = anneal_dsm_score_estimation_gennorm(score, X, sigmas, None,
                                                               self.config.training.anneal_power,
                                                               hook)
                tb_logger.add_scalar('loss', loss, global_step=step)
                tb_hook()

                logging.info(f"step: {step}, loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    with torch.no_grad():
                        if self.args.beta == 2.0:
                            test_dsm_loss = anneal_dsm_score_estimation(test_score, test_X, sigmas, None,
                                                                        self.config.training.anneal_power,
                                                                        hook=test_hook)
                        else:
                            test_dsm_loss = anneal_dsm_score_estimation_gennorm(test_score, test_X, sigmas, None,
                                                                                self.config.training.anneal_power,
                                                                                hook=test_hook)
                        tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
                        test_tb_hook()
                        logging.info(f"step: {step}, test_loss: {test_dsm_loss.item()}")

                        del test_score

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(states, os.path.join(self.args.log_path, f'checkpoint_{step}.pth'))
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()

                        init_samples = torch.rand(36, self.config.data.channels,
                                                  self.config.data.image_size, self.config.data.image_size,
                                                  device=self.config.device)
                        init_samples = data_transform(self.config, init_samples)

                        all_samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                               self.config.sampling.n_steps_each,
                                                               self.config.sampling.step_lr,
                                                               final_only=True, verbose=True,
                                                               denoise=self.config.sampling.denoise)

                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, 6)
                        save_image(image_grid,
                                   os.path.join(self.args.log_sample_path, f'image_grid_{step}.png'))
                        torch.save(sample, os.path.join(self.args.log_sample_path, f'samples_{step}.pth'))

                        del test_score
                        del all_samples

    def sample(self):
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        dataset, _ = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                num_workers=4)

        score.eval()

        if not self.config.sampling.fid:
            if self.config.sampling.inpainting:
                data_iter = iter(dataloader)
                refer_images, _ = next(data_iter)
                refer_images = refer_images.to(self.config.device)
                width = int(np.sqrt(self.config.sampling.batch_size))
                init_samples = torch.rand(width, width, self.config.data.channels,
                                          self.config.data.image_size,
                                          self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)
                all_samples = anneal_Langevin_dynamics_inpainting(init_samples, refer_images[:width, ...], score,
                                                                  sigmas,
                                                                  self.config.data.image_size,
                                                                  self.config.sampling.n_steps_each,
                                                                  self.config.sampling.step_lr)

                torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pth'))
                refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1,
                                                                                                     *refer_images.shape[
                                                                                                      1:])
                save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'), nrow=width)

                if not self.config.sampling.final_only:
                    for i, sample in enumerate(tqdm.tqdm(all_samples)):
                        sample = sample.view(self.config.sampling.batch_size, self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, f'image_grid_{i}.png'))
                        torch.save(sample, os.path.join(self.args.image_folder, f'completion_{i}.pth'))
                else:
                    sample = all_samples[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        f'image_grid_{self.config.sampling.ckpt_id}.png'))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    f'completion_{self.config.sampling.ckpt_id}.pth'))

            elif self.config.sampling.interpolation:
                if self.config.sampling.data_init:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                else:
                    init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_size,
                                              device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics_interpolation(init_samples, score, sigmas,
                                                                     self.config.sampling.n_interpolations,
                                                                     self.config.sampling.n_steps_each,
                                                                     self.config.sampling.step_lr, verbose=True,
                                                                     final_only=self.config.sampling.final_only)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, nrow=self.config.sampling.n_interpolations)
                        save_image(image_grid, os.path.join(self.args.image_folder, f'image_grid_{i}.png'))
                        torch.save(sample, os.path.join(self.args.image_folder, f'samples_{i}.pth'))
                else:
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, self.config.sampling.n_interpolations)
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        f'image_grid_{self.config.sampling.ckpt_id}.png'))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    f'samples_{self.config.sampling.ckpt_id}.pth'))

            else:
                if self.config.sampling.data_init:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                else:
                    init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_size,
                                              device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=True,
                                                       final_only=self.config.sampling.final_only,
                                                       denoise=self.config.sampling.denoise)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, f'image_grid_{i}.png'))
                        torch.save(sample, os.path.join(self.args.image_folder, f'samples_{i}.pth'))
                else:
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        f'image_grid_{self.config.sampling.ckpt_id}.png'))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    f'samples_{self.config.sampling.ckpt_id}.pth'))

        else:
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // self.config.sampling.batch_size
            if self.config.sampling.data_init:
                dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                        num_workers=4)
                data_iter = iter(dataloader)

            img_id = 0
            for _ in tqdm.tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
                if self.config.sampling.data_init:
                    try:
                        samples, _ = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    samples = samples + sigmas_th[0] * torch.randn_like(samples)
                else:
                    samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size, device=self.config.device)
                    samples = data_transform(self.config, samples)

                all_samples = anneal_Langevin_dynamics(samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=False,
                                                       denoise=self.config.sampling.denoise)

                samples = all_samples[-1]
                for img in samples:
                    img = inverse_data_transform(self.config, img)

                    save_image(img, os.path.join(self.args.image_folder, f'image_{img_id}.png'))
                    img_id += 1

    def test(self):
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas = get_sigmas(self.config)

        dataset, test_dataset = get_dataset(self.args, self.config)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.test.batch_size, shuffle=True,
                                     num_workers=self.config.data.num_workers, drop_last=True)

        verbose = False
        for ckpt in range(self.config.test.begin_ckpt, self.config.test.end_ckpt + 1, 5000):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            step = 0
            mean_loss = 0.
            mean_grad_norm = 0.
            average_grad_scale = 0.
            for x, _ in test_dataloader:
                step += 1

                x = x.to(self.config.device)
                x = data_transform(self.config, x)

                with torch.no_grad():
                    if self.args.beta == 2.0:
                        test_loss = anneal_dsm_score_estimation(score, x, sigmas, None,
                                                                self.config.training.anneal_power)
                    else:
                        test_loss = anneal_dsm_score_estimation_gennorm(score, x, sigmas, None,
                                                                        self.config.training.anneal_power)
                    if verbose:
                        logging.info(f"step: {step}, test_loss: {test_loss.item()}")

                    mean_loss += test_loss.item()

            mean_loss /= step
            mean_grad_norm /= step
            average_grad_scale /= step

            logging.info(f"ckpt: {ckpt}, average test loss: {mean_loss}")

    def fast_fid(self):
        ### Test the fids of ensembled checkpoints.
        ### Shouldn't be used for models with ema
        if self.config.fast_fid.ensemble:
            if self.config.model.ema:
                raise RuntimeError("Cannot apply ensembling to models with EMA.")
            self.fast_ensemble_fid()
            return

        from evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        for ckpt in range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, self.config.fast_fid.ckpt_interval):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, f'ckpt_{ckpt}')
            os.makedirs(output_path, exist_ok=True)
            for _ in tqdm(range(num_iters), desc=f'Checkpoint {ckpt}'):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, f'sample_{id}.png'))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print(f"ckpt: {ckpt}, fid: {fid}")

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def fast_ensemble_fid(self):
        from evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle

        num_ensembles = 5
        scores = [NCSN(self.config).to(self.config.device) for _ in range(num_ensembles)]
        scores = [torch.nn.DataParallel(score) for score in scores]

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        for ckpt in range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, 5000):
            begin_ckpt = max(self.config.fast_fid.begin_ckpt, ckpt - (num_ensembles - 1) * 5000)
            index = 0
            for i in range(begin_ckpt, ckpt + 5000, 5000):
                states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{i}.pth'),
                                    map_location=self.config.device)
                scores[index].load_state_dict(states[0])
                scores[index].eval()
                index += 1

            def scorenet(x, labels):
                num_ckpts = (ckpt - begin_ckpt) // 5000 + 1
                return sum([scores[i](x, labels) for i in range(num_ckpts)]) / num_ckpts

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, f'ckpt_{ckpt}')
            os.makedirs(output_path, exist_ok=True)
            for i in tqdm(range(num_iters), desc=f'Checkpoint {ckpt}'):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, scorenet, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, f'sample_{id}.png'))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print(f"ckpt: {ckpt}, fid: {fid}")

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)
