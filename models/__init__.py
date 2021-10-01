import argparse
# import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import Tensor
from typing import List, Optional


def get_sigmas(config: argparse.Namespace) -> Tensor:
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas


@torch.no_grad()
def vald(x: Tensor,
         model: torch.nn.Module,
         sigmas: Tensor,
         n_steps_each: Optional[int] = 200,
         step_lr: Optional[float] = 0.000008,
         verbose: Optional[bool] = False,
         denoise: Optional[bool] = True
         ) -> List[Tensor]:
    model.train()
    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = (torch.ones(x.shape[0], device=x.device) * c).long()
            # step_size = step_lr * (sigma / sigmas[-1]) ** 2
            step_size = step_lr * (sigmas[0] / sigmas[-1]) ** 2
            for _ in range(n_steps_each):
                grad, _ = model(x, labels)
                x += step_size * grad

                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                image_norm = torch.norm(x.view(x.shape[0], -1), dim=-1).mean()
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                images.append(x.to('cpu'))
                if verbose:
                    print("level: {:.4f}, step_size: {:.4f}, grad_norm: {:.4f}, image_norm: {:.4f}, grad_mean_norm: {:.4f}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = ((len(sigmas) - 1) * torch.ones(x.shape[0], device=x.device)).long()
            x += sigmas[-1] ** 2 * model(x, last_noise)[0]
            images.append(x.to('cpu'))

        return images


@torch.no_grad()
def ald(x: Tensor,
        model: torch.nn.Module,
        sigmas: Tensor,
        n_steps_each: Optional[int] = 200,
        step_lr: Optional[float] = 0.000008,
        verbose: Optional[bool] = False,
        denoise: Optional[bool] = True
        ) -> List[Tensor]:
    model.eval()
    images = []

    with torch.no_grad():
        # grad_norms = []
        for c, sigma in enumerate(sigmas):
            labels = (torch.ones(x.shape[0], device=x.device) * c).long()
            # step_size = step_lr * (sigma / sigmas[-1]) ** 2
            step_size = step_lr * (sigmas[0] / sigmas[-1]) ** 2
            for _ in range(n_steps_each):
                grad = model(x, labels)

                noise = torch.randn_like(x)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                # grad_norms.append(grad_norm)
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x += step_size * grad + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x.view(x.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                images.append(x.to('cpu'))
                if verbose:
                    print("level: {:.4f}, step_size: {:.4f}, grad_norm: {:.4f}, image_norm: {:.4f}, snr: {:.4f}, grad_mean_norm: {:.4f}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
        # plt.plot(grad_norms)
        # plt.show()

        if denoise:
            last_noise = ((len(sigmas) - 1) * torch.ones(x.shape[0], device=x.device)).long()
            x += sigmas[-1] ** 2 * model(x, last_noise)
            images.append(x.to('cpu'))

        return images


@torch.no_grad()
def ald_inpaint(x: Tensor,
                refer_image: Tensor,
                model: torch.nn.Module,
                sigmas: Tensor,
                image_size: int,
                n_steps_each: Optional[int] = 100,
                step_lr: Optional[float] = 0.000008
                ) -> List[Tensor]:
    """Currently only good for 32x32 images. Assuming the right half is missing."""
    model.eval()
    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
    x = x.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = (torch.ones(x.shape[0], device=x.device) * c).long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for _ in range(n_steps_each):
                images.append(x.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x) * np.sqrt(step_size * 2)
                grad = model(x, labels)
                x += step_size * grad + noise
                print("class: {:.4f}, step_size: {:.4f}, mean {:.4f}, max {:.4f}".format(
                    c, step_size, grad.abs().mean(), grad.abs().max()))

    return images


@torch.no_grad()
def ald_interp(x: Tensor,
               model: torch.nn.Module,
               sigmas: Tensor,
               n_interpolations: int,
               n_steps_each: Optional[int] = 200,
               step_lr: Optional[float] = 0.000008,
               verbose: Optional[bool] = False
               ) -> List[Tensor]:
    model.eval()
    images = []

    n_rows = x.shape[0]

    x = x[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x = x.reshape(-1, *x.shape[2:])

    for c, sigma in enumerate(sigmas):
        labels = (torch.ones(x.shape[0], device=x.device) * c).long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for _ in range(n_steps_each):
            grad = model(x, labels)

            noise_p = torch.randn(n_rows, *x.shape[1:], device=x.device)
            noise_q = torch.randn(n_rows, *x.shape[1:], device=x.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x.view(x.shape[0], -1), dim=-1).mean()

            x += step_size * grad + noise * np.sqrt(step_size * 2)

            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm

            images.append(x.to('cpu'))
            if verbose:
                print("level: {:.4f}, step_size: {:.4f}, image_norm: {:.4f}, grad_norm: {:.4f}, snr: {:.4f}".format(
                    c, step_size, image_norm.item(), grad_norm.item(), snr.item()))

    return images
