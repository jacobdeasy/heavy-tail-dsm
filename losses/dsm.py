import numpy as np
import torch

from torch import Tensor
from typing import Any, Callable, Optional


def anneal_dsm_score_estimation(scorenet: torch.nn.Module,
                                samples: Tensor,
                                sigmas: Tensor,
                                labels: Optional[Any] = None,
                                anneal_power: Optional[float] = 2.0,
                                hook: Optional[Callable] = None,
                                var: Optional[bool] = False
                                ) -> Tensor:
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))

    noise = torch.randn_like(samples) * used_sigmas

    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    if var:
        scores, kl = scorenet(perturbed_samples, labels)
    else:
        scores = scorenet(perturbed_samples, labels)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=(1, 2, 3)) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)


def anneal_dsm_score_estimation_gennorm(scorenet: torch.nn.Module,
                                        samples: Tensor,
                                        sigmas: Tensor,
                                        beta: float,
                                        labels: Optional[Any] = None,
                                        anneal_power: Optional[float] = 2.0,
                                        hook: Optional[Callable] = None,
                                        var: Optional[bool] = False
                                        ) -> Tensor:
    if beta == 2.0:
        return anneal_dsm_score_estimation(scorenet, samples, sigmas, None, anneal_power, hook, var)

    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))

    alpha = 2 ** 0.5
    gamma = np.random.gamma(shape=1+1/beta, scale=2**(beta/2), size=samples.shape)
    delta = alpha * gamma ** (1 / beta) / (2 ** 0.5)
    perturbation = torch.tensor((2 * np.random.rand(*samples.shape) - 1) * delta).float().to(samples.device)

    perturbed_samples = samples + perturbation * used_sigmas
    target = _gennorm_score(perturbed_samples, mu=samples, alpha=used_sigmas * 2.0 ** 0.5, beta=beta)
    if var:
        scores, kl = scorenet(perturbed_samples, labels)
    else:
        scores = scorenet(perturbed_samples, labels)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=(1, 2, 3)) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)


def _gennorm_score(x: Tensor,
                   mu: Optional[float] = 0.0,
                   alpha: Optional[float] = 1.0,
                   beta: Optional[float] = 2.0
                   ) -> Tensor:
    return - (beta / alpha ** beta) * torch.sign(x - mu) * torch.abs(x - mu) ** (beta - 1)
