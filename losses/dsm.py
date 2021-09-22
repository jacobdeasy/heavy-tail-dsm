import numpy as np
import torch

from scipy.stats import gennorm
from torch.distributions.gamma import Gamma


def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=(1, 2, 3)) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)


def anneal_dsm_score_estimation_gennorm(scorenet, samples, sigmas, beta, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))

    alpha = 2 ** 0.5
    gamma = np.random.gamma(shape=1+1/beta, scale=2**(beta/2), size=samples.shape)
    delta = alpha * gamma ** (1 / beta) / (2 ** 0.5)
    perturbation = torch.tensor((2 * np.random.rand(*samples.shape) - 1) * delta).float().to(samples.device)

    perturbed_samples = samples + perturbation * used_sigmas
    target = _gennorm_score(perturbed_samples, mu=samples, alpha=used_sigmas * 2.0 ** 0.5, beta=beta)
    scores = scorenet(perturbed_samples, labels)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=(1, 2, 3)) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)


def _gennorm_score(x, mu=0.0, alpha=1.0, beta=2.0):
    return - (beta / alpha ** beta) * torch.sign(x - mu) * torch.abs(x - mu) ** (beta - 1)
