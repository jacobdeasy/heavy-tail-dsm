import math
import torch
import torch.nn as nn

from argparse import Namespace
from torch import Tensor

from .layers import *
from .normalization import get_normalization


class CondRefineNetDilated(nn.Module):
    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.config = config

        self.channels = config.data.channels
        self.channel_dim = -1 if config.training.channels_last else 1
        self.ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = get_act(config)
        self.norm = get_normalization(config)

        self.logit_transform = config.data.logit_transform
        self.begin_conv = nn.Conv2d(self.channels, self.ngf, kernel_size=3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )
        self.res2 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )
        self.res3 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=2),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=2)]
        )
        if config.data.image_size == 28:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=True, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=False, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
            )

        self.refine1 = CondRefineBlock([2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act, start=True)
        self.refine2 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act)
        self.refine3 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, self.num_classes, self.norm, act=act)
        self.refine4 = CondRefineBlock([self.ngf, self.ngf], self.ngf, self.num_classes, self.norm, act=act, end=True)

        self.normalizer = self.norm(self.ngf, self.num_classes)
        if config.model.variational:
            self.end_conv = nn.Conv2d(self.ngf, 2 * self.channels, kernel_size=3, stride=1, padding=1)
        else:
            self.end_conv = nn.Conv2d(self.ngf, self.channels, kernel_size=3, stride=1, padding=1)

    def _compute_cond_module(self, module: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        for m in module:
            x = m(x, y)

        return x

    def _reparameterise_conv(self, x: Tensor) -> Tensor:
        mu, log_var = x.split(self.channels, dim=self.channel_dim)
        # prior_var = 1
        # kl_loss = (0.5 * (log_var.exp() / prior_var - 1 - log_var + math.log(prior_var))).mean()
        kl = {
            'loss': (0.5 * (log_var.exp() - 1 - log_var)).mean(),
            'log_var': log_var
        }
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std, device=std.device)
            return mu + std * eps, kl
        else:
            return mu, kl

    @torch.cuda.amp.autocast()
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if not self.logit_transform:
            x = 2 * x - 1.
        output = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, output, y)
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        layer4 = self._compute_cond_module(self.res4, layer3, y)

        ref1 = self.refine1([layer4], y, layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], y, layer2.shape[2:])
        output = self.refine4([layer1, ref3], y, layer1.shape[2:])

        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)

        if self.config.model.variational:
            output, kl = self._reparameterise_conv(output)
            return output, kl
        else:
            return output


class NCSNdeeper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=True)
        self.ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = get_act(config)
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, self.ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(self.ngf, self.num_classes)

        self.end_conv = nn.Conv2d(self.ngf, config.data.channels, 3, stride=1, padding=1)
        spec_norm = config.model.spec_norm

        self.res1 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, spec_norm=spec_norm),
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, spec_norm=spec_norm)]
        )

        self.res2 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, spec_norm=spec_norm),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, spec_norm=spec_norm)]
        )

        self.res3 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, spec_norm=spec_norm),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, spec_norm=spec_norm)]
        )

        self.res4 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 4 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=2, spec_norm=spec_norm),
            ConditionalResidualBlock(4 * self.ngf, 4 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=2, spec_norm=spec_norm)]
        )

        self.res5 = nn.ModuleList([
            ConditionalResidualBlock(4 * self.ngf, 4 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=4, spec_norm=spec_norm),
            ConditionalResidualBlock(4 * self.ngf, 4 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=4, spec_norm=spec_norm)]
        )

        self.refine1 = CondRefineBlock([4 * self.ngf], 4 * self.ngf, self.num_classes, self.norm, act=act, start=True, spec_norm=spec_norm)
        self.refine2 = CondRefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act, spec_norm=spec_norm)
        self.refine3 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act, spec_norm=spec_norm)
        self.refine4 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, self.num_classes, self.norm, act=act, spec_norm=spec_norm)
        self.refine5 = CondRefineBlock([self.ngf, self.ngf], self.ngf, self.num_classes, self.norm, act=act, end=True, spec_norm=spec_norm)

    def _compute_cond_module(self, module, x, y):
        for m in module:
            x = m(x, y)
        return x

    @torch.cuda.amp.autocast()
    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output, y)
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        layer4 = self._compute_cond_module(self.res4, layer3, y)
        layer5 = self._compute_cond_module(self.res5, layer4, y)

        ref1 = self.refine1([layer5], y, layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], y, layer4.shape[2:])
        ref3 = self.refine3([layer3, ref2], y, layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], y, layer2.shape[2:])
        output = self.refine5([layer1, ref4], y, layer1.shape[2:])

        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)

        return output
