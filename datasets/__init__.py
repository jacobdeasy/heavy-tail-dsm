import argparse
import numpy as np
import os
import torch
from torch.utils import data
import torchvision.transforms as transforms

from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, LSUN
from typing import Optional, Tuple

from datasets.celeba import CelebA
from datasets.ffhq import FFHQ


def get_dataset(args: argparse.Namespace,
                config: argparse.Namespace
                ) -> Tuple[Dataset, Dataset]:
    if config.data.random_flip is False:
        train_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])

    if config.data.dataset == 'MNIST':
        dataset = MNIST(os.path.join(args.exp, 'datasets', 'mnist'), train=True, download=True,
                        transform=train_transform)
        # dataset.data = torch.cat((
        #     dataset.data[(dataset.targets == 1)],
        #     dataset.data[(dataset.targets == 8)][:3000]))
        # dataset.targets = torch.cat((
        #     dataset.targets[(dataset.targets == 1)],
        #     dataset.targets[(dataset.targets == 8)][:3000]))
        test_dataset = MNIST(os.path.join(args.exp, 'datasets', 'mnist_test'), train=False, download=True,
                             transform=test_transform)
        # test_dataset.data = torch.cat((
        #     test_dataset.data[(test_dataset.targets == 1)],
        #     test_dataset.data[(test_dataset.targets == 8)][:500]))
        # test_dataset.targets = torch.cat((
        #     test_dataset.targets[(test_dataset.targets == 1)],
        #     test_dataset.targets[(test_dataset.targets == 8)][:500]))

    elif config.data.dataset == 'FashionMNIST':
        dataset = FashionMNIST(os.path.join(args.exp, 'datasets', 'mnist'), train=True, download=True,
                               transform=train_transform)
        test_dataset = FashionMNIST(os.path.join(args.exp, 'datasets', 'mnist_test'), train=False, download=True,
                                    transform=test_transform)

    elif config.data.dataset == 'CIFAR10':
        dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
                          transform=train_transform)
        test_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
                               transform=test_transform)

    elif config.data.dataset == 'CELEBA':
        if config.data.random_flip:
            dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=True)
        else:
            dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

        test_dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba_test'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=True)

    elif config.data.dataset == 'LSUN':
        train_folder = '{}_train'.format(config.data.category)
        val_folder = '{}_val'.format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                             ]))
        else:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

        test_dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[val_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]), resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.ToTensor(),
                           resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    return dataset, test_dataset


def logit_transform(image: Tensor, lam: Optional[float] = 1e-6) -> Tensor:
    image = lam + (1 - 2 * lam) * image

    return torch.log(image) - torch.log1p(-image)


def data_transform(config: argparse.Namespace, X: Tensor) -> Tensor:
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config: argparse.Namespace, X: Tensor) -> Tensor:
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
