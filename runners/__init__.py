import argparse
import numpy as np
import torch

from torch import Tensor
from typing import Any, Callable, List, Optional, Tuple, Union


def fast_collate(batch: List[Tuple],
                 memory_format: Any
                 ) -> Tuple[Tensor, Tensor]:
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        np_array = np.asarray(img, dtype=np.uint8)
        if np_array.ndim < 3:
            np_array = np.expand_dims(np_array, axis=-1)
        np_array = np.moveaxis(np_array, source=2, destination=0)
        tensor[i] += torch.from_numpy(np_array)
    tensor = tensor.float()

    return tensor, targets


def get_hooks(tb_logger: torch.utils.tensorboard.SummaryWriter,
              config: argparse.Namespace,
              sigmas: torch.Tensor,
              step: Optional[int] = 0
              ) -> Union[Callable, Callable, Callable, Callable]:
    if config.training.log_all_sigmas:
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
    
    return hook, test_hook, tb_hook, test_tb_hook
