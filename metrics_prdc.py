import argparse
import numpy as np
import os

from PIL import Image
from prdc import compute_prdc
from tqdm import tqdm
from typing import Optional


def run_prdc(gen_dir: str,
             ref_dir: str,
             max_samples: Optional[int] = 10000
             ) -> None:
    ref_ims = []
    for path in tqdm(os.listdir(ref_dir)[:max_samples]):
        ref_ims.append(np.array(Image.open(os.path.join(ref_dir, path))))
    ref_ims = np.stack(ref_ims).reshape(-1, 28 * 28)

    gen_ims = []
    for path in tqdm(os.listdir(gen_dir)[:max_samples]):
        gen_ims.append(np.array(Image.open(os.path.join(gen_dir, path))))
    gen_ims = np.stack(gen_ims)[..., 0].reshape(-1, 28 * 28)

    metrics = compute_prdc(
        real_features=ref_ims,
        fake_features=gen_ims,
        nearest_k=5
    )

    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--gen_dir', type=str, help='Folder of samples for metrics.')
    parser.add_argument('--ref_dir', type=str, help='Folder of reference data for metrics.')
    parser.add_argument('--max_samples', type=int, help='Maximum number of samples to compare.')
    args = parser.parse_args()

    run_prdc(args.gen_dir, args.ref_dir)
