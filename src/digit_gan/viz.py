"""Shared image-grid plotting helper.

Used by both ``train_gan.py`` (per-epoch sample grids) and ``sample.py``
(random-sample and latent-interpolation grids) so the make_grid/imshow/savefig
boilerplate lives in one place.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import utils as vutils


def save_grid_image(
    images: torch.Tensor,
    outpath: str,
    title: str,
    nrow: int,
    figsize: tuple[float, float] = (6, 6),
) -> None:
    """Arrange ``images`` into a grid and save it as a PNG.

    Args:
        images: Batch of images, shape (N, C, H, W), values in [-1, 1].
        outpath: Destination file path for the PNG.
        title: Title drawn above the grid.
        nrow: Number of images per row in the grid.
        figsize: Matplotlib figure size in inches.
    """
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
