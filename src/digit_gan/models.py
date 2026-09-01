"""DCGAN model definitions: Generator and Discriminator."""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN generator: maps a latent vector z to an image."""

    def __init__(self, z_dim: int = 100, img_ch: int = 1, img_size: int = 28) -> None:
        super().__init__()
        self.img_ch = img_ch
        self.img_size = img_size
        base = 64
        if img_size == 28:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(z_dim, base * 4, 7, 1, 0, bias=False),
                nn.BatchNorm2d(base * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 2, img_ch, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        else:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(z_dim, base * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(base * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base),
                nn.ReLU(True),
                nn.ConvTranspose2d(base, img_ch, 4, 2, 1, bias=False),
                nn.Tanh(),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """DCGAN discriminator: classifies an image as real or fake."""

    def __init__(self, img_ch: int = 1, img_size: int = 28) -> None:
        super().__init__()
        base = 64
        if img_size == 28:
            self.net = nn.Sequential(
                nn.Conv2d(img_ch, base, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear((base * 2) * 7 * 7, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(img_ch, base, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear((base * 4) * 4 * 4, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (no sigmoid); pair with ``nn.BCEWithLogitsLoss``."""
        return self.net(x)
