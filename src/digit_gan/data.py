"""Dataset loading and device selection."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(dataset: str) -> tuple[Dataset, int, int]:
    """Load the requested dataset, downloading it under ``data/`` if needed.

    Returns (dataset, img_channels, img_size).
    """
    if dataset == "mnist":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
        return train_ds, 1, 28
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
    return train_ds, 3, 32
