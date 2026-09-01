"""Unit tests for build_dataset() that don't touch the network.

We mock torchvision's dataset classes so we can verify build_dataset()
wires up the right dataset class, root dir, and transform for each
`--dataset` choice, without requiring a real MNIST/CIFAR-10 download.
The real download path is exercised separately by the network-dependent
test in test_integration_mnist.py.
"""

from unittest.mock import patch

import train_gan


def test_build_dataset_mnist_wiring():
    with patch("train_gan.datasets.MNIST") as mock_mnist:
        mock_mnist.return_value = "fake-mnist-dataset"
        ds, img_ch, img_size = train_gan.build_dataset("mnist")

    assert ds == "fake-mnist-dataset"
    assert (img_ch, img_size) == (1, 28)
    _, kwargs = mock_mnist.call_args
    assert kwargs["root"] == "data"
    assert kwargs["train"] is True
    assert kwargs["download"] is True


def test_build_dataset_cifar10_wiring():
    with patch("train_gan.datasets.CIFAR10") as mock_cifar:
        mock_cifar.return_value = "fake-cifar-dataset"
        ds, img_ch, img_size = train_gan.build_dataset("cifar10")

    assert ds == "fake-cifar-dataset"
    assert (img_ch, img_size) == (3, 32)
    _, kwargs = mock_cifar.call_args
    assert kwargs["root"] == "data"
    assert kwargs["train"] is True
    assert kwargs["download"] is True
