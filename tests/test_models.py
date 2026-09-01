import torch

from digit_gan.models import Discriminator, Generator


def test_generator_mnist_output_shape():
    G = Generator(z_dim=100, img_ch=1, img_size=28)
    z = torch.randn(4, 100, 1, 1)
    out = G(z)
    assert out.shape == (4, 1, 28, 28)
    assert out.min() >= -1.0 and out.max() <= 1.0  # Tanh output range


def test_generator_cifar_output_shape():
    G = Generator(z_dim=100, img_ch=3, img_size=32)
    z = torch.randn(4, 100, 1, 1)
    out = G(z)
    assert out.shape == (4, 3, 32, 32)


def test_discriminator_mnist_output_shape():
    D = Discriminator(img_ch=1, img_size=28)
    x = torch.randn(4, 1, 28, 28)
    out = D(x)
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()  # raw logits, unbounded but finite


def test_discriminator_cifar_output_shape():
    D = Discriminator(img_ch=3, img_size=32)
    x = torch.randn(4, 3, 32, 32)
    out = D(x)
    assert out.shape == (4, 1)


def test_generator_discriminator_roundtrip():
    """A generator's output must be a valid discriminator input."""
    G = Generator(z_dim=100, img_ch=1, img_size=28)
    D = Discriminator(img_ch=1, img_size=28)
    z = torch.randn(2, 100, 1, 1)
    fake = G(z)
    score = D(fake)
    assert score.shape == (2, 1)
