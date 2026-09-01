import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import train_gan
from train_gan import (
    Discriminator,
    Generator,
    device,
    parse_args,
    save_checkpoints,
    train_one_epoch,
)


def _tiny_loader(batch_size: int = 8, n: int = 16) -> DataLoader:
    images = torch.randn(n, 1, 28, 28)
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)


def test_train_one_epoch_runs_and_updates_weights():
    dev = device()
    G = Generator(z_dim=10, img_ch=1, img_size=28).to(dev)
    D = Discriminator(img_ch=1, img_size=28).to(dev)
    crit = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loader = _tiny_loader()

    before = [p.clone() for p in G.parameters()]
    g_losses, d_losses = train_one_epoch(
        G, D, loader, crit, optG, optD, dev, z_dim=10, epoch=1, epochs=1
    )
    after = list(G.parameters())

    assert len(g_losses) == len(d_losses) == len(loader)
    assert all(torch.isfinite(torch.tensor(v)) for v in g_losses + d_losses)
    # Generator weights must have moved after a training step.
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    outdir = str(tmp_path)
    G = Generator(z_dim=10, img_ch=1, img_size=28)
    D = Discriminator(img_ch=1, img_size=28)
    save_checkpoints(G, D, outdir)

    assert os.path.exists(os.path.join(outdir, "G_last.pth"))
    assert os.path.exists(os.path.join(outdir, "D_last.pth"))

    state = torch.load(os.path.join(outdir, "G_last.pth"), weights_only=True)
    G2 = Generator(z_dim=10, img_ch=1, img_size=28)
    G2.load_state_dict(state)  # must not raise
    for p1, p2 in zip(G.parameters(), G2.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.dataset == "mnist"
    assert args.epochs == 20
    assert args.batch_size == 128
    assert args.z_dim == 100
    assert args.seed == 42
    assert args.num_workers == 2


def test_parse_args_overrides():
    args = parse_args(
        ["--dataset", "cifar10", "--epochs", "2", "--batch-size", "16", "--num-workers", "0"]
    )
    assert args.dataset == "cifar10"
    assert args.epochs == 2
    assert args.batch_size == 16
    assert args.num_workers == 0


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--epochs", "0"],
        ["--epochs", "-1"],
        ["--batch-size", "0"],
        ["--batch-size", "-4"],
        ["--z-dim", "0"],
        ["--lr", "0"],
        ["--lr", "-0.1"],
        ["--num-workers", "-1"],
    ],
)
def test_parse_args_rejects_invalid_values(bad_args):
    with pytest.raises(SystemExit):
        parse_args(bad_args)


def test_train_one_epoch_cifar10_shapes_and_updates():
    """The training step must also work for the 3-channel/32x32 CIFAR-10 branch."""
    dev = device()
    G = Generator(z_dim=10, img_ch=3, img_size=32).to(dev)
    D = Discriminator(img_ch=3, img_size=32).to(dev)
    crit = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    images = torch.randn(16, 3, 32, 32)
    labels = torch.zeros(16, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=True)

    before = [p.clone() for p in D.parameters()]
    g_losses, d_losses = train_one_epoch(
        G, D, loader, crit, optG, optD, dev, z_dim=10, epoch=1, epochs=1
    )
    after = list(D.parameters())

    assert len(g_losses) == len(d_losses) == len(loader)
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_main_end_to_end_cli(tmp_path, monkeypatch):
    """Run the real CLI entrypoint end-to-end with a synthetic dataset (no download)."""
    monkeypatch.chdir(tmp_path)

    def fake_build_dataset(dataset: str):
        images = torch.randn(24, 1, 28, 28)
        labels = torch.zeros(24, dtype=torch.long)
        return TensorDataset(images, labels), 1, 28

    monkeypatch.setattr(train_gan, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_gan.py",
            "--dataset",
            "mnist",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--z-dim",
            "10",
            "--outdir",
            "outputs",
        ],
    )

    train_gan.main()

    outdir = tmp_path / "outputs"
    assert (outdir / "G_last.pth").exists()
    assert (outdir / "D_last.pth").exists()
    assert (outdir / "training_curves.png").exists()
    assert (outdir / "samples" / "epoch_001.png").exists()

    # Checkpoint must be loadable back into a matching model.
    state = torch.load(outdir / "G_last.pth", weights_only=True)
    G2 = Generator(z_dim=10, img_ch=1, img_size=28)
    G2.load_state_dict(state)  # must not raise
