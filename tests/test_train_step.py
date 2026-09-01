import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    crit = nn.BCELoss()
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


def test_parse_args_overrides():
    args = parse_args(["--dataset", "cifar10", "--epochs", "2", "--batch-size", "16"])
    assert args.dataset == "cifar10"
    assert args.epochs == 2
    assert args.batch_size == 16
