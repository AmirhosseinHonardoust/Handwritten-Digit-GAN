"""Train a DCGAN on MNIST (default) or CIFAR-10.

Usage:
    digit-gan-train --dataset mnist --epochs 20 --batch-size 128 \
        --z-dim 100 --outdir outputs
    # or: python -m digit_gan.train --dataset mnist ...
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_dataset, device
from .models import Discriminator, Generator
from .viz import save_grid_image


def save_loss_curves(G_losses: list[float], D_losses: list[float], outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(G_losses, label="G_loss")
    ax.plot(D_losses, label="D_loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def train_one_epoch(
    G: Generator,
    D: Discriminator,
    loader: DataLoader,
    crit: nn.Module,
    optG: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    dev: torch.device,
    z_dim: int,
    epoch: int,
    epochs: int,
) -> tuple[list[float], list[float]]:
    """Run one training epoch. Returns (G_losses, D_losses) for this epoch."""
    G_losses: list[float] = []
    D_losses: list[float] = []
    for real, _ in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
        real = real.to(dev)
        bsz = real.size(0)
        valid = torch.ones(bsz, 1, device=dev)
        fake_label = torch.zeros(bsz, 1, device=dev)

        # Train D
        noise = torch.randn(bsz, z_dim, 1, 1, device=dev)
        fake = G(noise).detach()
        D_real = D(real)
        loss_real = crit(D_real, valid)
        D_fake = D(fake)
        loss_fake = crit(D_fake, fake_label)
        loss_D = (loss_real + loss_fake) * 0.5
        optD.zero_grad()
        loss_D.backward()
        optD.step()

        # Train G
        noise2 = torch.randn(bsz, z_dim, 1, 1, device=dev)
        gen = G(noise2)
        out = D(gen)
        loss_G = crit(out, valid)
        optG.zero_grad()
        loss_G.backward()
        optG.step()

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())
    return G_losses, D_losses


def save_epoch_samples(G: Generator, fixed_noise: torch.Tensor, outdir: str, epoch: int) -> None:
    with torch.no_grad():
        samples = G(fixed_noise).cpu()
    outpath = os.path.join(outdir, "samples", f"epoch_{epoch:03d}.png")
    save_grid_image(samples, outpath, title=f"Epoch {epoch}", nrow=8)


def save_checkpoints(G: Generator, D: Discriminator, outdir: str) -> None:
    torch.save(G.state_dict(), os.path.join(outdir, "G_last.pth"))
    torch.save(D.state_dict(), os.path.join(outdir, "D_last.pth"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--z-dim", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker processes (default: 2; use 0 on constrained/CI machines)",
    )
    args = ap.parse_args(argv)

    if args.epochs < 1:
        ap.error(f"--epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        ap.error(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.z_dim < 1:
        ap.error(f"--z-dim must be >= 1, got {args.z_dim}")
    if args.lr <= 0:
        ap.error(f"--lr must be > 0, got {args.lr}")
    if args.num_workers < 0:
        ap.error(f"--num-workers must be >= 0, got {args.num_workers}")

    return args


def main() -> None:
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "samples"), exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds, img_ch, img_size = build_dataset(args.dataset)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dev = device()
    G = Generator(z_dim=args.z_dim, img_ch=img_ch, img_size=img_size).to(dev)
    D = Discriminator(img_ch=img_ch, img_size=img_size).to(dev)

    # BCEWithLogitsLoss combines Sigmoid + BCELoss in one numerically stable op;
    # Discriminator outputs raw logits (no Sigmoid) to pair with it.
    crit = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=dev)
    all_G_losses: list[float] = []
    all_D_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        g_losses, d_losses = train_one_epoch(
            G, D, loader, crit, optG, optD, dev, args.z_dim, epoch, args.epochs
        )
        all_G_losses.extend(g_losses)
        all_D_losses.extend(d_losses)
        save_epoch_samples(G, fixed_noise, args.outdir, epoch)

    save_loss_curves(all_G_losses, all_D_losses, os.path.join(args.outdir, "training_curves.png"))
    save_checkpoints(G, D, args.outdir)
    print("[OK] Training finished. Models & samples saved.")


if __name__ == "__main__":  # pragma: no cover
    main()
