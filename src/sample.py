"""Generate random samples and a latent-space interpolation from a trained Generator.

Usage:
    python src/sample.py --model outputs/G_last.pth --dataset mnist --outdir outputs
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import utils as vutils

from train_gan import Generator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        help="Path to saved Generator state_dict (e.g., outputs/G_last.pth)",
    )
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--z-dim", type=int, default=100)
    ap.add_argument("--outdir", type=str, default="outputs")
    return ap.parse_args(argv)


def load_generator(model_path: str, z_dim: int, img_ch: int, img_size: int) -> Generator:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(z_dim=z_dim, img_ch=img_ch, img_size=img_size).to(dev)
    # weights_only=True restricts unpickling to tensors/known types, avoiding
    # arbitrary code execution from an untrusted checkpoint file.
    state = torch.load(model_path, map_location=dev, weights_only=True)
    G.load_state_dict(state)
    G.eval()
    return G


def save_random_grid(G: Generator, z_dim: int, outdir: str) -> None:
    dev = next(G.parameters()).device
    z = torch.randn(64, z_dim, 1, 1, device=dev)
    with torch.no_grad():
        samples = G(z).cpu()
    grid = vutils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title("Random Samples")
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "samples_grid.png"), dpi=160)
    plt.close()


def save_interpolation(G: Generator, z_dim: int, outdir: str) -> None:
    dev = next(G.parameters()).device
    z1 = torch.randn(1, z_dim, 1, 1, device=dev)
    z2 = torch.randn(1, z_dim, 1, 1, device=dev)
    alphas = torch.linspace(0, 1, 10, device=dev).view(-1, 1, 1, 1)
    z_interp = (1 - alphas) * z1 + alphas * z2
    with torch.no_grad():
        imgs = G(z_interp).cpu()
    grid = vutils.make_grid(imgs, nrow=10, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(12, 2))
    plt.axis("off")
    plt.title("Latent Interpolation")
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "interpolation.png"), dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    img_ch, img_size = (1, 28) if args.dataset == "mnist" else (3, 32)

    G = load_generator(args.model, args.z_dim, img_ch, img_size)
    save_random_grid(G, args.z_dim, args.outdir)
    save_interpolation(G, args.z_dim, args.outdir)
    print("[OK] Samples and interpolation saved.")


if __name__ == "__main__":
    main()
