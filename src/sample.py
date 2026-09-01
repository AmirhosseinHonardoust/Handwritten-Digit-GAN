"""Generate random samples and a latent-space interpolation from a trained Generator.

Usage:
    python src/sample.py --model outputs/G_last.pth --dataset mnist --outdir outputs
"""

from __future__ import annotations

import argparse
import os

import torch

from train_gan import Generator
from viz import save_grid_image


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
    outpath = os.path.join(outdir, "samples_grid.png")
    save_grid_image(samples, outpath, title="Random Samples", nrow=8)


def save_interpolation(G: Generator, z_dim: int, outdir: str) -> None:
    dev = next(G.parameters()).device
    z1 = torch.randn(1, z_dim, 1, 1, device=dev)
    z2 = torch.randn(1, z_dim, 1, 1, device=dev)
    alphas = torch.linspace(0, 1, 10, device=dev).view(-1, 1, 1, 1)
    z_interp = (1 - alphas) * z1 + alphas * z2
    with torch.no_grad():
        imgs = G(z_interp).cpu()
    outpath = os.path.join(outdir, "interpolation.png")
    save_grid_image(imgs, outpath, title="Latent Interpolation", nrow=10, figsize=(12, 2))


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    img_ch, img_size = (1, 28) if args.dataset == "mnist" else (3, 32)

    G = load_generator(args.model, args.z_dim, img_ch, img_size)
    save_random_grid(G, args.z_dim, args.outdir)
    save_interpolation(G, args.z_dim, args.outdir)
    print("[OK] Samples and interpolation saved.")


if __name__ == "__main__":  # pragma: no cover
    main()
