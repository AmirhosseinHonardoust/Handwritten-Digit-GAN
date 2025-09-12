import argparse, os, torch
from torchvision import utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from train_gan import Generator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to saved Generator state_dict (e.g., outputs/G_last.pth)")
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist","cifar10"])
    ap.add_argument("--z-dim", type=int, default=100)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    img_ch, img_size = (1,28) if args.dataset=="mnist" else (3,32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(z_dim=args.z_dim, img_ch=img_ch, img_size=img_size).to(device)
    state = torch.load(args.model, map_location=device)
    G.load_state_dict(state); G.eval()

    z = torch.randn(64, args.z_dim, 1, 1, device=device)
    with torch.no_grad():
        samples = G(z).cpu()
    grid = vutils.make_grid(samples, nrow=8, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(6,6)); plt.axis("off"); plt.title("Random Samples")
    plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "samples_grid.png"), dpi=160); plt.close()

    z1 = torch.randn(1, args.z_dim, 1, 1, device=device)
    z2 = torch.randn(1, args.z_dim, 1, 1, device=device)
    alphas = torch.linspace(0,1,10, device=device).view(-1,1,1,1)
    z_interp = (1 - alphas) * z1 + alphas * z2
    with torch.no_grad():
        imgs = G(z_interp).cpu()
    grid2 = vutils.make_grid(imgs, nrow=10, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(12,2)); plt.axis("off"); plt.title("Latent Interpolation")
    plt.imshow(np.transpose(grid2.numpy(), (1,2,0)))
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "interpolation.png"), dpi=160); plt.close()
    print("[OK] Samples and interpolation saved.")

if __name__ == "__main__":
    main()
