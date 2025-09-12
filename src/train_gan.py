import argparse, os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_ch=1, img_size=28):
        super().__init__()
        self.img_ch = img_ch
        self.img_size = img_size
        base = 64
        if img_size == 28:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(z_dim, base*4, 7, 1, 0, bias=False),
                nn.BatchNorm2d(base*4), nn.ReLU(True),
                nn.ConvTranspose2d(base*4, base*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base*2), nn.ReLU(True),
                nn.ConvTranspose2d(base*2, img_ch, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        else:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(z_dim, base*4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(base*4), nn.ReLU(True),
                nn.ConvTranspose2d(base*4, base*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base*2), nn.ReLU(True),
                nn.ConvTranspose2d(base*2, base, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base), nn.ReLU(True),
                nn.ConvTranspose2d(base, img_ch, 4, 2, 1, bias=False),
                nn.Tanh()
            )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_ch=1, img_size=28):
        super().__init__()
        base = 64
        if img_size == 28:
            self.net = nn.Sequential(
                nn.Conv2d(img_ch, base, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base, base*2, 4, 2, 1, bias=False), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear((base*2)*7*7, 1),
                nn.Sigmoid()
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(img_ch, base, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base, base*2, 4, 2, 1, bias=False), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base*2, base*4, 4, 2, 1, bias=False), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear((base*4)*4*4, 1),
                nn.Sigmoid()
            )
    def forward(self, x):
        return self.net(x)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_loss_curves(G_losses, D_losses, outpath):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(G_losses, label="G_loss")
    ax.plot(D_losses, label="D_loss")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss"); ax.set_title("Training Curves")
    ax.legend(); fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist","cifar10"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--z-dim", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "samples"), exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.dataset == "mnist":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
        img_ch, img_size = 1, 28
    else:
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
        img_ch, img_size = 3, 32
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    dev = device()
    G = Generator(z_dim=args.z_dim, img_ch=img_ch, img_size=img_size).to(dev)
    D = Discriminator(img_ch=img_ch, img_size=img_size).to(dev)

    crit = nn.BCELoss()
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=dev)
    G_losses, D_losses = [], []

    for epoch in range(1, args.epochs+1):
        for i, (real, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")):
            real = real.to(dev)
            bsz = real.size(0)
            valid = torch.ones(bsz, 1, device=dev)
            fake_label = torch.zeros(bsz, 1, device=dev)

            # Train D
            noise = torch.randn(bsz, args.z_dim, 1, 1, device=dev)
            fake = G(noise).detach()
            D_real = D(real); loss_real = crit(D_real, valid)
            D_fake = D(fake); loss_fake = crit(D_fake, fake_label)
            loss_D = (loss_real + loss_fake) * 0.5
            optD.zero_grad(); loss_D.backward(); optD.step()

            # Train G
            noise2 = torch.randn(bsz, args.z_dim, 1, 1, device=dev)
            gen = G(noise2)
            out = D(gen)
            loss_G = crit(out, valid)
            optG.zero_grad(); loss_G.backward(); optG.step()

            G_losses.append(loss_G.item()); D_losses.append(loss_D.item())

        with torch.no_grad():
            samples = G(fixed_noise).cpu()
            grid = vutils.make_grid(samples, nrow=8, normalize=True, value_range=(-1,1))
            plt.figure(figsize=(6,6)); plt.axis("off"); plt.title(f"Epoch {epoch}")
            plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "samples", f"epoch_{epoch:03d}.png"), dpi=160); plt.close()

    save_loss_curves(G_losses, D_losses, os.path.join(args.outdir, "training_curves.png"))
    torch.save(G.state_dict(), os.path.join(args.outdir, "G_last.pth"))
    torch.save(D.state_dict(), os.path.join(args.outdir, "D_last.pth"))
    print("[OK] Training finished. Models & samples saved.")

if __name__ == "__main__":
    main()
