# Handwritten Digit GAN

[![CI](https://github.com/AmirhosseinHonardoust/Handwritten-Digit-GAN/actions/workflows/ci.yml/badge.svg)](https://github.com/AmirhosseinHonardoust/Handwritten-Digit-GAN/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch implementation of a simple Deep Convolutional GAN (DCGAN) trained on the MNIST dataset to generate realistic handwritten digits. The project includes training scripts, sample generation, interpolation in the latent space, and visualizations of training dynamics.

This project builds and trains a GAN from scratch to learn the distribution of handwritten digits (MNIST). The **Generator** learns to produce fake digit images from random noise, while the **Discriminator** learns to distinguish between real and fake samples. After training, the model can generate entirely new, human-like handwritten digits.

---

## Features
- DCGAN architecture in PyTorch
- Trainable on MNIST (can extend to CIFAR-10)
- Save & load generator/discriminator weights
- Random sample generation
- Latent space interpolation
- Training curve visualization (Generator vs Discriminator losses)

---

## Results

### Latent Interpolation
Shows smooth transitions between digits when interpolating in latent space:

<img width="1920" height="320" alt="interpolation" src="https://github.com/user-attachments/assets/bb2dd21d-b159-4588-b45c-d84e95c3cf28" />

---

### Random Samples
Generated handwritten digits from random noise:

<img width="960" height="960" alt="samples_grid" src="https://github.com/user-attachments/assets/5842d3b0-3b50-4c56-9870-0d0b75fae511" />

---

### Training Curves
Generator and Discriminator losses during training:

<img width="960" height="640" alt="training_curves" src="https://github.com/user-attachments/assets/69849337-ab41-4d84-8976-fc0ebefd457d" />

---

## Project Structure
```
GAN-MNIST/
├─ data/                # MNIST dataset (auto-downloaded, gitignored)
├─ outputs/             # Saved models, images, and plots (generated, gitignored)
│  ├─ samples/          # Generated samples per epoch
│  ├─ G_last.pth        # Generator weights
│  ├─ D_last.pth        # Discriminator weights
│  ├─ training_curves.png
│  ├─ samples_grid.png
│  └─ interpolation.png
├─ src/
│  └─ digit_gan/        # Installable package (pip install -e .)
│     ├─ models.py      # Generator / Discriminator
│     ├─ data.py        # Dataset loading + device selection
│     ├─ train.py       # Training CLI (digit-gan-train)
│     ├─ sample.py      # Sampling CLI (digit-gan-sample)
│     └─ viz.py         # Shared image-grid plotting helper
├─ tests/                # Unit tests (pytest)
├─ .github/workflows/    # CI (lint, type-check, tests)
└─ README.md
```
`data/` and `outputs/` are generated locally by the scripts below and are not
committed to the repository (see `.gitignore`).

---

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps   # installs the `digit-gan-train` / `digit-gan-sample` CLIs
```

---

## Train the GAN
```bash
digit-gan-train --dataset mnist --epochs 20 --batch-size 128 --z-dim 100 --outdir outputs
# or, without installing the CLI: python -m digit_gan.train --dataset mnist ...
```

---

## Generate Samples
```bash
digit-gan-sample --model outputs/G_last.pth --dataset mnist --outdir outputs
# or: python -m digit_gan.sample --model outputs/G_last.pth --dataset mnist --outdir outputs
```

---

## Recommendations
- Train for more epochs (50–100) for higher quality images.
- Try one-sided label smoothing for additional training stability.
- Experiment with CIFAR-10 for color image generation.
- On a machine with few CPU cores, pass `--num-workers 0` to `train_gan.py` to
  avoid DataLoader worker-count warnings.

---

## Development

Install dev tooling and run the local quality gate (matches CI):
```bash
pip install -r requirements-dev.txt
pip install -e . --no-deps
ruff check --select E,F,I,B,SIM,UP src/ tests/
black --check src/ tests/
mypy src/
pytest -v --cov=src --cov-report=term-missing
```
The fast suite runs on synthetic data (no network, 90%+ coverage enforced).
A separate suite exercises the real MNIST download path; run it with
`pytest -v -m slow` (needs network) — CI runs it on every push automatically.

Optionally, install [pre-commit](https://pre-commit.com) hooks to run
ruff/black/mypy automatically before each commit:
```bash
pip install pre-commit
pre-commit install
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
