<div align="center">

# Handwritten Digit GAN

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DCGAN-orange)
![Dataset](https://img.shields.io/badge/Dataset-MNIST%20%2F%20CIFAR--10-green)
![Coverage](https://img.shields.io/badge/Coverage-90%25%2B%20enforced-red)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Handwritten-Digit-GAN/actions/workflows/ci.yml/badge.svg)](https://github.com/AmirhosseinHonardoust/Handwritten-Digit-GAN/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

A PyTorch implementation of a **Deep Convolutional GAN (DCGAN)** trained on MNIST (with an optional CIFAR-10 path), using an installable **Generator/Discriminator package**, a **training CLI**, a **sampling and latent-interpolation CLI**, **loss-curve visualization**, and a **tested, linted, type-checked codebase with CI**.

> **Note:** GAN training is stochastic. Image quality depends on epoch count, batch size, and hardware, and results will vary between runs. The samples below come from a single training run and are meant to illustrate the pipeline end to end, not to represent a state-of-the-art benchmark.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Sampling and Latent Interpolation](#sampling-and-latent-interpolation)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Recommendations](#recommendations)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

A Generative Adversarial Network learns to produce realistic data by pitting two networks against each other. This project builds and trains a DCGAN from scratch to learn the distribution of handwritten digits (MNIST):

- the **Generator** learns to turn random noise into fake digit images
- the **Discriminator** learns to tell real MNIST digits apart from the Generator's fakes
- the two networks train adversarially until the Generator produces convincing, human-like digits

The project is packaged as an installable module with dedicated CLIs for training and sampling, a shared visualization helper, unit tests with enforced coverage, and a CI workflow that lints, type-checks, and tests every push.

---

## Key Features

- **DCGAN architecture** in PyTorch (transposed-convolution Generator, convolutional Discriminator)
- **Trainable on MNIST** by default, with a **CIFAR-10** path built in
- **Installable CLIs**, `digit-gan-train` and `digit-gan-sample`, via `pip install -e .`
- **Checkpointing**, Generator/Discriminator weights saved as `.pth` state dicts
- **Safe checkpoint loading** with `weights_only=True` to avoid arbitrary code execution from untrusted files
- **Random sample generation** from a trained Generator
- **Latent-space interpolation** between two random points in z-space
- **Training curve visualization** (Generator vs. Discriminator loss)
- **Per-epoch sample grids** saved automatically during training
- **Unit tests and GitHub Actions CI** with lint, format, type-check, and coverage gates

---

## System Workflow

```text
Random noise (z ~ N(0, 1))
        ↓
Generator (transposed convolutions)
        ↓
Fake digit image
        ↓
Discriminator (convolutions) ← Real MNIST digit
        ↓
Real / fake logit
        ↓
Adversarial loss (BCEWithLogitsLoss) → updates G and D
        ↓
Per-epoch sample grid + loss tracking
        ↓
Saved checkpoints (G_last.pth / D_last.pth)
        ↓
Sampling CLI: random grid + latent interpolation
```

---

## Project Structure

```text
Handwritten-Digit-GAN/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/                     # MNIST/CIFAR-10 dataset (auto-downloaded, gitignored)
│
├── outputs/                  # Saved models, images, and plots (generated, gitignored)
│   ├── samples/              # Generated sample grid per epoch
│   ├── G_last.pth            # Generator weights
│   ├── D_last.pth            # Discriminator weights
│   ├── training_curves.png
│   ├── samples_grid.png
│   └── interpolation.png
│
├── src/
│   └── digit_gan/            # Installable package (pip install -e .)
│       ├── __init__.py
│       ├── models.py         # Generator / Discriminator
│       ├── data.py           # Dataset loading + device selection
│       ├── train.py          # Training CLI (digit-gan-train)
│       ├── sample.py         # Sampling CLI (digit-gan-sample)
│       └── viz.py            # Shared image-grid plotting helper
│
├── tests/
│   ├── test_build_dataset.py
│   ├── test_integration_mnist.py
│   ├── test_models.py
│   ├── test_sample.py
│   ├── test_train_step.py
│   └── test_viz.py
│
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── requirements-lock.txt
```

`data/` and `outputs/` are generated locally by the commands below and are not committed to the repository (see `.gitignore`).

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Handwritten-Digit-GAN.git
cd Handwritten-Digit-GAN
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Installing in editable mode registers the `digit-gan-train` and `digit-gan-sample` CLIs. For development tools (Ruff, Black, mypy, pytest):

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

Train the GAN:

```bash
digit-gan-train --dataset mnist --epochs 20 --batch-size 128 --z-dim 100 --outdir outputs
```

Generate samples and a latent interpolation from a trained Generator:

```bash
digit-gan-sample --model outputs/G_last.pth --dataset mnist --outdir outputs
```

Both CLIs are also runnable without installing the package:

```bash
python -m digit_gan.train --dataset mnist --epochs 20
python -m digit_gan.sample --model outputs/G_last.pth --dataset mnist
```

---

## Training

`digit-gan-train` builds the dataset, trains the Generator and Discriminator adversarially with `BCEWithLogitsLoss` and Adam (`betas=(0.5, 0.999)`), saves a sample grid after every epoch, and writes the loss-curve plot and final checkpoints when training finishes.

<div align="center">

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `mnist` | `mnist` or `cifar10` |
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `128` | Training batch size |
| `--z-dim` | `100` | Latent noise vector dimension |
| `--lr` | `2e-4` | Adam learning rate for both networks |
| `--outdir` | `outputs` | Output directory for checkpoints, samples, and plots |
| `--seed` | `42` | Random seed for reproducibility |
| `--num-workers` | `2` | DataLoader worker processes (use `0` on constrained/CI machines) |

</div>

Generated outputs include:

```text
outputs/samples/epoch_001.png ... epoch_NNN.png
outputs/training_curves.png
outputs/G_last.pth
outputs/D_last.pth
```

The Generator uses transposed convolutions with batch normalization and ReLU activations, ending in `Tanh`; the Discriminator uses strided convolutions with LeakyReLU and outputs a raw logit (no sigmoid), paired with `BCEWithLogitsLoss` for numerically stable training.

---

## Sampling and Latent Interpolation

`digit-gan-sample` loads a trained Generator checkpoint with `torch.load(..., weights_only=True)`, restricting deserialization to tensors and known types so an untrusted checkpoint cannot execute arbitrary code, and produces two outputs:

- `samples_grid.png`, 64 digits generated from independent random noise vectors
- `interpolation.png`, a 10-step walk between two random points in latent space, showing how the Generator smoothly morphs one digit into another

```bash
digit-gan-sample --model outputs/G_last.pth --dataset mnist --z-dim 100 --outdir outputs
```

---

## Visual Reports

### Generated digits

<div align="center">

| Random Samples | Latent Interpolation |
|---|---|
| ![Random samples](https://github.com/user-attachments/assets/5842d3b0-3b50-4c56-9870-0d0b75fae511) | ![Latent interpolation](https://github.com/user-attachments/assets/bb2dd21d-b159-4588-b45c-d84e95c3cf28) |
| **Analysis:** 64 digits generated from independent random noise vectors after training. Sharp, varied, recognizable digits indicate the Generator has learned the MNIST manifold rather than memorizing or mode-collapsing onto a few shapes. | **Analysis:** A 10-step walk between two random points in latent space. A smooth, gradual morph from one digit into another (rather than an abrupt jump) shows the latent space is continuous and meaningfully structured, not just noise the Generator memorized. |

</div>

<details>
<summary>Additional training curves chart</summary>

<div align="center">

![Training curves](https://github.com/user-attachments/assets/69849337-ab41-4d84-8976-fc0ebefd457d)

The training curves chart tracks Generator and Discriminator loss across every iteration. Neither loss collapsing to zero nor diverging to infinity is the goal, healthy adversarial training shows both losses oscillating and roughly balancing each other as the two networks compete.

</div>

</details>

---

## Testing and CI

Run unit tests locally:

```bash
pytest -v --cov=src --cov-report=term-missing
```

Compile, lint, format-check, and type-check:

```bash
ruff check --select E,F,I,B,SIM,UP src/ tests/
black --check src/ tests/
mypy src/
```

The fast suite runs entirely on synthetic data (no network access) with **90%+ coverage enforced**. A separate, network-dependent suite exercises the real MNIST download path and is marked `slow`:

```bash
pytest -v -m slow
```

CI runs the fast suite, lint, format, and type checks on every push and pull request, and runs the `slow` suite automatically as well. It is defined in:

```text
.github/workflows/ci.yml
```

Optionally, install [pre-commit](https://pre-commit.com) hooks to run Ruff/Black/mypy automatically before each commit:

```bash
pip install pre-commit
pre-commit install
```

---

## Code Quality

The project separates responsibilities across modules:

<div align="center">

| Module | Purpose |
|---|---|
| `src/digit_gan/models.py` | Generator and Discriminator architectures (DCGAN) |
| `src/digit_gan/data.py` | Dataset loading (MNIST/CIFAR-10) and device selection |
| `src/digit_gan/train.py` | Training CLI, adversarial training loop, checkpoints, loss curves |
| `src/digit_gan/sample.py` | Sampling CLI, safe checkpoint loading, random grid, latent interpolation |
| `src/digit_gan/viz.py` | Shared image-grid plotting helper used by both CLIs |

</div>

Tooling is configured through `pyproject.toml` (Ruff, Black, mypy, pytest, coverage) and `requirements-dev.txt`. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## Recommendations

Potential next steps for higher-quality results:

- Train for more epochs (50–100) for higher quality images
- Try one-sided label smoothing for additional training stability
- Experiment with CIFAR-10 for color image generation
- On a machine with few CPU cores, pass `--num-workers 0` to avoid DataLoader worker-count warnings

---

## Tech Stack

- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- tqdm
- Pillow
- pytest / pytest-cov
- Ruff / Black / mypy
- GitHub Actions

---

## Author

**AmirhosseinHonardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE) for details.
