# Handwritten Digit GAN

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
├─ data/                # MNIST dataset (auto-downloaded)
├─ outputs/             # Saved models, images, and plots
│  ├─ samples/          # Generated samples per epoch
│  ├─ G_last.pth        # Generator weights
│  ├─ D_last.pth        # Discriminator weights
│  ├─ training_curves.png
│  ├─ samples_grid.png
│  └─ interpolation.png
├─ src/
│  ├─ train_gan.py      # Training script
│  └─ sample.py         # Generate samples from trained model
└─ README.md
```

---

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Train the GAN
```bash
python src/train_gan.py --dataset mnist --epochs 20 --batch-size 128 --z-dim 100 --outdir outputs
```

---

## Generate Samples
```bash
python src/sample.py --model outputs/G_last.pth --dataset mnist --outdir outputs
```

---

## Recommendations
- Train for more epochs (50–100) for higher quality images.
- Try Label Smoothing or alternative loss functions (e.g., BCEWithLogitsLoss).
- Experiment with CIFAR-10 for color image generation.
