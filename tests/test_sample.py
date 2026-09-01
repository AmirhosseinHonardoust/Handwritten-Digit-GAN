import os
import sys

import torch

from digit_gan import sample
from digit_gan import train as train_gan
from digit_gan.models import Discriminator, Generator


def _make_checkpoint(tmp_path, z_dim=10, img_ch=1, img_size=28):
    G = Generator(z_dim=z_dim, img_ch=img_ch, img_size=img_size)
    path = tmp_path / "G_test.pth"
    torch.save(G.state_dict(), path)
    return str(path)


def test_parse_args_requires_model():
    args = sample.parse_args(["--model", "outputs/G_last.pth"])
    assert args.model == "outputs/G_last.pth"
    assert args.dataset == "mnist"
    assert args.z_dim == 100


def test_load_generator_roundtrip(tmp_path):
    ckpt_path = _make_checkpoint(tmp_path, z_dim=10, img_ch=1, img_size=28)
    G = sample.load_generator(ckpt_path, z_dim=10, img_ch=1, img_size=28)
    assert isinstance(G, Generator)
    assert not G.training  # load_generator() must call .eval()
    z = torch.randn(2, 10, 1, 1)
    with torch.no_grad():
        out = G(z)
    assert out.shape == (2, 1, 28, 28)


def test_save_random_grid_writes_file(tmp_path):
    G = Generator(z_dim=10, img_ch=1, img_size=28)
    G.eval()
    sample.save_random_grid(G, z_dim=10, outdir=str(tmp_path))
    assert (tmp_path / "samples_grid.png").exists()


def test_save_interpolation_writes_file(tmp_path):
    G = Generator(z_dim=10, img_ch=1, img_size=28)
    G.eval()
    sample.save_interpolation(G, z_dim=10, outdir=str(tmp_path))
    assert (tmp_path / "interpolation.png").exists()


def test_sample_main_end_to_end(tmp_path, monkeypatch):
    """Run the real sample.py CLI entrypoint end-to-end against a saved checkpoint."""
    ckpt_path = _make_checkpoint(tmp_path, z_dim=10, img_ch=1, img_size=28)
    outdir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample.py",
            "--model",
            ckpt_path,
            "--dataset",
            "mnist",
            "--z-dim",
            "10",
            "--outdir",
            str(outdir),
        ],
    )
    sample.main()

    assert (outdir / "samples_grid.png").exists()
    assert (outdir / "interpolation.png").exists()


def test_sample_main_loads_original_repo_checkpoint_format(tmp_path, monkeypatch):
    """Regression guard: sample.py must keep loading plain state_dict checkpoints
    (the format every checkpoint saved by save_checkpoints() uses), not just
    checkpoints saved in this test file's own format."""
    G = Generator(z_dim=100, img_ch=1, img_size=28)
    D = Discriminator(img_ch=1, img_size=28)
    train_gan.save_checkpoints(G, D, str(tmp_path))

    outdir = tmp_path / "out2"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample.py",
            "--model",
            os.path.join(str(tmp_path), "G_last.pth"),
            "--dataset",
            "mnist",
            "--outdir",
            str(outdir),
        ],
    )
    sample.main()
    assert (outdir / "samples_grid.png").exists()
