"""Integration test against the real build_dataset() / main() code path.

Unlike test_train_step.py, this does NOT monkeypatch build_dataset with a
synthetic dataset — it calls the actual torchvision MNIST loader, which
downloads the dataset on first use. This is the only test that exercises
the real data-loading code a user's CLI invocation actually runs.

Marked `slow` and skipped by default (see pyproject.toml addopts) since it
needs network access on a cold cache. Run explicitly with:
    pytest -m slow -v
CI runs it via a dedicated step (network access available there).
If the download fails (offline environment, mirror down), the test skips
cleanly instead of failing the suite.
"""

from urllib.error import URLError

import pytest
import torch
from torch.utils.data import Subset

import train_gan


@pytest.mark.slow
def test_integration_train_real_mnist_subset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    try:
        full_ds, img_ch, img_size = train_gan.build_dataset("mnist")
    except (RuntimeError, URLError, OSError) as exc:
        pytest.skip(f"MNIST dataset unavailable (no network?): {exc}")

    # Keep the run fast: real data, small slice of it.
    subset = Subset(full_ds, range(64))
    monkeypatch.setattr(train_gan, "build_dataset", lambda dataset: (subset, img_ch, img_size))
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_gan.py",
            "--dataset",
            "mnist",
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--z-dim",
            "10",
            "--outdir",
            "outputs",
        ],
    )

    train_gan.main()

    outdir = tmp_path / "outputs"
    assert (outdir / "G_last.pth").exists()
    assert (outdir / "D_last.pth").exists()
    assert (outdir / "samples" / "epoch_001.png").exists()

    # Checkpoint must load back and produce a valid-shaped sample from real data.
    state = torch.load(outdir / "G_last.pth", weights_only=True)
    G2 = train_gan.Generator(z_dim=10, img_ch=img_ch, img_size=img_size)
    G2.load_state_dict(state)
    G2.eval()
    with torch.no_grad():
        out = G2(torch.randn(2, 10, 1, 1))
    assert out.shape == (2, img_ch, img_size, img_size)
