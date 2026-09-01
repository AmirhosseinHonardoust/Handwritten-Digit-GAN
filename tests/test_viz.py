import torch

from viz import save_grid_image


def test_save_grid_image_writes_file(tmp_path):
    images = torch.randn(8, 1, 28, 28)
    outpath = tmp_path / "grid.png"
    save_grid_image(images, str(outpath), title="Test Grid", nrow=4)
    assert outpath.exists()
    assert outpath.stat().st_size > 0
