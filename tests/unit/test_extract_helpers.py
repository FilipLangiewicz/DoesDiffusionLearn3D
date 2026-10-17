from pathlib import Path

import numpy as np
import torch

from dd3d.extract import list_images, pool_tokens, task_slice


def test_pool_tokens_constant_input():
    out = pool_tokens(torch.ones(1, 1024, 6), 8)
    assert out.shape == (8 * 8 * 6,)
    assert out.dtype == np.float16
    assert np.all(out == 1)


def row_index_tokens(channels=2):
    rows = torch.arange(32, dtype=torch.float32).repeat_interleave(32)
    return rows[None, :, None].repeat(1, 1, channels)


def test_pool_tokens_averages_blocks():
    out = pool_tokens(row_index_tokens(), 8).reshape(8, 8, 2)
    expected = torch.arange(32, dtype=torch.float32).reshape(8, 4).mean(dim=1).numpy()
    for col in range(8):
        np.testing.assert_allclose(out[:, col, 0], expected)
        np.testing.assert_allclose(out[:, col, 1], expected)


def test_pool_tokens_channel_first_layout_matches():
    tokens = torch.randn(1, 1024, 5)
    image_layout = tokens.transpose(1, 2).reshape(1, 5, 32, 32)
    np.testing.assert_array_equal(pool_tokens(tokens, 8), pool_tokens(image_layout, 8))


def test_task_slice_covers_all_items_once():
    items = list(range(10))
    parts = [task_slice(items, k, 3) for k in range(3)]
    assert sorted(sum(parts, [])) == items


def test_list_images_order(repo_root):
    paths = list_images(repo_root / "data" / "final_example" / "images", 2)
    assert len(paths) == 6
    assert [p.parent.name for p in paths] == ["Sphere"] * 2 + ["SmoothCylinder"] * 2 + ["SmoothCube_v2"] * 2
    assert all(isinstance(p, Path) and p.suffix == ".png" for p in paths)
    assert paths[0].name < paths[1].name


def test_pool_patch_tokens_drops_the_class_token_and_keeps_the_cell_order():
    from dd3d.extract_vision import pool_patch_tokens

    side, c = 4, 2
    tokens = torch.zeros(1, 1 + side * side, c)
    tokens[0, 0] = 100.0  # class token: must not enter the mean
    grid_vals = torch.arange(side * side, dtype=torch.float32).reshape(side, side)
    tokens[0, 1:, 0] = grid_vals.reshape(-1)
    pooled = pool_patch_tokens(tokens, 2).reshape(2, 2, c)
    expected = grid_vals.reshape(2, 2, 2, 2).mean(dim=(1, 3)).numpy()
    assert np.allclose(pooled[..., 0], expected)
    assert np.allclose(pooled[..., 1], 0.0)
