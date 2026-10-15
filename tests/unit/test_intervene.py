import numpy as np
import torch

from dd3d.extract import pool_tokens
from dd3d.intervene import leace_fit, make_shift, spread_cells


def test_leace_removes_linear_covariance():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2000, 2))
    x = rng.normal(size=(2000, 12)) + z @ rng.normal(size=(2, 12))
    mean, a = leace_fit(x, z)
    erased = x - (x - mean) @ a.T
    cross = (erased - erased.mean(0)).T @ (z - z.mean(0)) / len(x)
    assert np.abs(cross).max() < 1e-4
    # The eraser changes only a small subspace: the rank of A equals the rank of cov(x, z).
    assert np.linalg.matrix_rank(a, tol=1e-4) == 2


def test_make_shift_reaches_the_target():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(2, 50))
    delta_y = rng.normal(size=(5, 2))
    d = make_shift(w, delta_y)
    assert np.allclose(d @ w.T, delta_y)


def test_spread_cells_keeps_the_cell_mean():
    grid, side, c = 4, 16, 3
    delta = torch.randn(2, grid * grid * c)
    tokens = spread_cells(delta, grid, side)
    assert tokens.shape == (2, side * side, c)
    for b in range(2):
        pooled = pool_tokens(tokens[b : b + 1], grid)
        assert np.allclose(pooled, delta[b].numpy(), atol=1e-2)


def test_cell_mask_to_tokens_follows_the_token_order():
    from dd3d.intervene import cell_mask_to_tokens

    cells = np.zeros((2, 2), dtype=bool)
    cells[0, 1] = True  # top right cell
    tokens = cell_mask_to_tokens(cells, 4).reshape(4, 4)
    assert tokens[:2, 2:].all() and tokens.sum() == 4
