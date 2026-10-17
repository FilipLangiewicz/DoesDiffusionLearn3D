import numpy as np
import torch

from dd3d.extract import pool_tokens
from dd3d.intervene import (
    apply_cell_erasers,
    cell_erasers,
    leace_factors,
    leace_fit,
    make_shift,
    spread_cells,
    token_cells,
)


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


def test_leace_factors_give_the_same_eraser():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(500, 2))
    x = rng.normal(size=(500, 20)) + z @ rng.normal(size=(2, 20))
    _, a = leace_fit(x, z)
    mean, left, right = leace_factors(x, z)
    assert np.allclose(left @ right.T, a, atol=1e-4)
    erased = x - (x - mean) @ (left @ right.T).T
    cross = (erased - erased.mean(0)).T @ (z - z.mean(0)) / len(x)
    assert np.abs(cross).max() < 1e-4


def test_cell_erasers_remove_a_code_that_depends_on_the_position():
    # Cell 0 and cell 1 carry the azimuth with opposite signs, so the mean of the
    # cells carries nothing and one pooled eraser cannot see the code.
    rng = np.random.default_rng(3)
    n, cells, c = 800, 4, 6
    z = rng.normal(size=(n, 2))
    sign = np.array([1.0, -1.0, 1.0, -1.0])
    x = rng.normal(size=(n, cells, c))
    x[:, :, :2] += 2.0 * sign[None, :, None] * z[:, None, :]
    mean, left, right = cell_erasers(x, z)
    erased = apply_cell_erasers(x, mean, left, right)
    flat = erased.reshape(n, -1)
    cross = (flat - flat.mean(0)).T @ (z - z.mean(0)) / n
    assert np.abs(cross).max() < 1e-4
    # The pooled eraser leaves the code: a linear fit on all cells still finds z.
    pm, pa = leace_fit(x.reshape(-1, c), np.repeat(z, cells, axis=0))
    pooled = (x - (x - pm) @ pa.T).reshape(n, -1)
    coef, *_ = np.linalg.lstsq(pooled - pooled.mean(0), z - z.mean(0), rcond=None)
    fit = (pooled - pooled.mean(0)) @ coef
    assert np.corrcoef(fit[:, 0], z[:, 0])[0, 1] > 0.9


def test_cell_erasers_of_shuffled_labels_keep_the_code():
    # The control eraser removes a subspace of the same rank, but the code of z stays.
    rng = np.random.default_rng(4)
    n, cells, c = 800, 4, 6
    z = rng.normal(size=(n, 2))
    x = rng.normal(size=(n, cells, c))
    x[:, :, :2] += 2.0 * z[:, None, :]
    erased = apply_cell_erasers(x, *cell_erasers(x, rng.permutation(z))).reshape(n, -1)
    coef, *_ = np.linalg.lstsq(erased - erased.mean(0), z - z.mean(0), rcond=None)
    fit = (erased - erased.mean(0)) @ coef
    assert np.corrcoef(fit[:, 0], z[:, 0])[0, 1] > 0.9


def test_token_cells_follow_the_token_order():
    cells = token_cells(2, 4).reshape(4, 4)
    assert (cells[:2, :2] == 0).all() and (cells[:2, 2:] == 1).all() and (cells[2:, 2:] == 3).all()


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
