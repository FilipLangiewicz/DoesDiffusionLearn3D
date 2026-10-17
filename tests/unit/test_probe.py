import numpy as np
import pandas as pd

from dd3d.probe import ProbeConfig, probe_direction, run, spatial_contributions
from dd3d.select import best_rows


def make_data(seed: int = 0, n: int = 300, grid: int = 2, channels: int = 5):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 360, n)
    el = rng.uniform(30, 75, n)
    x = rng.normal(size=(n, grid * grid * channels))
    x[:, 0] += 6 * np.sin(np.radians(az))
    x[:, 1] += 6 * np.cos(np.radians(az))
    x[:, 2] += el / 20
    parts = {"train": slice(0, 200), "val": slice(200, 250), "test": slice(250, n)}
    feats = {k: x[s] for k, s in parts.items()}
    labels = {k: (az[s], el[s]) for k, s in parts.items()}
    return feats, labels


def test_the_test_labels_do_not_change_the_probe():
    feats, labels = make_data()
    cfg = ProbeConfig(n_components=10)
    a = run(feats, labels, cfg)
    shuffled = dict(labels)
    az, el = labels["test"]
    shuffled["test"] = (az[::-1].copy(), el[::-1].copy())
    b = run(feats, shuffled, cfg)
    assert a.alpha == b.alpha
    assert np.allclose(a.model.predict(feats["test"]), b.model.predict(feats["test"]))


def test_the_probe_reads_a_linear_code():
    feats, labels = make_data()
    res = run(feats, labels, ProbeConfig(n_components=10))
    assert res.splits["test"]["az_mae"] < 12.0


def test_probe_direction_is_the_linear_map_of_the_probe():
    feats, labels = make_data()
    model = run(feats, labels, ProbeConfig(n_components=10)).model
    w = probe_direction(model)
    x1, x2 = feats["test"][:5], feats["test"][5:10]
    assert np.allclose(model.predict(x1) - model.predict(x2), (x1 - x2) @ w.T, atol=1e-6)


def test_spatial_contributions_add_up_to_the_prediction():
    grid = 2
    feats, labels = make_data(grid=grid)
    model = run(feats, labels, ProbeConfig(n_components=10)).model
    x = feats["test"]
    maps = spatial_contributions(model, x, grid)
    offset = model.predict(x) - maps.sum(axis=(2, 3))
    # The rest is the same constant for every image: the intercept of the probe.
    assert np.allclose(offset, offset[0], atol=1e-6)


def test_best_rows_select_on_val_only():
    rows = []
    for block, val_err, test_err in ((1, 5.0, 50.0), (2, 9.0, 1.0)):
        for split, err in (("val", val_err), ("test", test_err)):
            rows.append(
                {
                    "run": "sana",
                    "block": block,
                    "t": 100,
                    "grid": 4,
                    "split": split,
                    "az_mae": err,
                    "el_mae": 0.0,
                }
            )
    best = best_rows(pd.DataFrame(rows))
    assert set(best["block"]) == {1}
