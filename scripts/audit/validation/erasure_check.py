"""Audit check: does the LEACE eraser of the causal test remove what the 8 x 8 probe reads?

The plan fits one eraser on cell vectors, with the cells of all positions as
samples. LEACE then removes the linear information that is the same at every
position. A probe on the 8 x 8 grid has its own weights for each cell. Thus
it can still read a part of the azimuth after the eraser. The eraser is affine,
so the eraser of each token gives the same cell means as the eraser of each
cell mean. This script fits new probes on the erased cell features and
compares them with probes on the original features.

Output: erasure.csv and erasure.md in --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dd3d.data import pool_grid
from dd3d.probe import ProbeConfig, run
from dd3d.store import FeatureStore


def erase(x: np.ndarray, mean: np.ndarray, a: np.ndarray, grid: int, channels: int) -> np.ndarray:
    """Apply r(v) = v - A (v - mean) to every cell vector of the rows of x."""
    cells = x.reshape(len(x), grid * grid, channels).astype(np.float32)
    out = cells - (cells - mean) @ a.T
    return out.reshape(len(x), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Feature root (see dd3d.store).")
    parser.add_argument("--item", action="append", required=True, help="key=plan_file, one for each block.")
    parser.add_argument("--splits", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--channels", type=int, default=2240)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    splits = {n: pd.read_csv(args.splits / f"{n}.csv") for n in ("train", "val", "test", "test_cubes")}
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in splits.items()
    }
    store = FeatureStore(args.features)
    rows = []
    for item in args.item:
        key, plan_path = item.split("=")
        plan = np.load(plan_path, allow_pickle=False)
        mean = plan["eraser_mean"].astype(np.float32)
        a = plan["eraser_matrix"].astype(np.float32)
        feats = {k: store.load(key, v["image_id"].tolist()).astype(np.float32) for k, v in splits.items()}
        erased = {k: erase(x, mean, a, args.grid, args.channels) for k, x in feats.items()}
        variants = {
            "original, 8x8": feats,
            "erased, 8x8": erased,
            "original, 1x1": {k: pool_grid(x, args.grid, 1) for k, x in feats.items()},
            "erased, 1x1": {k: pool_grid(x, args.grid, 1) for k, x in erased.items()},
        }
        for name, f in variants.items():
            res = run(f, labels, ProbeConfig())
            for split in ("test", "test_cubes"):
                s = res.splits[split]
                rows.append(
                    {
                        "key": key,
                        "features": name,
                        "split": split,
                        "az_mae": s["az_mae"],
                        "az_ci_low": s["az_mae_ci"][0],
                        "az_ci_high": s["az_mae_ci"][1],
                    }
                )
                print(f"{key} {name:14s} {split:10s} az MAE {s['az_mae']:6.2f}", flush=True)
        del feats, erased, variants
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "erasure.csv", index=False, float_format="%.3f")
    (args.out / "erasure.md").write_text(df.to_markdown(index=False, floatfmt=".2f") + "\n")


if __name__ == "__main__":
    main()
