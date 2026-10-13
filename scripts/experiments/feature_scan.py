"""Fit the light probe on every block and timestep of extraction runs.

The input is a feature root with one folder for each run (for example sana,
sana_random_init, sana_legacy_noise). Each run folder has merged features in
block_<bb>/t_<ttt>/features.npy and a meta.json file with the grid of each block.
The script also fits the probe on two baselines: the pixels of the image and
the VAE latent. Every probe uses the same splits, the same PCA size and an alpha
from the val split only.

The output is one CSV row for each run, block, timestep, grid, target and split.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dd3d.data import add_scene_columns, load_feature_dir, pixel_features, target_columns
from dd3d.probe import ProbeConfig, run


def load_splits(splits_dir: Path, scenes_dir: Path | None) -> dict[str, pd.DataFrame]:
    """Return all split tables, with the local label columns."""
    splits = {p.stem: pd.read_csv(p) for p in sorted(Path(splits_dir).glob("*.csv"))}
    if scenes_dir is not None:
        splits = {k: add_scene_columns(v, scenes_dir) for k, v in splits.items()}
    return splits


def pool_grid(x: np.ndarray, grid: int, to: int) -> np.ndarray:
    """Average a (n, grid * grid * C) feature matrix to a smaller grid."""
    n = len(x)
    k = grid // to
    c = x.shape[1] // (grid * grid)
    return x.reshape(n, to, k, to, k, c).mean(axis=(2, 4)).reshape(n, -1)


def probe_rows(feats: dict, splits: dict, targets: list[str], info: dict, n_components: int) -> list[dict]:
    rows = []
    for target in targets:
        az_col, el_col = target_columns(target)
        labels = {k: (v[az_col].to_numpy(), v[el_col].to_numpy()) for k, v in splits.items() if k in feats}
        res = run(feats, labels, ProbeConfig(n_components=n_components))
        for split, s in res.splits.items():
            rows.append(
                {
                    **info,
                    "target": target,
                    "split": split,
                    "alpha": res.alpha,
                    "az_mae": s["az_mae"],
                    "az_ci_low": s["az_mae_ci"][0],
                    "az_ci_high": s["az_mae_ci"][1],
                    "el_mae": s["el_mae"],
                    "el_ci_low": s["el_mae_ci"][0],
                    "el_ci_high": s["el_mae_ci"][1],
                    "n": s["n"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Feature root with one folder per run.")
    parser.add_argument("--runs", default="sana,sana_random_init,sana_legacy_noise")
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--scenes", type=Path, default=None, help="Scene JSON folder (v1 only).")
    parser.add_argument("--images", type=Path, default=Path("data/final/images"))
    parser.add_argument("--targets", default="global")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument(
        "--scan-grid", type=int, default=4, help="Grid of the block scan (finer grids pool to it)."
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "scan.csv"
    done = set()
    rows: list[dict] = []
    if out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict("records")
        done = {(r["run"], r["block"], r["t"], r["grid"]) for r in rows}

    splits = load_splits(args.splits, args.scenes)
    targets = args.targets.split(",")
    ids = {k: v["image_id"].tolist() for k, v in splits.items()}

    def load_all(d: Path) -> dict:
        # A split without features in this folder (for example test_pos before its
        # extraction) is left out, and the train and val splits must be complete.
        known = set((d / "ids.txt").read_text().split())
        out = {}
        for k, v in ids.items():
            if set(v) <= known:
                out[k] = load_feature_dir(d, v)
            elif k in ("train", "val"):
                raise KeyError(f"split {k} has images without features in {d}")
            else:
                print(f"  split {k} left out: no features in {d}")
        return out

    def save(new_rows):
        rows.extend(new_rows)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    def report(info, feats):
        t0 = time.time()
        new = probe_rows(feats, splits, targets, info, args.n_components)
        cubes = [r for r in new if r["split"] == "test_cubes" and r["target"] == targets[0]]
        msg = f"cubes az {cubes[0]['az_mae']:6.2f} el {cubes[0]['el_mae']:5.2f}" if cubes else ""
        print(f"{info} {msg} ({time.time() - t0:.0f} s)", flush=True)
        save(new)

    if ("pixels", -1, -1, 64) not in done:
        report(
            {"run": "pixels", "block": -1, "t": -1, "grid": 64},
            {k: pixel_features(v, args.images, 64) for k, v in splits.items()},
        )
    for run_name in args.runs.split(","):
        root = args.features / run_name
        if not root.exists():
            print(f"skip {run_name}: no folder {root}")
            continue
        meta = json.loads((root / "meta.json").read_text())
        vae_dir = root / "vae_latent"
        if (vae_dir / "features.npy").exists() and ("vae_latent", -1, -1, 32) not in done:
            feats = load_all(vae_dir)
            report({"run": "vae_latent", "block": -1, "t": -1, "grid": 32}, feats)
        for b in meta["blocks"]:
            grid = meta["grid_of_block"][str(b)]
            for t in meta["timesteps"]:
                d = root / f"block_{b:02d}" / f"t_{t:03d}"
                grids = sorted({grid, min(grid, args.scan_grid)})
                if all((run_name, b, t, g) in done for g in grids):
                    continue
                feats = load_all(d)
                for g in grids:
                    if (run_name, b, t, g) in done:
                        continue
                    fg = feats if g == grid else {k: pool_grid(x, grid, g) for k, x in feats.items()}
                    report({"run": run_name, "block": b, "t": t, "grid": g}, fg)
                del feats
                gc.collect()
    print(f"-> {out_csv}")


if __name__ == "__main__":
    main()
