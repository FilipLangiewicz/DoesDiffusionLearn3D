"""Fit the light probe on every block and timestep of extraction runs.

The input is a feature root with one folder for each run (for example sana,
sana_random_init, sana_legacy_noise). The root can hold merged features, chunk
files or the tar files of the extraction jobs (see dd3d.store). Each run has a
meta.json file with the grid of each block.
The script also fits the probe on two baselines: the pixels of the image and
the VAE latent. Every probe uses the same splits, the same PCA size and an alpha
from the val split only.

The output is one CSV row for each run, block, timestep, grid, target and split.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import pandas as pd

from dd3d.data import add_scene_columns, pixel_features, pool_grid, target_columns
from dd3d.probe import ProbeConfig, run
from dd3d.store import FeatureStore


def load_splits(splits_dir: Path, scenes_dir: Path | None) -> dict[str, pd.DataFrame]:
    """Return all split tables, with the local label columns."""
    splits = {p.stem: pd.read_csv(p) for p in sorted(Path(splits_dir).glob("*.csv"))}
    if scenes_dir is not None:
        splits = {k: add_scene_columns(v, scenes_dir) for k, v in splits.items()}
    return splits


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
    parser.add_argument(
        "--extra-grids", default="1,2", help="Coarser grids for every block (1: the mean of all tokens)."
    )
    parser.add_argument("--blocks", default="", help="Comma list of blocks (empty: all).")
    parser.add_argument("--timesteps", default="", help="Comma list of timesteps (empty: all).")
    parser.add_argument("--no-pixels", action="store_true", help="Skip the pixel baseline.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    only_blocks = {int(b) for b in args.blocks.split(",") if b}
    only_t = {int(t) for t in args.timesteps.split(",") if t}
    out_csv = args.out / "scan.csv"
    done = set()
    rows: list[dict] = []
    if out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict("records")
        done = {(r["run"], r["block"], r["t"], r["grid"]) for r in rows}

    splits = load_splits(args.splits, args.scenes)
    targets = args.targets.split(",")
    ids = {k: v["image_id"].tolist() for k, v in splits.items()}

    store = FeatureStore(args.features) if args.features.exists() else None

    def load_all(key: str) -> dict | None:
        # A split without features (for example test_pos of a run that has no
        # extraction for it) is left out. Without train or val, the key is skipped.
        known = set(store.ids(key))
        if not (set(ids["train"]) <= known and set(ids["val"]) <= known):
            print(f"  skip {key}: the train or val split has images without features")
            return None
        out = {}
        for k, v in ids.items():
            if set(v) <= known:
                out[k] = store.load(key, v)
            else:
                print(f"  split {k} left out: no features for {key}")
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

    if not args.no_pixels and ("pixels", -1, -1, 64) not in done:
        report(
            {"run": "pixels", "block": -1, "t": -1, "grid": 64},
            {k: pixel_features(v, args.images, 64) for k, v in splits.items()},
        )
    runs = [] if store is None else (store.runs() if args.runs == "all" else args.runs.split(","))
    for run_name in runs:
        if run_name not in store.runs():
            print(f"skip {run_name}: no meta.json under {args.features}")
            continue
        meta = store.meta(run_name)
        vae_key = f"{run_name}/vae_latent"
        if store.has(vae_key) and ("vae_latent", -1, -1, 32) not in done:
            feats = load_all(vae_key)
            if feats is not None:
                report({"run": "vae_latent", "block": -1, "t": -1, "grid": 32}, feats)
        for b in meta["blocks"]:
            if only_blocks and b not in only_blocks:
                continue
            grid = meta["grid_of_block"][str(b)]
            for t in meta["timesteps"]:
                if only_t and t not in only_t:
                    continue
                key = f"{run_name}/block_{b:02d}/t_{t:03d}"
                extra = {g for g in (int(v) for v in args.extra_grids.split(",") if v) if g < grid}
                grids = sorted({grid, min(grid, args.scan_grid)} | extra)
                if all((run_name, b, t, g) in done for g in grids) or not store.has(key):
                    continue
                feats = load_all(key)
                if feats is None:
                    continue
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
