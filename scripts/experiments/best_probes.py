"""Fit the selected probe of each feature source again and compare the sources.

The input is the scan of feature_scan.py for one data set. For each run, the
val split selects the block, the timestep and the grid (dd3d.select). Then the
script does three things:

1. It fits each selected probe again and writes the prediction for each image
   of each split (predictions.csv).
2. It compares each source with the pixels, the VAE latent and the random-weight
   SANA on the same images. The paired bootstrap gives the mean difference of
   the azimuth error with a 95% interval (compare.csv).
3. It fits each probe on shuffled train and val labels (control task). The
   selectivity is the control error minus the real error (control.csv).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dd3d.data import pixel_features, pool_grid
from dd3d.metrics import decode_azimuth, paired_bootstrap_diff
from dd3d.probe import ProbeConfig, run
from dd3d.select import feature_key, selected
from dd3d.store import FeatureStore

REFERENCES = ("pixels", "vae_latent", "sana_random_init")


def load_source(row, store: FeatureStore, splits: dict, images: Path) -> dict | None:
    """Return the features of one selected source for every split that has them."""
    if row["run"] == "pixels":
        return {k: pixel_features(v, images, 64) for k, v in splits.items()}
    if row["run"] == "vae_latent":
        keys = [k for k in store.keys() if k.endswith("/vae_latent")]
        key, grid = (keys[0] if keys else None), None
    else:
        key = feature_key(row["run"], row["block"], row["t"])
        grid = store.meta(row["run"])["grid_of_block"][str(int(row["block"]))]
    if key is None or not store.has(key):
        return None
    known = set(store.ids(key))
    out = {}
    for k, v in splits.items():
        ids = v["image_id"].tolist()
        if set(ids) <= known:
            x = store.load(key, ids)
            out[k] = x if grid is None else pool_grid(x, grid, int(row["grid"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True, help="scan.csv of feature_scan.py")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--shuffles", type=int, default=3, help="Number of control fits for each source.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    splits = {p.stem: pd.read_csv(p) for p in sorted(args.splits.glob("*.csv"))}
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in splits.items()
    }
    sel = selected(pd.read_csv(args.scan))
    sel.to_csv(args.out / "selected.csv", index=False)
    store = FeatureStore(args.features)
    rng = np.random.default_rng(0)

    preds, errors, controls = [], {}, []
    for _, row in sel.iterrows():
        feats = load_source(row, store, splits, args.images)
        if feats is None or "train" not in feats or "val" not in feats:
            print(f"skip {row['run']}: no features")
            continue
        lab = {k: labels[k] for k in feats}
        res = run(feats, lab, ProbeConfig())
        name = row["run"]
        for split, x in feats.items():
            p = res.model.predict(x)
            az_err, el_err = res.errors[split]
            errors[(name, split)] = az_err
            preds.append(
                pd.DataFrame(
                    {
                        "source": name,
                        "split": split,
                        "image_id": splits[split]["image_id"].to_numpy(),
                        "az_true": labels[split][0],
                        "az_pred": decode_azimuth(p[:, 0], p[:, 1]),
                        "az_err": az_err,
                        "el_true": labels[split][1],
                        "el_pred": p[:, 2],
                        "el_err": el_err,
                    }
                )
            )
        # Control task: the same probe on shuffled labels. Each shuffle keeps the
        # pairs (azimuth, elevation) together, so only the link to the image breaks.
        for s in range(args.shuffles):
            shuf = dict(lab)
            for k in ("train", "val"):
                perm = rng.permutation(len(lab[k][0]))
                shuf[k] = (lab[k][0][perm], lab[k][1][perm])
            ctrl = run({k: feats[k] for k in feats}, shuf, ProbeConfig())
            for split in feats:
                if split in ("train", "val"):
                    continue
                c_err = (ctrl.errors[split][0]).mean()
                controls.append(
                    {
                        "source": name,
                        "split": split,
                        "shuffle": s,
                        "real_az_mae": float(errors[(name, split)].mean()),
                        "control_az_mae": float(c_err),
                        "selectivity": float(c_err - errors[(name, split)].mean()),
                    }
                )
        print(name, {k: round(float(v["az_mae"]), 2) for k, v in res.splits.items()}, flush=True)
        del feats

    pd.concat(preds).to_csv(args.out / "predictions.csv", index=False, float_format="%.3f")
    pd.DataFrame(controls).to_csv(args.out / "control.csv", index=False, float_format="%.3f")

    rows = []
    sources = sorted({s for s, _ in errors})
    for name in sources:
        for ref in REFERENCES:
            if ref == name:
                continue
            for split in sorted(splits):
                if split in ("train", "val") or (name, split) not in errors or (ref, split) not in errors:
                    continue
                mean, lo, hi = paired_bootstrap_diff(errors[(name, split)], errors[(ref, split)])
                rows.append(
                    {
                        "source": name,
                        "reference": ref,
                        "split": split,
                        "source_az_mae": float(errors[(name, split)].mean()),
                        "reference_az_mae": float(errors[(ref, split)].mean()),
                        "diff": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "better": bool(hi < 0),
                    }
                )
    comp = pd.DataFrame(rows)
    comp.to_csv(args.out / "compare.csv", index=False, float_format="%.3f")
    (args.out / "compare.md").write_text(comp.to_markdown(index=False, floatfmt=".2f") + "\n")
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=1))
    print(comp.round(2).to_string())


if __name__ == "__main__":
    main()
