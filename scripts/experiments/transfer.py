"""Fit a probe on one data set and test it on other data sets (transfer test).

Two questions use this test:

- Position: fit on v2 (object at the centre), test on v2r (random position).
  A representation of the light that does not depend on the position transfers.
- Cue: fit on v2r, test on v2rns (no object shadow) and v2rso (only the shadow).
  The paired scenes differ in one cue, so the error shows which cue the probe uses.

The fit uses the train split of the fit set and selects alpha on its val split.
The test uses the test and test_cubes splits of each test set.
The same rows run for the pixels, so each model has its pixel control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dd3d.data import load_feature_dir, pixel_features
from dd3d.probe import ProbeConfig, run


def splits_of(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(data_dir / "splits" / f"{n}.csv") for n in ("train", "val", "test", "test_cubes")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, help="Data set name of the fit, for example v2.")
    parser.add_argument("--tests", required=True, help="Comma list of test data sets, for example v2r.")
    parser.add_argument(
        "--features", default="", help="Comma list of run/block_<bb>/t_<ttt> (empty: only pixels)."
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fit_dir = Path("data") / args.fit
    fit_splits = splits_of(fit_dir)
    tests = {name: splits_of(Path("data") / name) for name in args.tests.split(",")}

    def frames():
        out = {"train": fit_splits["train"], "val": fit_splits["val"]}
        out[f"{args.fit}:test"] = fit_splits["test"]
        out[f"{args.fit}:test_cubes"] = fit_splits["test_cubes"]
        for name, sp in tests.items():
            out[f"{name}:test"] = sp["test"]
            out[f"{name}:test_cubes"] = sp["test_cubes"]
        return out

    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy())
        for k, v in frames().items()
    }

    def set_of(key: str) -> str:
        return args.fit if key in ("train", "val") else key.split(":")[0]

    rows = []
    sources = ["pixels"] + [f for f in args.features.split(",") if f]
    for src in sources:
        feats = {}
        for key, df in frames().items():
            name = set_of(key)
            if src == "pixels":
                feats[key] = pixel_features(df, Path("data") / name / "images", 64)
            else:
                feats[key] = load_feature_dir(Path("data/features") / name / src, df["image_id"].tolist())
        res = run(feats, labels, ProbeConfig())
        for key, s in res.splits.items():
            if key in ("train", "val"):
                continue
            rows.append(
                {
                    "features": src,
                    "fit": args.fit,
                    "test": key,
                    "az_mae": s["az_mae"],
                    "az_ci_low": s["az_mae_ci"][0],
                    "az_ci_high": s["az_mae_ci"][1],
                    "el_mae": s["el_mae"],
                }
            )
        print(src, {r["test"]: round(r["az_mae"], 1) for r in rows if r["features"] == src}, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "transfer.csv", index=False)
    (args.out / "transfer.md").write_text(df.to_markdown(index=False, floatfmt=".1f") + "\n")
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=1))


if __name__ == "__main__":
    main()
