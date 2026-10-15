"""Fit a probe on one data set and test it on other data sets (transfer test).

Two questions use this test:

- Position: fit on v2 (object at the centre), test on v2r (random position).
  A representation of the light that does not depend on the position transfers.
- Cue: fit on v2r, test on v2rns (no object shadow) and v2rso (only the shadow).
  The paired scenes differ in one cue, so the error shows which cue the probe uses.

The fit uses the train split of the fit set and selects alpha on its val split.
The test uses the test and test_cubes splits of each test set. The pixels run
first, so each model has its pixel control. With --scan, the val split of the
fit set selects the block, the timestep and the grid of each run in --runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dd3d.data import pixel_features, pool_grid
from dd3d.probe import ProbeConfig, run
from dd3d.select import feature_key, selected
from dd3d.store import FeatureStore


def splits_of(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(data_dir / "splits" / f"{n}.csv") for n in ("train", "val", "test", "test_cubes")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, help="Data set name of the fit, for example v2.")
    parser.add_argument("--tests", required=True, help="Comma list of test data sets, for example v2r.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--feature-root", type=Path, default=Path("data/features"))
    parser.add_argument(
        "--features",
        default="",
        help="Comma list of run/block_<bb>/t_<ttt>[@grid] keys (@1: mean of all tokens).",
    )
    parser.add_argument(
        "--pooled-grids",
        default="",
        help="Also fit each run of --runs at its best block and t for these grids.",
    )
    parser.add_argument("--scan", type=Path, default=None, help="scan.csv of the fit set (selection).")
    parser.add_argument("--runs", default="", help="Comma list of runs to select from --scan.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fit_splits = splits_of(args.data_root / args.fit)
    tests = {name: splits_of(args.data_root / name) for name in args.tests.split(",")}
    frames = {"train": fit_splits["train"], "val": fit_splits["val"]}
    frames[f"{args.fit}:test"] = fit_splits["test"]
    frames[f"{args.fit}:test_cubes"] = fit_splits["test_cubes"]
    for name, sp in tests.items():
        frames[f"{name}:test"] = sp["test"]
        frames[f"{name}:test_cubes"] = sp["test_cubes"]
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in frames.items()
    }

    def set_of(key: str) -> str:
        return args.fit if key in ("train", "val") else key.split(":")[0]

    stores: dict[str, FeatureStore] = {}

    def store(name: str) -> FeatureStore:
        if name not in stores:
            stores[name] = FeatureStore(args.feature_root / name)
        return stores[name]

    # Each source is (label, key, grid); key None means the pixels.
    sources: list[tuple[str, str | None, int | None]] = [("pixels", None, None)]
    for item in (f for f in args.features.split(",") if f):
        key, _, g = item.partition("@")
        sources.append((item, key, int(g) if g else None))
    if args.scan is not None and args.runs:
        sel = selected(pd.read_csv(args.scan))
        for _, row in sel[sel["run"].isin(args.runs.split(","))].iterrows():
            key = feature_key(row["run"], row["block"], row["t"])
            sources.append((f"{key}/grid_{int(row['grid'])}", key, int(row["grid"])))
        # The best block and timestep of each pooled grid, also selected on the val split.
        scan = pd.read_csv(args.scan)
        val = scan[(scan["split"] == "val") & (scan["target"] == "global")]
        for g in (int(v) for v in args.pooled_grids.split(",") if v):
            for run_name in args.runs.split(","):
                v = val[(val["run"] == run_name) & (val["grid"] == g)]
                if v.empty:
                    continue
                r = v.loc[(v["az_mae"] + v["el_mae"]).idxmin()]
                key = feature_key(run_name, r["block"], r["t"])
                sources.append((f"{key}/grid_{g}", key, g))

    sources = list(dict.fromkeys(sources))
    rows = []
    for label, key, grid in sources:
        feats = {}
        for split, df in frames.items():
            name = set_of(split)
            if key is None:
                feats[split] = pixel_features(df, args.data_root / name / "images", 64)
                continue
            s = store(name)
            x = s.load(key, df["image_id"].tolist())
            native = s.meta(key.split("/")[0])["grid_of_block"][str(int(key.split("/")[1].split("_")[1]))]
            feats[split] = pool_grid(x, native, grid or native)
        res = run(feats, labels, ProbeConfig())
        for split, s in res.splits.items():
            if split in ("train", "val"):
                continue
            rows.append(
                {
                    "features": label,
                    "fit": args.fit,
                    "test": split,
                    "az_mae": s["az_mae"],
                    "az_ci_low": s["az_mae_ci"][0],
                    "az_ci_high": s["az_mae_ci"][1],
                    "el_mae": s["el_mae"],
                }
            )
        print(label, {r["test"]: round(r["az_mae"], 1) for r in rows if r["features"] == label}, flush=True)
        del feats
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "transfer.csv", index=False)
    (args.out / "transfer.md").write_text(df.to_markdown(index=False, floatfmt=".1f") + "\n")
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=1))


if __name__ == "__main__":
    main()
