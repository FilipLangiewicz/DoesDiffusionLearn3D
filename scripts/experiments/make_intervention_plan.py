"""Write the plan file of a causal test (input of dd3d.intervene on Kaggle).

The plan has these conditions for each selected test image:

- base: no change (the SDEdit reconstruction);
- shift:probe_r<r>: the shortest feature change that moves the probe output to
  the azimuth label + 90 deg, scaled so that the mean norm of a cell vector is
  r times the mean token norm of the block (from the stats of the extraction);
- shift:random_r<r>: a random change with the same norm in each cell as the
  largest probe shift;
- erase: the LEACE eraser of the azimuth (sin and cos) in this block.

The probe and the eraser come from the grid features of one block and one
timestep (train split). In the test mode (--test-job), a random probe and random
features replace them, so the job can be tested before the features exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dd3d.data import load_feature_dir
from dd3d.intervene import leace_fit, make_shift
from dd3d.metrics import encode_targets
from dd3d.probe import ProbeConfig, probe_direction, run


def mean_token_norm(features: Path | None) -> float:
    """Return the mean token norm of the block and timestep of a feature folder."""
    if features is None:
        return 1.0
    block = int(features.parent.name.split("_")[1])
    t = int(features.name.split("_")[1])
    stats = pd.concat(pd.read_csv(f) for f in sorted((features.parents[1] / "stats").glob("*.csv")))
    sel = stats[(stats["block"] == block) & (stats["t"] == t)]
    return float(sel["mean_token_norm"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="Folder block_<bb>/t_<ttt> with grid features.")
    parser.add_argument("--splits", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--split", default="test", help="Split of the images to change.")
    parser.add_argument("--n-images", type=int, default=60)
    parser.add_argument("--relative", default="0.1,0.3,1.0", help="Shift norm / mean token norm.")
    parser.add_argument("--token-norm", type=float, default=None, help="Default: from the stats files.")
    parser.add_argument("--rotate", type=float, default=90.0, help="Target azimuth = label + rotate.")
    parser.add_argument("--leace-rows", type=int, default=30000, help="Cell rows for the LEACE fit.")
    parser.add_argument("--channels", type=int, default=2240)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--test-job", action="store_true", help="Random probe and eraser (test of the job).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    splits = {n: pd.read_csv(args.splits / f"{n}.csv") for n in ("train", "val", args.split)}
    pick = splits[args.split].sample(min(args.n_images, len(splits[args.split])), random_state=args.seed)
    d = args.grid * args.grid * args.channels
    report = {"split": args.split, "n_images": len(pick), "rotate": args.rotate}

    if args.test_job:
        w = rng.normal(size=(2, d)) / np.sqrt(d)
        cells = rng.normal(size=(5000, args.channels))
        z = cells[:, :2] + 0.1 * rng.normal(size=(5000, 2))
        report["mode"] = "test of the job (random probe and eraser)"
    else:
        labels = {
            k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy())
            for k, v in splits.items()
        }
        feats = {k: load_feature_dir(args.features, v["image_id"].tolist()) for k, v in splits.items()}
        res = run(feats, labels, ProbeConfig())
        w = probe_direction(res.model)[:2]
        report["probe"] = {k: {"az_mae": v["az_mae"], "el_mae": v["el_mae"]} for k, v in res.splits.items()}
        # Each grid cell of each train image is one LEACE sample with the label of its image.
        x = feats["train"].reshape(len(feats["train"]), args.grid * args.grid, args.channels)
        z_img = encode_targets(*labels["train"])[:, :2]
        rows = rng.choice(
            x.shape[0] * x.shape[1], size=min(args.leace_rows, x.shape[0] * x.shape[1]), replace=False
        )
        cells = x.reshape(-1, args.channels)[rows]
        z = np.repeat(z_img, x.shape[1], axis=0)[rows]
        del feats, x
    eraser_mean, eraser_matrix = leace_fit(cells, z)

    az = pick["light_azimuth_deg"].to_numpy()
    now = encode_targets(az, pick["light_elevation_deg"].to_numpy())[:, :2]
    target = encode_targets(az + args.rotate, pick["light_elevation_deg"].to_numpy())[:, :2]
    unit = make_shift(w, target - now)
    token_norm = args.token_norm or mean_token_norm(args.features)
    cells_of = lambda v: v.reshape(len(v), args.grid * args.grid, args.channels)  # noqa: E731
    # One factor for each image, so that every image gets the same relative shift.
    unit = unit / np.linalg.norm(cells_of(unit), axis=2).mean(axis=1)[:, None] * token_norm
    conditions, shifts = ["base"], [np.zeros_like(unit)]
    rel = [float(v) for v in args.relative.split(",")]
    for r in rel:
        conditions.append(f"shift:probe_r{r:g}")
        shifts.append(r * unit)
    big = cells_of(max(rel) * unit)
    rand = cells_of(rng.normal(size=unit.shape))
    rand *= (np.linalg.norm(big, axis=2) / np.linalg.norm(rand, axis=2))[:, :, None]
    conditions.append(f"shift:random_r{max(rel):g}")
    shifts.append(rand.reshape(len(rand), -1))
    conditions.append("erase")
    shifts.append(np.zeros_like(unit))
    shifts = np.stack(shifts, axis=1).astype(np.float16)  # (n_img, n_cond, D)
    report["mean_token_norm"] = token_norm
    report["relative_shifts"] = rel
    report["conditions"] = conditions

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out / "plan.npz",
        image_ids=pick["image_id"].to_numpy().astype(str),
        shapes=pick["shape"].to_numpy().astype(str),
        azimuth=az,
        conditions=np.array(conditions),
        shifts=shifts,
        eraser_mean=eraser_mean,
        eraser_matrix=eraser_matrix,
    )
    (args.out / "plan.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
