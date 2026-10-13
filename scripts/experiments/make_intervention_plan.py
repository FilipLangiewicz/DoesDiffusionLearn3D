"""Write the plan file of a causal test (input of dd3d.intervene on Kaggle).

The plan has these conditions for each selected test image:

- base: no change (the SDEdit reconstruction);
- shift:probe_x<s>: the shortest feature change that moves the probe output to
  the azimuth label + 90 deg, times the scale s;
- shift:random_x<s>: a random change with the same norm as shift:probe_x<s>;
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="Folder block_<bb>/t_<ttt> with grid features.")
    parser.add_argument("--splits", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--split", default="test", help="Split of the images to change.")
    parser.add_argument("--n-images", type=int, default=60)
    parser.add_argument("--scales", default="1,4,16")
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
    conditions, shifts = ["base"], [np.zeros_like(unit)]
    for s in [float(v) for v in args.scales.split(",")]:
        conditions.append(f"shift:probe_x{s:g}")
        shifts.append(s * unit)
    s_max = max(float(v) for v in args.scales.split(","))
    rand = rng.normal(size=unit.shape)
    rand *= (np.linalg.norm(s_max * unit, axis=1) / np.linalg.norm(rand, axis=1))[:, None]
    conditions.append(f"shift:random_x{s_max:g}")
    shifts.append(rand)
    conditions.append("erase")
    shifts.append(np.zeros_like(unit))
    shifts = np.stack(shifts, axis=1).astype(np.float16)  # (n_img, n_cond, D)
    report["shift_norm_x1"] = float(np.linalg.norm(unit, axis=1).mean())
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
