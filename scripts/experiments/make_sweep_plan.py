"""Write a sweep plan: one shift for each target angle and strength (input of dd3d.intervene).

The sweep is for the interactive viewer. For each selected image, the shift
turns the light direction of the features to the label azimuth + r. This
repeats for every r in --rotations and every strength in --scales. The shift family is the same as in
make_intervention_plan.py:

- pattern: k times the mean feature change between images whose light differs
  by r. A regression of the features on sin and cos of the azimuth gives it.
- probe: the shortest feature change that turns the probe output by r, times k.

The condition names are shift:<family>_k<k>_r<r>, plus base.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from make_intervention_plan import pattern_of

from dd3d.intervene import make_shift
from dd3d.metrics import encode_targets
from dd3d.probe import ProbeConfig, probe_direction, run
from dd3d.store import FeatureStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Feature root (dd3d.store).")
    parser.add_argument("--key", default="sana/block_10/t_400")
    parser.add_argument("--splits", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--family", default="pattern", choices=("pattern", "probe"))
    parser.add_argument("--scales", default="1,2,4")
    parser.add_argument("--rotations", default="30,60,90,120,150,180,210,240,270,300,330")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    splits = {n: pd.read_csv(args.splits / f"{n}.csv") for n in ("train", "val", args.split)}
    sub = splits[args.split]
    # One image of each shape where possible, with azimuths far apart.
    per_shape = -(-args.n_images // sub["shape"].nunique())
    pick = pd.concat(
        g.sample(min(len(g), per_shape), random_state=args.seed) for _, g in sub.groupby("shape", sort=False)
    )
    pick = pick.head(args.n_images)
    store = FeatureStore(args.features)
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in splits.items()
    }
    feats = {k: store.load(args.key, v["image_id"].tolist()) for k, v in splits.items()}
    res = run(feats, labels, ProbeConfig())
    w = probe_direction(res.model)[:2]
    b = pattern_of(feats["train"], encode_targets(*labels["train"])[:, :2])
    before = res.model.predict(feats[args.split][sub.index.get_indexer(pick.index)])[:, :2]
    del feats

    az = pick["light_azimuth_deg"].to_numpy()
    el = pick["light_elevation_deg"].to_numpy()
    now = encode_targets(az, el)[:, :2]
    d = w.shape[1]
    conditions, shifts = ["base"], [np.zeros((len(pick), d))]
    for k in (float(v) for v in args.scales.split(",")):
        for r in (float(v) for v in args.rotations.split(",")):
            if args.family == "pattern":
                delta = (encode_targets(az + r, el)[:, :2] - now) @ b
            else:
                a = np.radians(r)
                turned = np.stack(
                    [
                        before[:, 0] * np.cos(a) + before[:, 1] * np.sin(a),
                        before[:, 1] * np.cos(a) - before[:, 0] * np.sin(a),
                    ],
                    axis=1,
                )
                delta = make_shift(w, turned - before)
            conditions.append(f"shift:{args.family}_k{k:g}_r{r:g}")
            shifts.append(k * delta)
    shifts = np.stack(shifts, axis=1).astype(np.float16)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out / "plan.npz",
        image_ids=pick["image_id"].to_numpy().astype(str),
        shapes=pick["shape"].to_numpy().astype(str),
        azimuth=az,
        conditions=np.array(conditions),
        shifts=shifts,
    )
    report = {
        "key": args.key,
        "family": args.family,
        "images": pick["image_id"].tolist(),
        "conditions": conditions,
    }
    (args.out / "plan.json").write_text(json.dumps(report, indent=1))
    print(f"{len(pick)} images, {len(conditions)} conditions -> {args.out}")


if __name__ == "__main__":
    main()
