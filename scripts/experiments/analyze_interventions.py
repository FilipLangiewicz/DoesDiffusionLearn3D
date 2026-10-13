"""Measure the light direction in the output images of a causal test.

A pixel probe, fitted on the renders of the data set, predicts the azimuth of
each output image. For each condition the script reports:

- rotation: the signed azimuth change against the base condition (the shift
  aims at +rotate deg). A causal direction gives a rotation near the target;
- label error: the azimuth error against the label of the input render;
- pixel change: the mean absolute pixel difference against the base image.

All means have 95% bootstrap intervals over the images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dd3d.data import pixel_features
from dd3d.metrics import bootstrap_ci, circular_abs_error, decode_azimuth
from dd3d.probe import ProbeConfig, run


def signed_diff(a, b) -> np.ndarray:
    """Return a - b in degrees, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs", type=Path, required=True, help="Output folder of the job (one folder per condition)."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--images", type=Path, default=Path("data/v2/images"))
    parser.add_argument("--rotate", type=float, default=90.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    splits = {n: pd.read_csv(args.splits / f"{n}.csv") for n in ("train", "val")}
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in splits.items()
    }
    feats = {k: pixel_features(v, args.images, 64) for k, v in splits.items()}
    probe = run(feats, labels, ProbeConfig()).model

    plan = np.load(args.plan, allow_pickle=False)
    ids = [str(s) for s in plan["image_ids"]]
    shapes = [str(s) for s in plan["shapes"]]
    az_label = plan["azimuth"]
    conditions = [str(c) for c in plan["conditions"]]
    frame = pd.DataFrame({"image_id": ids, "shape": shapes})

    preds, pixels = {}, {}
    for cond in conditions:
        folder = args.outputs / cond.replace(":", "_")
        x = pixel_features(frame, folder, 64)
        p = probe.predict(x)
        preds[cond] = decode_azimuth(p[:, 0], p[:, 1])
        pixels[cond] = np.stack(
            [
                np.asarray(Image.open(folder / s / f"{i}.png").convert("RGB"), dtype=np.float32)
                for i, s in zip(ids, shapes, strict=True)
            ]
        )

    rows = []
    for cond in conditions:
        rot = signed_diff(preds[cond], preds["base"])
        err = circular_abs_error(preds[cond], az_label)
        change = np.abs(pixels[cond] - pixels["base"]).mean(axis=(1, 2, 3))
        rows.append(
            {
                "condition": cond,
                "rotation_mean": float(rot.mean()),
                "rotation_ci": bootstrap_ci(rot),
                "toward_target_share": float((np.abs(signed_diff(rot, args.rotate)) < 45).mean()),
                "label_az_error": float(err.mean()),
                "label_az_error_ci": bootstrap_ci(err),
                "pixel_change": float(change.mean()),
            }
        )
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "results.csv", index=False)
    (args.out / "results.json").write_text(json.dumps(rows, indent=1))
    print(df.round(2).to_string())


if __name__ == "__main__":
    main()
