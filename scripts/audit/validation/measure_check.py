"""Audit check: repeat the calibration numbers of the two shadow measures on the renders.

The article gives two calibration results without a result file:

- the dark-region measure (dd3d.cues.shadow_azimuth) has a median error of
  6.4 deg on the v2 renders and 10.3 deg on the cubes;
- the kept share (dd3d.cues.shadow_kept) is 1.00 for the same image and 0.00
  to 0.01 when the light turns by 90 deg (v2 and v2rot render pairs).

This script computes both numbers again. It also gives the share of images
without a measure (NaN), because the article reports medians.

Output: measures.csv and measures.md in --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dd3d.cues import shadow_azimuth, shadow_kept
from dd3d.metrics import circular_abs_error


def load(images: Path, shape: str, image_id: str) -> np.ndarray:
    return np.asarray(Image.open(images / shape / f"{image_id}.png").convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True, help="Split folder of v2.")
    parser.add_argument("--images", type=Path, required=True, help="Image folder of v2.")
    parser.add_argument("--rot-images", type=Path, required=True, help="Image folder of v2rot.")
    parser.add_argument("--pairs", type=Path, required=True, help="configs/kaggle/patch_pairs.txt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for split in ("test", "test_cubes"):
        df = pd.read_csv(args.splits / f"{split}.csv")
        items = zip(df["image_id"], df["shape"], strict=True)
        pred = np.array([shadow_azimuth(load(args.images, s, i)) for i, s in items])
        ok = ~np.isnan(pred)
        err = circular_abs_error(pred[ok], df["light_azimuth_deg"].to_numpy()[ok])
        rows.append(
            {
                "check": f"dark-region azimuth error, v2 {split}",
                "n": int(len(df)),
                "nan_share": float(1 - ok.mean()),
                "median": float(np.median(err)),
                "mean": float(err.mean()),
            }
        )
    same, turned = [], []
    for key in args.pairs.read_text().strip().split(","):
        shape = key.rsplit("_", 1)[0]
        base = load(args.images, shape, f"v2_{key}")
        rot = load(args.rot_images, shape, f"v2rot_{key}")
        same.append(shadow_kept(base, base))
        turned.append(shadow_kept(base, rot))
    for name, values in (("kept share, same render", same), ("kept share, v2 against v2rot", turned)):
        v = np.array(values, dtype=float)
        ok = ~np.isnan(v)
        rows.append(
            {
                "check": name,
                "n": int(len(v)),
                "nan_share": float(1 - ok.mean()),
                "median": float(np.median(v[ok])),
                "mean": float(v[ok].mean()),
            }
        )
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "measures.csv", index=False, float_format="%.4f")
    (args.out / "measures.md").write_text(df.to_markdown(index=False, floatfmt=".3f") + "\n")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
