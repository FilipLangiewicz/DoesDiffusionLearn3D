"""Calibrate the measures of the output light on renders with a known light.

The causal test and the patch test read the output light with four measures.
This script gives the scale of each measure on renders, where the true light
is known:

- dark region (dd3d.cues.shadow_azimuth): the azimuth error on the v2 renders
  (test split and cubes), and the share of images without a measure.
- kept share (dd3d.cues.shadow_kept): the value for the same render. Also the
  value for a render of the same scene with the light turned by 90 deg (v2rot).
- pixel probe (64 x 64 pixels of the whole image): the azimuth error on the v2
  renders. Also the rotation that it reads between a v2 render and its v2rot render.
- crop probe (a crop of the object, dd3d.cues.crop_features): the same two values.

The v2rot renders have the light azimuth of v2 + 90 deg, so a correct measure
reads +90 deg for these pairs.

Output: measures.csv and measures.md in --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dd3d.cues import crop_features, shadow_azimuth, shadow_kept
from dd3d.data import pixel_features
from dd3d.metrics import circular_abs_error, decode_azimuth
from dd3d.probe import ProbeConfig, run


def signed_diff(a, b) -> np.ndarray:
    """Return a - b in degrees, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def load(images: Path, shape: str, image_id: str) -> np.ndarray:
    return np.asarray(Image.open(images / shape / f"{image_id}.png").convert("RGB"))


def summary(name: str, values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    return {
        "measure": name,
        "n": int(len(v)),
        "nan_share": float(1 - ok.mean()),
        "median": float(np.median(v[ok])),
        "mean": float(v[ok].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True, help="Split folder of v2.")
    parser.add_argument("--images", type=Path, required=True, help="Image folder of v2.")
    parser.add_argument("--rot-images", type=Path, required=True, help="Image folder of v2rot.")
    parser.add_argument("--pairs", type=Path, required=True, help="configs/kaggle/patch_pairs.txt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    splits = {n: pd.read_csv(args.splits / f"{n}.csv") for n in ("train", "val", "test", "test_cubes")}
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in splits.items()
    }
    readers = {
        "pixel probe": lambda df, d: pixel_features(df, d, 64),
        "crop probe": lambda df, d: crop_features(df, d),
    }
    probes = {
        name: run({k: read(splits[k], args.images) for k in ("train", "val")}, labels, ProbeConfig()).model
        for name, read in readers.items()
    }

    def predict(name: str, df: pd.DataFrame, images: Path) -> np.ndarray:
        p = probes[name].predict(readers[name](df, images))
        return decode_azimuth(p[:, 0], p[:, 1])

    rows = []
    for split in ("test", "test_cubes"):
        df = splits[split]
        az = df["light_azimuth_deg"].to_numpy()
        items = zip(df["image_id"], df["shape"], strict=True)
        dark = np.array([shadow_azimuth(load(args.images, s, i)) for i, s in items])
        ok = ~np.isnan(dark)
        err = np.full(len(dark), np.nan)
        err[ok] = circular_abs_error(dark[ok], az[ok])
        rows.append(summary(f"dark region: azimuth error, v2 {split}", err))
        for name in readers:
            rows.append(
                summary(
                    f"{name}: azimuth error, v2 {split}",
                    circular_abs_error(predict(name, df, args.images), az),
                )
            )

    keys = args.pairs.read_text().strip().split(",")
    shapes = [k.rsplit("_", 1)[0] for k in keys]
    base = pd.DataFrame({"image_id": [f"v2_{k}" for k in keys], "shape": shapes})
    turned = pd.DataFrame({"image_id": [f"v2rot_{k}" for k in keys], "shape": shapes})
    same, moved, dark_rot = [], [], []
    for b, t, s in zip(base["image_id"], turned["image_id"], shapes, strict=True):
        img_b, img_t = load(args.images, s, b), load(args.rot_images, s, t)
        same.append(shadow_kept(img_b, img_b))
        moved.append(shadow_kept(img_b, img_t))
        dark_rot.append(float(signed_diff(shadow_azimuth(img_t), shadow_azimuth(img_b))))
    rows.append(summary("kept share: same render", np.array(same)))
    rows.append(summary("kept share: render with the light turned by 90 deg", np.array(moved)))
    rows.append(summary("dark region: rotation for a light turned by 90 deg", np.array(dark_rot)))
    for name in readers:
        rot = signed_diff(predict(name, turned, args.rot_images), predict(name, base, args.images))
        rows.append(summary(f"{name}: rotation for a light turned by 90 deg", rot))

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "measures.csv", index=False, float_format="%.4f")
    (args.out / "measures.md").write_text(df.to_markdown(index=False, floatfmt=".3f") + "\n")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
