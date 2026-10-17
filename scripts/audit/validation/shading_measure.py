"""Audit check: measure the light and dark sides of the object alone in the intervention outputs.

The article reads the output light with a pixel probe on the whole image. That
probe also sees the cast shadow and the floor. This script fits the same probe
recipe on a crop of the object (dd3d.cues.object_crop). In the crop the floor
is grey, so the cast shadow is not in the input. The script then reads the
outputs of the causal tests and of the patch test with both probes. Both
readings are rotations against the base output of the same image.

Each --set has the form name=outputs_folder=plan_file. A plan with target_ids
is a patch plan, and its outputs also have the condition "source".

Output: summary.csv, summary.md and per_image.csv in --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dd3d.cues import object_crop
from dd3d.data import pixel_features
from dd3d.metrics import bootstrap_ci, circular_abs_error, decode_azimuth
from dd3d.probe import ProbeConfig, evaluate, run


def signed_diff(a, b) -> np.ndarray:
    """Return a - b in degrees, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def crop_features(df: pd.DataFrame, images_dir: Path) -> np.ndarray:
    """Return the object crops (48 x 48 RGB, floor grey) in [0, 1] as one row for each image."""
    rows = []
    for image_id, shape in zip(df["image_id"], df["shape"], strict=True):
        rgb = np.asarray(Image.open(Path(images_dir) / shape / f"{image_id}.png").convert("RGB"))
        rows.append(object_crop(rgb).astype(np.float32).ravel() / 255.0)
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", action="append", required=True, help="name=outputs_folder=plan_file")
    parser.add_argument("--splits", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--images", type=Path, default=Path("data/v2/images"))
    parser.add_argument("--rotate", type=float, default=90.0, help="Target rotation of the conditions.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    splits = {n: pd.read_csv(args.splits / f"{n}.csv") for n in ("train", "val", "test", "test_cubes")}
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in splits.items()
    }
    readers = {"crop": lambda df, d: crop_features(df, d), "pixels": lambda df, d: pixel_features(df, d, 64)}
    probes, check = {}, []
    for name, read in readers.items():
        fit = {k: read(splits[k], args.images) for k in ("train", "val")}
        probes[name] = run(fit, {k: labels[k] for k in fit}, ProbeConfig()).model
        for split in ("test", "test_cubes"):
            res, *_ = evaluate(probes[name], read(splits[split], args.images), *labels[split])
            check.append({"probe": name, "split": split, "az_mae": res["az_mae"], "az_ci": res["az_mae_ci"]})
            print(f"probe {name} on the {split} renders: azimuth MAE {res['az_mae']:.2f}", flush=True)

    rows, per_image = [], []
    for item in args.set:
        set_name, outputs, plan_path = item.split("=")
        outputs = Path(outputs)
        plan = np.load(plan_path, allow_pickle=False)
        patch = "target_ids" in plan.files
        ids = plan["target_ids"] if patch else plan["image_ids"]
        frame = pd.DataFrame({"image_id": ids.astype(str), "shape": plan["shapes"].astype(str)})
        conditions = [str(c) for c in plan["conditions"]]
        if patch:
            conditions = ["base", "source", *[c for c in conditions if c != "base"]]
        target = args.rotate
        preds = {}
        for cond in conditions:
            folder = outputs / cond.replace(":", "_")
            for name, read in readers.items():
                p = probes[name].predict(read(frame, folder))
                preds[(cond, name)] = decode_azimuth(p[:, 0], p[:, 1])
        for cond in conditions:
            row = {"set": set_name, "condition": cond, "n": len(frame)}
            for name in readers:
                rot = signed_diff(preds[(cond, name)], preds[("base", name)])
                row[f"{name}_rotation_mean"] = float(rot.mean())
                row[f"{name}_rotation_ci"] = bootstrap_ci(rot)
                row[f"{name}_rotation_median"] = float(np.median(rot))
                row[f"{name}_toward_target_share"] = float((np.abs(signed_diff(rot, target)) < 45).mean())
                if not patch:
                    err = circular_abs_error(preds[(cond, name)], plan["azimuth"])
                    row[f"{name}_label_error"] = float(err.mean())
                for k, image_id in enumerate(frame["image_id"]):
                    item_row = {"set": set_name, "condition": cond, "probe": name, "image_id": image_id}
                    per_image.append({**item_row, "rotation": rot[k]})
            rows.append(row)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "summary.csv", index=False, float_format="%.3f")
    pd.DataFrame(per_image).to_csv(args.out / "per_image.csv", index=False, float_format="%.2f")
    pd.DataFrame(check).to_csv(args.out / "probe_check.csv", index=False, float_format="%.3f")
    cols = ["set", "condition", "n", "pixels_rotation_mean", "crop_rotation_mean", "crop_rotation_median"]
    cols += ["pixels_toward_target_share", "crop_toward_target_share"]
    md = df[cols].to_markdown(index=False, floatfmt=".2f")
    (args.out / "summary.md").write_text(pd.DataFrame(check).to_markdown(index=False) + "\n\n" + md + "\n")
    print(df[cols].round(2).to_string())


if __name__ == "__main__":
    main()
