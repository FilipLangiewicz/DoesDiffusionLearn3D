"""Measure the light direction in the outputs of the patch test.

A pixel probe, fitted on the v2 renders, predicts the azimuth of each output.
For each condition (block and region) the script reports the signed rotation
against the base output. It also gives this rotation as a fraction of the light
offset between source and target: 1 means that the output follows the source.
It also reports the rotation of the source output, as the upper bound. The
figure is a map of the mean fraction over the blocks and the regions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from dd3d.data import pixel_features  # noqa: E402
from dd3d.metrics import bootstrap_ci, decode_azimuth  # noqa: E402
from dd3d.probe import ProbeConfig, run  # noqa: E402


def signed_diff(a, b) -> np.ndarray:
    """Return a - b in degrees, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, nargs="+", required=True, help="Output folders of the job.")
    parser.add_argument("--plans", type=Path, nargs="+", required=True, help="The plan files, same order.")
    parser.add_argument("--splits", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--images", type=Path, default=Path("data/v2/images"))
    parser.add_argument("--offset", type=float, default=90.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    splits = {n: pd.read_csv(args.splits / f"{n}.csv") for n in ("train", "val")}
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in splits.items()
    }
    probe = run(
        {k: pixel_features(v, args.images, 64) for k, v in splits.items()}, labels, ProbeConfig()
    ).model

    def predict(folder: Path, frame: pd.DataFrame) -> np.ndarray:
        p = probe.predict(pixel_features(frame, folder, 64))
        return decode_azimuth(p[:, 0], p[:, 1])

    rows = []
    for outputs, plan_path in zip(args.outputs, args.plans, strict=True):
        plan = np.load(plan_path, allow_pickle=False)
        frame = pd.DataFrame(
            {"image_id": plan["target_ids"].astype(str), "shape": plan["shapes"].astype(str)}
        )
        base = predict(outputs / "base", frame)
        src = predict(outputs / "source", frame)
        imgs = {}
        for cond in ["base", "source", *[str(c) for c in plan["conditions"] if str(c) != "base"]]:
            folder = outputs / cond.replace(":", "_")
            pred = base if cond == "base" else (src if cond == "source" else predict(folder, frame))
            pix = np.stack(
                [
                    np.asarray(Image.open(folder / s / f"{i}.png").convert("RGB"), dtype=np.float32)
                    for i, s in zip(frame["image_id"], frame["shape"], strict=True)
                ]
            )
            if cond == "base":
                imgs["base"] = pix
            rot = signed_diff(pred, base)
            for k, image_id in enumerate(frame["image_id"]):
                rows.append(
                    {
                        "image_id": image_id,
                        "condition": cond,
                        "label_az": float(plan["azimuth"][k]),
                        "pred_az": float(pred[k]),
                        "rotation": float(rot[k]),
                        "pixel_change": float(np.abs(pix[k] - imgs["base"][k]).mean())
                        if "base" in imgs
                        else 0.0,
                    }
                )
    per_image = pd.DataFrame(rows)
    summary = []
    for cond, g in per_image.groupby("condition", sort=False):
        frac = g["rotation"].to_numpy() / args.offset
        summary.append(
            {
                "condition": cond,
                "rotation_mean": float(g["rotation"].mean()),
                "fraction_mean": float(frac.mean()),
                "fraction_ci": bootstrap_ci(frac),
                "toward_source_share": float((np.abs(signed_diff(g["rotation"], args.offset)) < 45).mean()),
                "pixel_change": float(g["pixel_change"].mean()),
                "n": len(g),
            }
        )
    df = pd.DataFrame(summary)
    args.out.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(args.out / "per_image.csv", index=False, float_format="%.3f")
    df.to_csv(args.out / "summary.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(df.round(3).to_string())

    patch = df[df["condition"].str.startswith("patch:")].copy()
    if len(patch):
        patch["block"] = patch["condition"].str.extract(r"b(\d+)_")[0].astype(int)
        patch["region"] = patch["condition"].str.extract(r"b\d+_(\w+)$")[0]
        table = patch.pivot(index="block", columns="region", values="fraction_mean")
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        im = ax.imshow(table.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(table.index)), [f"block {b}" for b in table.index])
        for (r, c), v in np.ndenumerate(table.to_numpy()):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7)
        src_frac = df.loc[df["condition"] == "source", "fraction_mean"]
        ax.set_title(f"rotation / {args.offset:g} deg (source output: {src_frac.iloc[0]:.2f})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(args.out / "patch_map.png", dpi=150)
        table.to_csv(args.out / "patch_map.csv")


if __name__ == "__main__":
    main()
