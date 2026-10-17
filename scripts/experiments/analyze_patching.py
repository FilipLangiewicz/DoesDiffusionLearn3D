"""Measure the light direction in the outputs of the patch test.

A pixel probe, fitted on the v2 renders, predicts the azimuth of each output.
For each condition (block and region) the script reports the signed rotation
against the base output. It also gives this rotation as a fraction of the light
offset between source and target: 1 means that the output follows the source.
It also reports the rotation of the source output, as the upper bound. A second
measure reads the direction of the dark shadow region (dd3d.cues.shadow_azimuth).
It uses no fitted model and does not look at the object, so a patch of the object
tokens cannot move it by a copy of the object shading. Two more columns give the
share of the crisp target shadow that stays (shadow_kept) and the share of the
crisp source shadow that appears (source_shadow_shown). The
figure is a map of the mean fraction over the blocks and the regions.

The patched tokens copy the source image at their own place. The object cells
also show the floor next to the object, where the cast shadow starts, so the
two shadow shares above include a copy of pixels. The columns *_outside
measure only outside the patched cells, grown by one token (16 px), because
the decoder spreads the change of a token to its neighbours. Only these
columns show a shadow that the model draws at another place. The column
source_shadow_outside_base gives the same share for the base output.

The column crop_rotation is the rotation that a probe on a crop of the object
reads (dd3d.cues.crop_features), so it shows the object alone.
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

from dd3d.cues import crisp_shadow_mask, crop_features, dilate, shadow_azimuth, shadow_kept  # noqa: E402
from dd3d.data import pixel_features  # noqa: E402
from dd3d.metrics import bootstrap_ci, decode_azimuth  # noqa: E402
from dd3d.probe import ProbeConfig, run  # noqa: E402

SIDE = 32  # token grid of the transformer
MARGIN_PX = 16  # one token of the 512 px output
MIN_PX = 30  # the same minimum as dd3d.cues.shadow_kept


def signed_diff(a, b) -> np.ndarray:
    """Return a - b in degrees, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def region_pixels(token_mask: np.ndarray, size: int, margin: int) -> np.ndarray:
    """Spread a (SIDE * SIDE,) token mask to a (size, size) pixel mask and grow it by margin pixels."""
    k = size // SIDE
    cells = np.kron(token_mask.reshape(SIDE, SIDE), np.ones((k, k), dtype=bool)).astype(bool)
    return dilate(cells, margin)


def share(part: np.ndarray, whole: np.ndarray) -> float:
    """Return |part| / |whole|, or NaN if the whole has fewer than MIN_PX pixels."""
    n = int(whole.sum())
    return float(part.sum() / n) if n >= MIN_PX else float("nan")


def without(rgb: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Return the image with the region set to the floor median, so that it is never dark."""
    if not region.any():
        return rgb
    out = rgb.copy()
    out[region] = np.median(rgb[~region], axis=0).astype(rgb.dtype)
    return out


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

    crop_probe = run(
        {k: crop_features(v, args.images) for k, v in splits.items()}, labels, ProbeConfig()
    ).model

    def predict(folder: Path, frame: pd.DataFrame) -> np.ndarray:
        p = probe.predict(pixel_features(frame, folder, 64))
        return decode_azimuth(p[:, 0], p[:, 1])

    def predict_crop(folder: Path, frame: pd.DataFrame) -> np.ndarray:
        p = crop_probe.predict(crop_features(frame, folder))
        return decode_azimuth(p[:, 0], p[:, 1])

    rows = []
    for outputs, plan_path in zip(args.outputs, args.plans, strict=True):
        plan = np.load(plan_path, allow_pickle=False)
        frame = pd.DataFrame(
            {"image_id": plan["target_ids"].astype(str), "shape": plan["shapes"].astype(str)}
        )
        base = predict(outputs / "base", frame)
        src = predict(outputs / "source", frame)
        crop_base = predict_crop(outputs / "base", frame)
        index = {str(c): i for i, c in enumerate(plan["conditions"])}
        imgs, crisp = {}, {}
        for cond in ["base", "source", *[str(c) for c in plan["conditions"] if str(c) != "base"]]:
            folder = outputs / cond.replace(":", "_")
            pred = base if cond == "base" else (src if cond == "source" else predict(folder, frame))
            crop_rot = signed_diff(predict_crop(folder, frame), crop_base)
            pix = np.stack(
                [
                    np.asarray(Image.open(folder / s / f"{i}.png").convert("RGB"), dtype=np.float32)
                    for i, s in zip(frame["image_id"], frame["shape"], strict=True)
                ]
            )
            # The second measure: the direction of the dark shadow region (no fitted model).
            shadow = np.array([shadow_azimuth(img.astype(np.uint8)) for img in pix])
            if cond == "base":
                imgs["base"] = pix
                base_shadow = shadow
            if cond in ("base", "source"):
                imgs[cond] = pix
                crisp[cond] = [crisp_shadow_mask(img.astype(np.uint8)) for img in pix]
            rot = signed_diff(pred, base)
            srot = signed_diff(shadow, base_shadow)
            for k, image_id in enumerate(frame["image_id"]):
                img = pix[k].astype(np.uint8)
                outside = {}
                if "source" in crisp:
                    mask = plan["masks"][k, index[cond]] if cond in index else np.zeros(SIDE * SIDE, bool)
                    far = ~region_pixels(mask, img.shape[0], MARGIN_PX)
                    src_far, tgt_far = crisp["source"][k] & far, crisp["base"][k] & far
                    here = crisp_shadow_mask(img)
                    base_img = imgs["base"][k].astype(np.uint8)
                    turn = signed_diff(
                        shadow_azimuth(without(img, ~far)), shadow_azimuth(without(base_img, ~far))
                    )
                    outside = {
                        "source_shadow_outside": share(src_far & here, src_far),
                        "source_shadow_outside_base": share(src_far & crisp["base"][k], src_far),
                        "target_kept_outside": share(tgt_far & here, tgt_far),
                        "shadow_rotation_outside": float(turn),
                    }
                rows.append(
                    {
                        "image_id": image_id,
                        "condition": cond,
                        "label_az": float(plan["azimuth"][k]),
                        "pred_az": float(pred[k]),
                        "rotation": float(rot[k]),
                        "shadow_rotation": float(srot[k]),
                        "shadow_kept": shadow_kept(imgs["base"][k].astype(np.uint8), pix[k].astype(np.uint8)),
                        "source_shadow_shown": shadow_kept(
                            imgs["source"][k].astype(np.uint8), pix[k].astype(np.uint8)
                        )
                        if "source" in imgs
                        else float("nan"),
                        "pixel_change": float(np.abs(pix[k] - imgs["base"][k]).mean())
                        if "base" in imgs
                        else 0.0,
                        "crop_rotation": float(crop_rot[k]),
                        **outside,
                    }
                )
    per_image = pd.DataFrame(rows)
    summary = []
    for cond, g in per_image.groupby("condition", sort=False):
        frac = g["rotation"].to_numpy() / args.offset
        sfrac = g["shadow_rotation"].dropna().to_numpy() / args.offset
        summary.append(
            {
                "condition": cond,
                "rotation_mean": float(g["rotation"].mean()),
                "fraction_mean": float(frac.mean()),
                "fraction_ci": bootstrap_ci(frac),
                "toward_source_share": float((np.abs(signed_diff(g["rotation"], args.offset)) < 45).mean()),
                "pixel_change": float(g["pixel_change"].mean()),
                "shadow_fraction_median": float(np.median(sfrac)) if len(sfrac) else float("nan"),
                "shadow_fraction_mean": float(sfrac.mean()) if len(sfrac) else float("nan"),
                "shadow_fraction_ci": bootstrap_ci(sfrac) if len(sfrac) else (float("nan"), float("nan")),
                "shadow_n": int(len(sfrac)),
                "shadow_kept": float(g["shadow_kept"].mean()),
                "source_shadow_shown": float(g["source_shadow_shown"].mean()),
                "crop_fraction_mean": float(g["crop_rotation"].mean() / args.offset),
                "crop_fraction_median": float(g["crop_rotation"].median() / args.offset),
                "source_shadow_outside": float(g["source_shadow_outside"].mean()),
                "source_shadow_outside_base": float(g["source_shadow_outside_base"].mean()),
                "target_kept_outside": float(g["target_kept_outside"].mean()),
                "shadow_fraction_outside_median": float(g["shadow_rotation_outside"].median() / args.offset),
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
        patch.pivot(index="block", columns="region", values="shadow_fraction_median").to_csv(
            args.out / "patch_map_shadow.csv"
        )


if __name__ == "__main__":
    main()
