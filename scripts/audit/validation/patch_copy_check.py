"""Audit check: does a patch of the object tokens move the cast shadow, or copy it?

The object region of the patch test is a set of token cells. These cells also
show the floor next to the object, where the cast shadow starts. Thus a patch
of these cells can copy a part of the source shadow directly. This script
splits the crisp shadow of the source output into the part inside the object
cells and the part outside them. For each condition it gives the share of each
part that appears in the output. It also measures the direction of the dark
region after it removes the object cells from the image.

Output: per_pair.csv, summary.csv, summary.md and examples.png in --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from dd3d.cues import crisp_shadow_mask, dilate, shadow_azimuth  # noqa: E402

SIDE = 32  # token grid of the transformer
MIN_PX = 30  # the same minimum as dd3d.cues.shadow_kept


def signed_diff(a, b) -> np.ndarray:
    """Return a - b in degrees, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def token_mask_to_pixels(mask: np.ndarray, size: int) -> np.ndarray:
    """Spread a (SIDE * SIDE,) token mask to a (size, size) pixel mask (row-major tokens)."""
    k = size // SIDE
    return np.kron(mask.reshape(SIDE, SIDE), np.ones((k, k), dtype=bool)).astype(bool)


def share(part: np.ndarray, whole: np.ndarray) -> float:
    """Return |part| / |whole|, or NaN if the whole has fewer than MIN_PX pixels."""
    n = int(whole.sum())
    return float(part.sum() / n) if n >= MIN_PX else float("nan")


def without_cells(rgb: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return the image with the given pixels set to the floor median, so that they are never dark."""
    out = rgb.copy()
    out[cells] = np.median(rgb[~cells], axis=0).astype(rgb.dtype)
    return out


def load(folder: Path, shape: str, image_id: str) -> np.ndarray:
    return np.asarray(Image.open(folder / shape / f"{image_id}.png").convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, nargs="+", required=True, help="Output folders of the job.")
    parser.add_argument("--plans", type=Path, nargs="+", required=True, help="The plan files, same order.")
    parser.add_argument("--margins", default="0,16", help="Growth of the object cells in pixels.")
    parser.add_argument("--offset", type=float, default=90.0)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    margins = [int(m) for m in args.margins.split(",")]

    rows, examples = [], []
    for outputs, plan_path in zip(args.outputs, args.plans, strict=True):
        plan = np.load(plan_path, allow_pickle=False)
        conditions = [str(c) for c in plan["conditions"]]
        # All object conditions use the same cells; take the first full-step one.
        obj_c = next(c for c, name in enumerate(conditions) if name.endswith("_object"))
        names = ["base", "source", *[c for c in conditions if c != "base"]]
        pairs = zip(plan["target_ids"].astype(str), plan["shapes"].astype(str), strict=True)
        for i, (tid, shape) in enumerate(pairs):
            imgs = {n: load(outputs / n.replace(":", "_"), shape, tid) for n in names}
            size = imgs["base"].shape[0]
            cells0 = token_mask_to_pixels(plan["masks"][i, obj_c], size)
            src_shadow = crisp_shadow_mask(imgs["source"])
            tgt_shadow = crisp_shadow_mask(imgs["base"])
            crisp = {n: crisp_shadow_mask(img) for n, img in imgs.items()}
            for m in margins:
                cells = dilate(cells0, m)
                s_in, s_out = src_shadow & cells, src_shadow & ~cells
                t_out = tgt_shadow & ~cells
                base_dir = shadow_azimuth(without_cells(imgs["base"], cells))
                for n in names:
                    p = crisp[n]
                    rows.append(
                        {
                            "pair": tid,
                            "condition": n,
                            "margin_px": m,
                            "source_shadow_px": int(src_shadow.sum()),
                            "source_shadow_in_cells": share(s_in, src_shadow),
                            "shown_all": share(src_shadow & p, src_shadow),
                            "shown_in_cells": share(s_in & p, s_in),
                            "shown_outside_cells": share(s_out & p, s_out),
                            "target_kept_outside_cells": share(t_out & p, t_out),
                            "dark_rotation_outside_cells": float(
                                signed_diff(shadow_azimuth(without_cells(imgs[n], cells)), base_dir)
                            ),
                        }
                    )
            if len(examples) < args.examples:
                pname = "patch:b10_object" if "patch:b10_object" in names else names[2]
                examples.append((tid, imgs["base"], imgs["source"], imgs[pname], cells0, src_shadow, pname))

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "per_pair.csv", index=False, float_format="%.4f")
    agg = (
        df.groupby(["margin_px", "condition"], sort=False)
        .agg(
            pairs=("pair", "nunique"),
            source_shadow_in_cells=("source_shadow_in_cells", "mean"),
            shown_all=("shown_all", "mean"),
            shown_in_cells=("shown_in_cells", "mean"),
            shown_outside_cells=("shown_outside_cells", "mean"),
            shown_outside_n=("shown_outside_cells", "count"),
            target_kept_outside_cells=("target_kept_outside_cells", "mean"),
            dark_rotation_outside_median=("dark_rotation_outside_cells", "median"),
        )
        .reset_index()
    )
    agg["dark_fraction_outside_median"] = agg["dark_rotation_outside_median"] / args.offset
    agg.to_csv(args.out / "summary.csv", index=False, float_format="%.4f")
    (args.out / "summary.md").write_text(agg.to_markdown(index=False, floatfmt=".3f") + "\n")
    print(agg.round(3).to_string())

    if examples:
        fig, axes = plt.subplots(len(examples), 3, figsize=(7.5, 2.6 * len(examples)))
        axes = np.atleast_2d(axes)
        for r, (tid, base, src, patched, cells, sshadow, pname) in enumerate(examples):
            for c, (img, title) in enumerate(((base, "base (target)"), (src, "source"), (patched, pname))):
                ax = axes[r][c]
                ax.imshow(img)
                ax.contour(cells, levels=[0.5], colors="red", linewidths=0.8)
                ax.contour(sshadow, levels=[0.5], colors="lime", linewidths=0.8)
                ax.set_title(f"{tid}\n{title}" if c == 0 else title, fontsize=7)
                ax.axis("off")
        fig.suptitle("red: object cells of the patch; green: crisp shadow of the source output", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / "examples.png", dpi=100)


if __name__ == "__main__":
    main()
