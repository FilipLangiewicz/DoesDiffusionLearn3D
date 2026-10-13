"""Make the figures and the result tables of the article draft.

Inputs: the renders in data/audit_renders, the dataset images, the pixel cue
results and the shadow audit. Outputs go to docs/article/figures and
docs/article/tables.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from dd3d.cues import floor_only, highlight_position, load_rgb, object_crop, object_mask  # noqa: E402
from dd3d.geometry import azimuth_to_image_angle, project  # noqa: E402


def conventions(renders: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    center = project(np.zeros(3))[:2]
    for ax, az in zip(axes, (0, 90, 180, 270), strict=True):
        ax.imshow(Image.open(renders / f"conv_az{az:03d}_sun1.png").convert("RGB"))
        ang = np.radians(azimuth_to_image_angle(az))
        tip = center + 170 * np.array([np.cos(ang), -np.sin(ang)])
        ax.annotate(
            "",
            xy=tuple(tip),
            xytext=tuple(center),
            arrowprops={"arrowstyle": "->", "color": "#ffd600", "lw": 3},
        )
        ax.set_title(f"azimuth {az}°", fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "conventions.png", dpi=130)
    plt.close(fig)


def examples(src: Path, out: Path) -> None:
    ims = [Image.open(src / n).convert("RGB") for n in ("readme_good.png", "readme_bad.png")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, im in zip(axes, ims, strict=True):
        ax.imshow(im)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=130)
    plt.close(fig)


def cue_panel(images: Path, results_csv: Path, out: Path, tables: Path) -> None:
    rgb = load_rgb(images, "SmoothCube_v2", "SmoothCube_v2_new_000004")
    mask = object_mask(rgb)
    x, y = highlight_position(rgb, mask)
    df = pd.read_csv(results_csv)
    fig = plt.figure(figsize=(13, 3.6))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.1, 2.2])
    panels = [
        (rgb, "full image"),
        (floor_only(rgb, mask), "floor only"),
        (object_crop(rgb, mask, size=256), "object only"),
    ]
    for i, (img, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(img)
        if i == 0:
            ax.plot(x * rgb.shape[1], y * rgb.shape[0], "o", ms=9, mec="black", mfc="#ff1744")
            title += " (+ highlight position)"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(grid[0, 4])
    order = [
        "full image",
        "floor only (object removed)",
        "object only (crop)",
        "floor highlight position (2 numbers)",
    ]
    short = ["full image", "floor only", "object only", "highlight (2 numbers)"]
    width = 0.2
    for j, (target, split, color) in enumerate(
        [
            ("global", "test", "#1565c0"),
            ("global", "test_cubes", "#64b5f6"),
            ("local", "test", "#2e7d32"),
            ("local", "test_cubes", "#81c784"),
        ]
    ):
        sub = df[(df.target == target) & (df.split == split)].set_index("variant").loc[order]
        pos = np.arange(len(order)) + (j - 1.5) * width
        err = np.stack([sub.az_mae - sub.az_ci_low, sub.az_ci_high - sub.az_mae])
        ax.bar(pos, sub.az_mae, width, yerr=err, color=color, label=f"{target} label, {split}", capsize=2)
    ax.axhline(14.01, color="black", ls="--", lw=1)
    ax.text(3.45, 14.6, "SANA probe on cubes (original)", fontsize=7, ha="right")
    ax.set_xticks(np.arange(len(order)), short, fontsize=8)
    ax.set_ylabel("azimuth MAE (degrees)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "cues.png", dpi=130)
    plt.close(fig)

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Test (spheres, cylinders)} & \multicolumn{2}{c}{Cubes} \\",
        r"Variant & Label & Azimuth & Elevation & Azimuth & Elevation \\",
        r"\midrule",
    ]
    for variant, name in zip(order, short, strict=True):
        for target in ("global", "local"):
            r = {
                s: df[(df.variant == variant) & (df.target == target) & (df.split == s)].iloc[0]
                for s in ("test", "test_cubes")
            }
            cells = []
            for s in ("test", "test_cubes"):
                cells.append(f"{r[s].az_mae:.1f} [{r[s].az_ci_low:.1f}, {r[s].az_ci_high:.1f}]")
                cells.append(f"{r[s].el_mae:.1f}")
            lines.append(f"{name if target == 'global' else ''} & {target} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "cues.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, default=Path("data/audit_renders"))
    parser.add_argument("--images", type=Path, default=Path("data/final/images"))
    parser.add_argument("--cues", type=Path, default=Path("reports/experiments/pixel_cues/results.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/article"))
    args = parser.parse_args()
    figs, tables = args.out / "figures", args.out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    conventions(args.renders, figs)
    examples(Path("docs/figures"), figs)
    if args.cues.exists():
        cue_panel(args.images, args.cues, figs, tables)
    shutil.copy("reports/audit/shadow_audit.png", figs / "shadow_audit.png")
    print("figures ->", figs)


if __name__ == "__main__":
    main()
