"""Show where in the image a linear light probe reads the light direction.

A linear probe on a grid of pooled features is a sum over the grid cells. Thus
the contribution of each cell to the prediction is exact (dd3d.probe.
spatial_contributions). The script fits the probe on the train split. On the
test images, it maps the mean absolute contribution of each cell to the sine
and the cosine of the azimuth. The map is drawn over the mean test image.

The same map for the pixel probe uses the cells of the pixel grid.
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

from dd3d.data import load_feature_dir, pixel_features  # noqa: E402
from dd3d.probe import ProbeConfig, run, spatial_contributions  # noqa: E402


def mean_image(df: pd.DataFrame, images_dir: Path, size: int = 256) -> np.ndarray:
    acc = np.zeros((size, size, 3))
    for image_id, shape in zip(df["image_id"], df["shape"], strict=True):
        img = Image.open(images_dir / shape / f"{image_id}.png").convert("RGB").resize((size, size))
        acc += np.asarray(img, dtype=np.float64) / 255.0
    return acc / len(df)


def cell_map(model, x: np.ndarray, grid: int) -> np.ndarray:
    """Return the mean absolute contribution to sin and cos, normalized to a sum of 1."""
    contrib = spatial_contributions(model, x, grid)  # (n, 3, grid, grid)
    m = np.abs(contrib[:, :2]).mean(axis=(0, 1))
    return m / m.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, nargs="*", default=[], help="Feature folders (block/t).")
    parser.add_argument("--labels", nargs="*", default=[], help="One label for each feature folder.")
    parser.add_argument("--pixel-grid", type=int, default=16, help="Grid of the pixel probe (0: none).")
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--images", type=Path, default=Path("data/final/images"))
    parser.add_argument("--title", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    splits = {n: pd.read_csv(args.splits / f"{n}.csv") for n in ("train", "val", "test", "test_cubes")}
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in splits.items()
    }
    test = pd.concat([splits["test"], splits["test_cubes"]])
    background = mean_image(test.sample(min(300, len(test)), random_state=0), args.images)

    panels = []
    if args.pixel_grid:
        feats = {k: pixel_features(v, args.images, args.pixel_grid) for k, v in splits.items()}
        res = run(feats, labels, ProbeConfig(n_components=min(256, 3 * args.pixel_grid**2)))
        x = np.concatenate([feats["test"], feats["test_cubes"]])
        panels.append(
            (f"pixels {args.pixel_grid}x{args.pixel_grid}", cell_map(res.model, x, args.pixel_grid), res)
        )
    for folder, label in zip(args.features, args.labels, strict=True):
        meta = json.loads((folder.parents[1] / "meta.json").read_text())
        grid = meta["grid_of_block"][str(int(folder.parent.name.split("_")[1]))]
        feats = {k: load_feature_dir(folder, v["image_id"].tolist()) for k, v in splits.items()}
        res = run(feats, labels, ProbeConfig())
        x = np.concatenate([feats["test"], feats["test_cubes"]])
        panels.append((label, cell_map(res.model, x, grid), res))
        del feats

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    axes = np.atleast_1d(axes)
    rows = []
    for ax, (label, m, res) in zip(axes, panels, strict=True):
        # A gray mean image under a half-transparent map keeps the scene visible.
        ax.imshow(background.mean(axis=2), cmap="gray", vmin=0, vmax=1)
        ax.imshow(m / m.max(), cmap="inferno", alpha=0.45, extent=(0, 256, 256, 0), interpolation="nearest")
        cubes = res.splits["test_cubes"]
        ax.set_title(f"{label}\ncubes az MAE {cubes['az_mae']:.1f} deg", fontsize=9)
        ax.axis("off")
        rows.append({"panel": label, "cubes_az_mae": cubes["az_mae"], "map": m.round(5).tolist()})
    fig.suptitle(args.title, fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    args.out.with_suffix(".json").write_text(json.dumps(rows))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
