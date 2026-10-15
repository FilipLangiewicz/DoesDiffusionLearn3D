"""Show where in the image a linear light probe reads the azimuth.

A linear probe on a grid of pooled features is a sum over the grid cells.
Thus the contribution of each cell to the prediction is exact
(dd3d.probe.spatial_contributions). For each test image the script takes the
absolute contribution of each cell to sin and cos of the azimuth. The scene
geometry then gives three regions of each image:

- object: the cells that show a part of the object;
- shadow: the other cells that show a part of the cast shadow;
- rest: all other cells (mostly floor).

The output gives the share of the contribution in each region, and the share of
the area of each region. A share above the area share means that the probe
reads this region more than the other cells. The figure shows the mean map
and, for sets with random positions, the mean map centred on the object.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from dd3d.data import pixel_features, pool_grid  # noqa: E402
from dd3d.geometry import IMAGE_SIZE, project  # noqa: E402
from dd3d.probe import ProbeConfig, run, spatial_contributions  # noqa: E402
from dd3d.select import feature_key  # noqa: E402
from dd3d.store import FeatureStore  # noqa: E402

TESTS = ("test", "test_cubes")


def regions(row: pd.Series, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean (grid, grid) masks of the object and of its cast shadow.

    A cylinder with the radius and the height of the object stands for the
    object. Its shadow comes from the sun direction of the label.
    """
    x, y, size = float(row.get("x", 0.0)), float(row.get("y", 0.0)), float(row["size"])
    cell = IMAGE_SIZE / grid
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([x + size * np.cos(ang), y + size * np.sin(ang)], axis=1)
    heights = np.linspace(0.0, 2.0 * size, 9)
    body = np.concatenate([np.c_[ring, np.full(len(ring), h)] for h in heights])
    az, el = np.radians(row["light_azimuth_deg"]), np.radians(row["light_elevation_deg"])
    u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    ground = body - u[None, :] * (body[:, 2:3] / u[2])

    def mask(points):
        px = project(points)[:, :2]
        px = px[(px >= 0).all(1) & (px < IMAGE_SIZE).all(1)]
        m = np.zeros((grid, grid), dtype=bool)
        idx = (px // cell).astype(int)
        m[idx[:, 1], idx[:, 0]] = True
        return m

    obj = mask(body)
    return obj, mask(ground) & ~obj


def centred(m: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """Place a (grid, grid) map on a (2 grid - 1) canvas with the object cell at the centre."""
    g = m.shape[0]
    rows, cols = np.nonzero(obj)
    r0, c0 = int(round(rows.mean())), int(round(cols.mean()))
    out = np.full((2 * g - 1, 2 * g - 1), np.nan)
    out[g - 1 - r0 : 2 * g - 1 - r0, g - 1 - c0 : 2 * g - 1 - c0] = m
    return out


def mean_image(df: pd.DataFrame, images: Path, size: int = 256) -> np.ndarray:
    acc = np.zeros((size, size))
    for image_id, shape in zip(df["image_id"], df["shape"], strict=True):
        img = Image.open(images / shape / f"{image_id}.png").convert("L").resize((size, size))
        acc += np.asarray(img, dtype=np.float64) / 255.0
    return acc / len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=None, help="Feature root (dd3d.store).")
    parser.add_argument("--keys", default="", help="Comma list of run/block_<bb>/t_<ttt> keys.")
    parser.add_argument("--scan", type=Path, default=None, help="Select the best fine-grid key of each run.")
    parser.add_argument("--runs", default="sana", help="Runs to select from --scan.")
    parser.add_argument("--grid", type=int, default=8, help="Grid of the maps.")
    parser.add_argument("--pixel-grid", type=int, default=8, help="Grid of the pixel probe (0: none).")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="labels.csv with x, y and size.")
    parser.add_argument("--title", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(args.labels).set_index("image_id")
    splits = {p.stem: pd.read_csv(p) for p in sorted(args.splits.glob("*.csv"))}
    splits = {k: v for k, v in splits.items() if k in ("train", "val", *TESTS)}
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in splits.items()
    }
    test = pd.concat([splits[k] for k in TESTS])
    geo = lab.loc[test["image_id"]].reset_index()
    masks = [regions(r, args.grid) for _, r in geo.iterrows()]

    keys = [k for k in args.keys.split(",") if k]
    if args.scan is not None:
        scan = pd.read_csv(args.scan)
        scan = scan[(scan["target"] == "global") & (scan["grid"] == args.grid) & (scan["split"] == "val")]
        for r in args.runs.split(","):
            g = scan[scan["run"] == r]
            if len(g):
                best = g.loc[(g["az_mae"] + g["el_mae"]).idxmin()]
                keys.append(feature_key(r, best["block"], best["t"]))
    store = FeatureStore(args.features) if args.features is not None else None

    panels = []
    if args.pixel_grid:
        feats = {k: pixel_features(v, args.images, args.pixel_grid) for k, v in splits.items()}
        n_comp = min(256, 3 * args.pixel_grid**2)
        panels.append((f"pixels {args.pixel_grid}x{args.pixel_grid}", feats, args.pixel_grid, n_comp))
    for key in keys:
        native = store.meta(key.split("/")[0])["grid_of_block"][str(int(key.split("/")[1].split("_")[1]))]
        feats = {
            k: pool_grid(store.load(key, v["image_id"].tolist()), native, args.grid)
            for k, v in splits.items()
        }
        panels.append((key, feats, args.grid, 256))

    background = mean_image(test.sample(min(300, len(test)), random_state=0), args.images)
    rows, figs = [], []
    for label, feats, grid, n_comp in panels:
        res = run(feats, labels, ProbeConfig(n_components=n_comp))
        x = np.concatenate([feats[k] for k in TESTS])
        contrib = spatial_contributions(res.model, x, grid)  # (n, 3, grid, grid)
        m = np.abs(contrib[:, :2]).sum(axis=1)
        m = m / m.sum(axis=(1, 2), keepdims=True)
        if grid != args.grid:
            m = pool_grid(m[..., None].reshape(len(m), -1), grid, args.grid).reshape(
                len(m), args.grid, args.grid
            )
            m = m / m.sum(axis=(1, 2), keepdims=True)
        share = {"object": [], "shadow": [], "rest": []}
        area = {"object": [], "shadow": [], "rest": []}
        cmaps = []
        for mi, (obj, sh) in zip(m, masks, strict=True):
            rest = ~(obj | sh)
            for name, mask in (("object", obj), ("shadow", sh), ("rest", rest)):
                share[name].append(mi[mask].sum())
                area[name].append(mask.mean())
            cmaps.append(centred(mi, obj))
        az_mae = {k: res.splits[k]["az_mae"] for k in TESTS}
        rows.append(
            {
                "source": label,
                **{f"{k}_az_mae": v for k, v in az_mae.items()},
                **{f"share_{k}": float(np.mean(v)) for k, v in share.items()},
                **{f"area_{k}": float(np.mean(v)) for k, v in area.items()},
            }
        )
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cmean = np.nanmean(np.stack(cmaps), axis=0)
        figs.append((label, m.mean(0), cmean, az_mae))
        print(rows[-1], flush=True)
        del feats

    fig, axes = plt.subplots(2, len(figs), figsize=(3.2 * len(figs), 6.4), squeeze=False)
    for j, (label, mean_map, cmap, az_mae) in enumerate(figs):
        ax = axes[0][j]
        ax.imshow(background, cmap="gray", vmin=0, vmax=1)
        ax.imshow(mean_map / mean_map.max(), cmap="inferno", alpha=0.5, extent=(0, 256, 256, 0))
        ax.set_title(f"{label}\nMAE {az_mae['test']:.1f} / {az_mae['test_cubes']:.1f} deg", fontsize=8)
        ax.axis("off")
        ax = axes[1][j]
        ax.imshow(cmap, cmap="inferno")
        c = cmap.shape[0] // 2
        ax.plot([c], [c], "c+", ms=10)
        ax.set_title("centred on the object", fontsize=8)
        ax.axis("off")
    fig.suptitle(args.title, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "maps.png", dpi=150)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "regions.csv", index=False, float_format="%.4f")
    (args.out / "regions.md").write_text(df.to_markdown(index=False, floatfmt=".3f") + "\n")
    np.savez(args.out / "maps.npz", **{f"mean_{i}": f[1] for i, f in enumerate(figs)})
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=1))


if __name__ == "__main__":
    main()
