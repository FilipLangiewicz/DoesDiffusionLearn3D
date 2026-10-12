"""Measure the real cast shadows in Blender renders and check the arrow geometry.

Input: the renders of make_render_specs.py (shadows.json) and the dataset images.
For each scene the script makes two shadow maps:

- key shadow: render without the object minus render with the object, Lamp_Key only;
- full shadow: render without the object minus the dataset image (all lamps).

The difference of the two maps is the shadow of the constant sun lamp (plus some
ambient occlusion). The script measures the direction from the object base to
the centroid of each map. It compares this direction with the exact arrow and
with the legacy arrow, and writes a CSV.
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

from dd3d.cues import dilate, object_mask  # noqa: E402
from dd3d.geometry import (  # noqa: E402
    BASE_CAMERA,
    SUN_TRAVEL_DIRECTION,
    angle_diff_deg,
    fit_camera_jitter,
    image_angle_deg,
    lamp_position,
    legacy_arrow_angle_deg,
    project,
    shadow_segment_px,
)

LUM = np.array([0.299, 0.587, 0.114])


def lum(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) @ LUM


def centroid_angle(diff: np.ndarray, origin_px: np.ndarray, thr: float) -> tuple[float, float]:
    """Return the image angle from the origin to the weighted centroid of ``diff > thr``."""
    w = np.where(diff > thr, diff, 0.0)
    if w.sum() == 0:
        return float("nan"), 0.0
    ys, xs = np.mgrid[0 : diff.shape[0], 0 : diff.shape[1]]
    c = np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])
    return float(image_angle_deg(origin_px, c)), float(w.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, default=Path("data/audit_renders"))
    parser.add_argument("--images", type=Path, default=Path("data/final/images"))
    parser.add_argument("--scenes", type=Path, default=Path("data/final/scenes"))
    parser.add_argument("--out", type=Path, default=Path("reports/audit"))
    parser.add_argument("--thr", type=float, default=4.0, help="Luminance difference (0-255) for shadow pixels.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, panels = [], []
    for bg_key_path in sorted(args.renders.glob("*_bg_key.png")):
        image_id = bg_key_path.name.removesuffix("_bg_key.png")
        shape = image_id.rsplit("_new_", 1)[0]
        scene = json.loads((args.scenes / shape / f"{image_id}.json").read_text())
        obj = scene["objects"][0]
        p = np.array(obj["3d_coords"])
        jitter, _ = fit_camera_jitter(p, obj["pixel_coords"])
        cam = BASE_CAMERA + jitter
        lamp = lamp_position(scene["light_azimuth_deg"], scene["light_elevation_deg"])

        img_path = args.images / shape / f"{image_id}.png"
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        mask = dilate(object_mask(rgb), 3)
        key_diff = lum(bg_key_path) - lum(args.renders / f"{image_id}_obj_key.png")
        full_diff = lum(args.renders / f"{image_id}_bg_full.png") - lum(img_path)
        key_diff[mask] = 0.0
        full_diff[mask] = 0.0
        sun_diff = np.clip(full_diff - key_diff, 0.0, None)

        start, end = shadow_segment_px(p, lamp, cam)
        exact = float(image_angle_deg(start, end))
        legacy = float(legacy_arrow_angle_deg(p, lamp))
        base = p * np.array([1.0, 1.0, 0.0])
        ground = SUN_TRAVEL_DIRECTION * np.array([1.0, 1.0, 0.0])
        sun_exact = float(image_angle_deg(project(base, cam)[:2], project(base + ground, cam)[:2]))
        key_meas, key_mass = centroid_angle(key_diff, start, args.thr)
        sun_meas, sun_mass = centroid_angle(sun_diff, start, args.thr)
        floor_full = lum(args.renders / f"{image_id}_bg_full.png")
        floor_key = lum(bg_key_path)
        rows.append(
            {
                "image_id": image_id,
                "az": scene["light_azimuth_deg"],
                "el": scene["light_elevation_deg"],
                "key_shadow_measured": key_meas,
                "key_shadow_exact": exact,
                "key_shadow_legacy": legacy,
                "err_exact": float(angle_diff_deg(key_meas, exact)),
                "err_legacy": float(angle_diff_deg(key_meas, legacy)),
                "sun_shadow_measured": sun_meas,
                "sun_shadow_exact": sun_exact,
                "err_sun": float(angle_diff_deg(sun_meas, sun_exact)),
                "sun_to_key_shadow_mass": sun_mass / max(key_mass, 1e-9),
                "floor_light_from_key_lamp": float(floor_key.mean() / floor_full.mean()),
            }
        )
        panels.append((image_id, rgb, key_diff, sun_diff, start, end, legacy, sun_exact, cam, base, ground))

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "shadow_audit.csv", index=False, float_format="%.2f")
    print(df[["image_id", "err_exact", "err_legacy", "err_sun", "sun_to_key_shadow_mass"]].to_string())
    print(
        f"mean error vs measured key shadow: exact {df.err_exact.mean():.1f} deg, legacy {df.err_legacy.mean():.1f}"
    )
    print(f"key lamp share of the mean floor brightness: {df.floor_light_from_key_lamp.mean():.2f}")

    n = min(4, len(panels))
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.5 * n))
    for row_axes, (image_id, rgb, key_diff, sun_diff, start, end, legacy, _sun_exact, cam, base, ground) in zip(
        np.atleast_2d(axes), panels[:n], strict=False
    ):
        ax0, ax1, ax2 = row_axes
        ax0.imshow(rgb)
        length = np.linalg.norm(end - start)
        leg = start + length * np.array([np.cos(np.radians(legacy)), -np.sin(np.radians(legacy))])
        ax0.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#00c853", "lw": 2.5})
        ax0.annotate("", xy=leg, xytext=start, arrowprops={"arrowstyle": "->", "color": "#ffab00", "lw": 2})
        sun_tip = project(base + 2.0 * ground / np.linalg.norm(ground), cam)[:2]
        ax0.annotate("", xy=sun_tip, xytext=start, arrowprops={"arrowstyle": "->", "color": "#616161", "lw": 2})
        ax0.set_title(f"{image_id}\ngreen: exact key shadow, orange: old arrow, grey: sun", fontsize=7)
        ax1.imshow(key_diff, cmap="magma", vmin=0, vmax=40)
        ax1.set_title("shadow of the key lamp (render difference)", fontsize=7)
        ax2.imshow(sun_diff, cmap="magma", vmin=0, vmax=40)
        ax2.set_title("extra shadow from the constant sun lamp", fontsize=7)
        for ax in row_axes:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out / "shadow_audit.png", dpi=110)
    print(f"figure -> {args.out / 'shadow_audit.png'}")


if __name__ == "__main__":
    main()
