"""Test the light options for dataset v2 with a few controlled renders.

The object stands at the scene centre, so the label is the light direction at
the object for every variant. The variants are:

- dataset: AREA key lamp, sun lamp Area on, world light 0.2 (the current data).
- no_sun: the same, but the sun lamp Area is off.
- sun_key: a SUN key lamp (direction only), sun lamp Area off, world light 0.2.
- sun_key_dark: the same as sun_key, with the world light 0.05.

Each scene also has a render without the object, so the script can measure the
cast shadow and the floor highlight. Use "spec" to write the Blender jobs and
"analyze" to measure the renders.
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
from dd3d.geometry import angle_diff_deg, image_angle_deg, lamp_position, project, shadow_segment_px  # noqa: E402

VARIANTS = {
    "dataset": {"key_type": "AREA", "sun_on": True, "world_strength": 0.2},
    "no_sun": {"key_type": "AREA", "sun_on": False, "world_strength": 0.2},
    "sun_key": {"key_type": "SUN", "sun_on": False, "world_strength": 0.2, "flat_ground": True},
    "sun_key_dark": {"key_type": "SUN", "sun_on": False, "world_strength": 0.05, "flat_ground": True},
}
ANGLES = [(30, 35), (150, 35), (270, 35), (30, 65), (150, 65), (270, 65)]
LUM = np.array([0.299, 0.587, 0.114])


def spec(out_dir: Path, key_strength: float) -> list[dict]:
    jobs = []
    for name, opts in VARIANTS.items():
        for az, el in ANGLES:
            for with_obj in (True, False):
                jobs.append(
                    {
                        "out": str(out_dir / f"{name}_az{az:03d}_el{el}_{'obj' if with_obj else 'bg'}.png"),
                        "shape": "Sphere" if with_obj else None,
                        "size": 1.0,
                        "xy": [0.0, 0.0],
                        "theta": 0.0,
                        "az": az,
                        "el": el,
                        "jitter": [0, 0, 0],
                        "key_on": True,
                        "samples": 64,
                        "key_strength": key_strength,
                        **opts,
                    }
                )
    return jobs


def lum(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) @ LUM


def analyze(render_dir: Path, out: Path) -> pd.DataFrame:
    rows = []
    p = np.array([0.0, 0.0, 1.0])
    for name in VARIANTS:
        for az, el in ANGLES:
            stem = render_dir / f"{name}_az{az:03d}_el{el}"
            obj_path, bg_path = Path(f"{stem}_obj.png"), Path(f"{stem}_bg.png")
            if not obj_path.exists() or not bg_path.exists():
                continue
            rgb = np.asarray(Image.open(obj_path).convert("RGB"))
            mask = dilate(object_mask(rgb), 3)
            bg, img = lum(bg_path), lum(obj_path)
            floor = bg[~mask]
            shadow = np.clip(bg - img, 0, None)
            shadow[mask] = 0
            w = np.where(shadow > 4, shadow, 0)
            ys, xs = np.mgrid[0:512, 0:512]
            start, end = shadow_segment_px(p, lamp_position(az, el))
            if w.sum() > 0:
                c = np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])
                meas = float(image_angle_deg(start, c))
            else:
                meas = float("nan")
            obj_lum = img[object_mask(rgb)]
            rows.append(
                {
                    "variant": name,
                    "az": az,
                    "el": el,
                    "floor_mean": float(floor.mean()),
                    # The highlight measure: brightest floor region against the median floor.
                    "floor_highlight": float(np.percentile(floor, 99) - np.median(floor)),
                    "shadow_depth": float(np.percentile(shadow[~mask], 99.5)),
                    "shadow_area_px": int((shadow > 4).sum()),
                    "shadow_dir_error": float(angle_diff_deg(meas, image_angle_deg(start, end))),
                    "object_contrast": float(np.percentile(obj_lum, 95) - np.percentile(obj_lum, 5)),
                }
            )
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "lighting_study.csv", index=False, float_format="%.2f")
    summary = df.groupby("variant", sort=False)[
        [
            "floor_mean",
            "floor_highlight",
            "shadow_depth",
            "shadow_area_px",
            "shadow_dir_error",
            "object_contrast",
        ]
    ].mean()
    (out / "lighting_study.md").write_text(summary.to_markdown(floatfmt=".1f") + "\n")
    print(summary.round(1).to_string())

    fig, axes = plt.subplots(len(VARIANTS), 3, figsize=(9, 3 * len(VARIANTS)))
    for row, name in zip(axes, VARIANTS, strict=True):
        for ax, (az, el) in zip(row, ANGLES[:3], strict=True):
            path = render_dir / f"{name}_az{az:03d}_el{el}_obj.png"
            if path.exists():
                ax.imshow(Image.open(path).convert("RGB"))
                ang = np.radians(image_angle_deg(*shadow_segment_px(p, lamp_position(az, el))) + 180)
                c = project(np.zeros(3))[:2]
                ax.annotate(
                    "",
                    xy=tuple(c + 120 * np.array([np.cos(ang), -np.sin(ang)])),
                    xytext=tuple(c),
                    arrowprops={"arrowstyle": "->", "color": "#ffd600", "lw": 2.5},
                )
            ax.set_title(f"{name}, az={az}, el={el}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "lighting_study.png", dpi=100)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec")
    s.add_argument("--render-dir", type=Path, default=Path("data/audit_renders/lighting"))
    s.add_argument("--spec", type=Path, required=True)
    s.add_argument("--key-strength", type=float, default=3.0)
    a = sub.add_parser("analyze")
    a.add_argument("--render-dir", type=Path, default=Path("data/audit_renders/lighting"))
    a.add_argument("--out", type=Path, default=Path("reports/audit"))
    args = parser.parse_args()
    if args.cmd == "spec":
        args.render_dir.mkdir(parents=True, exist_ok=True)
        jobs = spec(args.render_dir.resolve(), args.key_strength)
        args.spec.write_text(json.dumps(jobs, indent=1))
        print(f"{len(jobs)} jobs -> {args.spec}")
    else:
        analyze(args.render_dir, args.out)


if __name__ == "__main__":
    main()
