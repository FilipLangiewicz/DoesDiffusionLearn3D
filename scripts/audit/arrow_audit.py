"""Compare the ZZSN arrows with the exact shadow geometry for all scenes.

For each scene, the script fits the camera jitter and projects the cast shadow
of the key lamp with the full camera model. Then it compares the result with
the arrow of the ZZSN script visualize_predictions.py. It also measures how far
the label (lamp around the origin) is from the light direction at the object.
The output is one CSV row for each scene and a short text summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dd3d.geometry import (
    BASE_CAMERA,
    angle_diff_deg,
    fit_camera_jitter,
    image_angle_deg,
    lamp_position,
    legacy_arrow_angle_deg,
    local_light_direction,
    project,
    shadow_segment_px,
)


def audit_scene(scene_path: Path) -> dict:
    data = json.loads(scene_path.read_text())
    obj = data["objects"][0]
    p = np.array(obj["3d_coords"], dtype=float)
    az, el = data["light_azimuth_deg"], data["light_elevation_deg"]
    lamp = lamp_position(az, el)
    jitter, residual = fit_camera_jitter(p, obj["pixel_coords"])
    cam = BASE_CAMERA + jitter
    start, end = shadow_segment_px(p, lamp, cam)
    true_angle = float(image_angle_deg(start, end))
    legacy_angle = float(legacy_arrow_angle_deg(p, lamp))
    center_px = project(p, cam)[:2]
    loc_az, loc_el = local_light_direction(p, az, el)
    return {
        "image_id": scene_path.stem,
        "shape": scene_path.parent.name,
        "az": az,
        "el": el,
        "x": p[0],
        "y": p[1],
        "z": p[2],
        "jitter_x": jitter[0],
        "jitter_y": jitter[1],
        "jitter_z": jitter[2],
        "fit_residual_px": residual,
        "true_shadow_angle": true_angle,
        "legacy_arrow_angle": legacy_angle,
        "legacy_arrow_error": float(angle_diff_deg(true_angle, legacy_angle)),
        "shadow_len_px": float(np.linalg.norm(end - start)),
        "base_to_center_px": float(np.linalg.norm(center_px - start)),
        "local_az": float(loc_az),
        "local_el": float(loc_el),
        "label_vs_local_az": float(angle_diff_deg(az, loc_az)),
        "label_vs_local_el": float(loc_el - el),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, default=Path("data/final/scenes"))
    parser.add_argument("--out", type=Path, default=Path("reports/audit/arrow_audit.csv"))
    args = parser.parse_args()
    paths = sorted(args.scenes.glob("*/*_new_*.json"))
    df = pd.DataFrame([audit_scene(p) for p in paths])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.4f")

    err = df["legacy_arrow_error"]
    lines = [
        f"scenes: {len(df)}",
        f"camera fit: max residual {df['fit_residual_px'].max():.2f} px",
        f"legacy arrow vs exact shadow direction: mean {err.mean():.2f} deg, median {err.median():.2f}, "
        f"p90 {err.quantile(0.9):.2f}, max {err.max():.2f}",
        f"share of scenes with legacy arrow error > 10 deg: {(err > 10).mean():.3f}",
        f"legacy arrow origin: {df['base_to_center_px'].mean():.1f} px above the shadow start",
        f"label vs light at object, azimuth: mean {df['label_vs_local_az'].mean():.2f} deg, "
        f"median {df['label_vs_local_az'].median():.2f}",
        f"label vs light at object, elevation: mean {df['label_vs_local_el'].mean():+.2f} deg, "
        f"mean abs {df['label_vs_local_el'].abs().mean():.2f}",
    ]
    for line in lines:
        print(line)
    args.out.with_suffix(".txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
