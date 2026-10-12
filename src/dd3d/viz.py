"""Draw label and prediction arrows with the exact camera of each scene.

The arrows show the cast shadow of the key lamp: they start at the object base
and end at the shadow of the object centre. A long arrow means a low lamp.
The ZZSN figures used a different camera and an orthographic projection; see
docs/audit/labels_and_arrows.md for the comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from dd3d.geometry import (  # noqa: E402
    BASE_CAMERA,
    SUN_TRAVEL_DIRECTION,
    fit_camera_jitter,
    lamp_position,
    project,
    shadow_segment_px,
)
from dd3d.metrics import circular_abs_error  # noqa: E402

GT_COLOR = "#00c853"
PRED_COLOR = "#ff1744"
SUN_COLOR = "#9e9e9e"


def load_scene(scenes_dir: Path, image_id: str) -> dict:
    shape = image_id.rsplit("_new_", 1)[0]
    return json.loads((scenes_dir / shape / f"{image_id}.json").read_text())


def scene_camera(scene: dict) -> np.ndarray:
    obj = scene["objects"][0]
    jitter, _ = fit_camera_jitter(obj["3d_coords"], obj["pixel_coords"])
    return BASE_CAMERA + jitter


def sun_segment_px(obj_xyz, cam_pos, length: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the image segment of the constant sun shadow on the ground."""
    p = np.asarray(obj_xyz, dtype=float)
    base = p * np.array([1.0, 1.0, 0.0])
    ground = SUN_TRAVEL_DIRECTION * np.array([1.0, 1.0, 0.0])
    tip = base + length * ground / np.linalg.norm(ground)
    return project(base, cam_pos)[:2], project(tip, cam_pos)[:2]


def clip_segment(start, end, width: int, height: int, margin: float = 4.0) -> np.ndarray:
    """Return the end point, moved back along the segment until it is inside the image.

    Matplotlib does not draw an annotation arrow with its end outside the axes.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    d = end - start
    t = 1.0
    for axis, size in ((0, width), (1, height)):
        if d[axis] > 0 and end[axis] > size - margin:
            t = min(t, (size - margin - start[axis]) / d[axis])
        if d[axis] < 0 and end[axis] < margin:
            t = min(t, (margin - start[axis]) / d[axis])
    return start + max(t, 0.0) * d


def draw_prediction(
    image_path: Path,
    scene: dict,
    pred_az: float,
    pred_el: float,
    out_path: Path,
    show_sun: bool = False,
) -> None:
    """Save the image with the label arrow, the prediction arrow and the sun shadow."""
    img = Image.open(image_path).convert("RGB")
    obj = scene["objects"][0]
    p = np.asarray(obj["3d_coords"], dtype=float)
    cam = scene_camera(scene)
    gt_az, gt_el = scene["light_azimuth_deg"], scene["light_elevation_deg"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")
    arrows = [
        (shadow_segment_px(p, lamp_position(gt_az, gt_el), cam), GT_COLOR, "label (key lamp shadow)"),
        (shadow_segment_px(p, lamp_position(pred_az, pred_el), cam), PRED_COLOR, "prediction"),
    ]
    if show_sun:
        arrows.append((sun_segment_px(p, cam), SUN_COLOR, "constant sun lamp shadow"))
    for (start, end), color, label in arrows:
        end = clip_segment(start, end, img.width, img.height)
        ax.annotate(
            "",
            xy=tuple(end),
            xytext=tuple(start),
            arrowprops={"arrowstyle": "->,head_width=0.5,head_length=0.5", "color": color, "lw": 2.5},
        )
        ax.plot([], [], color=color, lw=2.5, label=label)
    start, _ = arrows[0][0]
    ax.plot(*start, "o", color="white", markeredgecolor="black", markersize=5)
    az_err = float(circular_abs_error(pred_az, gt_az))
    ax.set_title(
        f"label az={gt_az:.1f} el={gt_el:.1f} | pred az={pred_az:.1f} el={pred_el:.1f}\n"
        f"error az={az_err:.1f} el={abs(pred_el - gt_el):.1f}",
        fontsize=8,
    )
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
