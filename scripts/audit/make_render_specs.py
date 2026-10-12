"""Write Blender render specs for the audit of the light labels and the arrows.

Two spec files go to the output folder:

- conventions.json: a sphere at the origin and Lamp_Key at four azimuths, with
  and without the sun lamp Area. One more render shows the sun lamp only.
- shadows.json: dataset scenes without the object and with Lamp_Key only. The
  differences of these renders give the real cast shadows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dd3d.geometry import fit_camera_jitter

SIZES = {"large": 1.0, "xlarge": 1.4}
DEFAULT_SCENES = [
    "SmoothCube_v2_new_000170",
    "SmoothCube_v2_new_000629",
    "SmoothCube_v2_new_000004",
    "SmoothCube_v2_new_000304",
    "Sphere_new_000000",
    "Sphere_new_000137",
    "SmoothCylinder_new_000047",
    "SmoothCylinder_new_000211",
]


def scene_job(root: Path, image_id: str) -> dict:
    shape = image_id.rsplit("_new_", 1)[0]
    data = json.loads((root / "data/final/scenes" / shape / f"{image_id}.json").read_text())
    obj = data["objects"][0]
    jitter, _ = fit_camera_jitter(obj["3d_coords"], obj["pixel_coords"])
    return {
        "shape": shape,
        "size": SIZES[obj["size"]],
        "xy": obj["3d_coords"][:2],
        "theta": obj["rotation"],
        "az": data["light_azimuth_deg"],
        "el": data["light_elevation_deg"],
        "jitter": jitter.tolist(),
        "samples": 64,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("data/audit_renders"))
    parser.add_argument("--spec-dir", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.spec_dir.mkdir(parents=True, exist_ok=True)

    conv = []
    for az in (0, 90, 180, 270):
        for sun in (True, False):
            conv.append(
                {
                    "out": str(out / f"conv_az{az:03d}_sun{int(sun)}.png"),
                    "shape": "Sphere",
                    "size": 1.0,
                    "xy": [0.0, 0.0],
                    "theta": 0.0,
                    "az": az,
                    "el": 45.0,
                    "jitter": [0, 0, 0],
                    "key_on": True,
                    "sun_on": sun,
                    "samples": 64,
                }
            )
    conv.append(
        {
            "out": str(out / "conv_sun_only.png"),
            "shape": "Sphere",
            "size": 1.0,
            "xy": [0.0, 0.0],
            "theta": 0.0,
            "az": 0,
            "el": 45.0,
            "jitter": [0, 0, 0],
            "key_on": False,
            "sun_on": True,
            "samples": 64,
        }
    )
    (args.spec_dir / "conventions.json").write_text(json.dumps(conv, indent=1))

    shadows = []
    for image_id in args.scenes:
        base = scene_job(args.root, image_id)
        variants = {
            "bg_full": {"shape": None, "key_on": True, "sun_on": True},
            "obj_key": {"key_on": True, "sun_on": False},
            "bg_key": {"shape": None, "key_on": True, "sun_on": False},
        }
        for name, change in variants.items():
            job = {**base, **change, "out": str(out / f"{image_id}_{name}.png")}
            shadows.append(job)
    (args.spec_dir / "shadows.json").write_text(json.dumps(shadows, indent=1))
    print(f"conventions: {len(conv)} jobs, shadows: {len(shadows)} jobs")


if __name__ == "__main__":
    main()
