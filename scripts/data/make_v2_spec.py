"""Write the render spec and the labels of dataset v2.

Dataset v2 has one light source, a SUN key lamp. The light direction is the
same at every point, so the label is exact for the object. Other rules:

- the sun lamp Area of the CLEVR base scene is off;
- a flat ground replaces the curved backdrop (its wall blocks a sun from behind);
- no camera jitter;
- the object stands at the scene centre (option --random-position for a test set);
- the object rotation is random and in radians;
- the object size is uniform in [--size-min, --size-max].

The labels file has one row for each image with all render parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

# The Kaggle render job runs this script without an installed dd3d package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dd3d.geometry import object_in_frame  # noqa: E402

SHAPES = ("Sphere", "SmoothCylinder", "SmoothCube_v2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/v2"))
    parser.add_argument("--per-shape", type=int, default=1000)
    parser.add_argument("--el-min", type=float, default=30.0)
    parser.add_argument("--el-max", type=float, default=75.0)
    parser.add_argument("--size-min", type=float, default=0.8)
    parser.add_argument("--size-max", type=float, default=1.4)
    parser.add_argument("--random-position", type=float, default=0.0, help="Half-width of the x and y range.")
    parser.add_argument("--key-strength", type=float, default=3.0)
    parser.add_argument("--world-strength", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prefix", default="v2")
    parser.add_argument(
        "--frame-position",
        action="store_true",
        help="Random position in [-3, 3]^2 where the largest object is fully in the image.",
    )
    parser.add_argument("--no-object-shadow", action="store_true", help="The object casts no shadow.")
    parser.add_argument("--shadow-only", action="store_true", help="The camera sees only the shadow.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    # A second generator for the positions keeps the light, size and rotation
    # equal to those of the same seed without --frame-position.
    pos_rng = random.Random(args.seed + 1_000_003)
    image_dir = (args.out_dir / "images").resolve()
    jobs, rows = [], []
    for shape in SHAPES:
        (image_dir / shape).mkdir(parents=True, exist_ok=True)
        for i in range(args.per_shape):
            image_id = f"{args.prefix}_{shape}_{i:05d}"
            r = args.random_position
            row = {
                "image_id": image_id,
                "shape": shape,
                "light_azimuth_deg": round(rng.uniform(0.0, 360.0), 2),
                "light_elevation_deg": round(rng.uniform(args.el_min, args.el_max), 2),
                "size": round(rng.uniform(args.size_min, args.size_max), 3),
                "rotation_rad": round(rng.uniform(0.0, 2 * math.pi), 4),
                "x": round(rng.uniform(-r, r), 3) if r else 0.0,
                "y": round(rng.uniform(-r, r), 3) if r else 0.0,
            }
            if args.frame_position:
                # The test uses the largest size, so the region does not depend on the
                # light or on the size of this object.
                while True:
                    x, y = pos_rng.uniform(-3.0, 3.0), pos_rng.uniform(-3.0, 3.0)
                    if object_in_frame(x, y, size=args.size_max):
                        break
                row["x"], row["y"] = round(x, 3), round(y, 3)
            rows.append(row)
            jobs.append(
                {
                    "out": str(image_dir / shape / f"{image_id}.png"),
                    "shape": shape,
                    "size": row["size"],
                    "xy": [row["x"], row["y"]],
                    # render_spec passes theta to rotation_euler, which is in radians.
                    "theta": row["rotation_rad"],
                    "az": row["light_azimuth_deg"],
                    "el": row["light_elevation_deg"],
                    "jitter": [0.0, 0.0, 0.0],
                    "key_on": True,
                    "key_type": "SUN",
                    "key_strength": args.key_strength,
                    "sun_on": False,
                    "flat_ground": True,
                    "world_strength": args.world_strength,
                    "samples": args.samples,
                    "object_shadow": not args.no_object_shadow,
                    "object_camera": not args.shadow_only,
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "labels.csv", index=False)
    (args.out_dir / "render_spec.json").write_text(json.dumps(jobs, indent=1))
    print(f"{len(jobs)} jobs -> {args.out_dir / 'render_spec.json'}")


if __name__ == "__main__":
    main()
