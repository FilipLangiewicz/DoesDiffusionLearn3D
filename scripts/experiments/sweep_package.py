"""Pack the outputs of a sweep job into one small JSON file for the interactive viewer.

For each image and condition, the file has a JPEG of the output (base64) and the
azimuth that a pixel probe reads from it. The pixel probe is fitted on the
renders of the data set. The viewer shows the render, the output for a chosen
target angle and strength, and the three directions: label, target and probe value.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dd3d.data import pixel_features
from dd3d.metrics import decode_azimuth
from dd3d.probe import ProbeConfig, run


def jpeg(path: Path, size: int, quality: int) -> str:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True, help="Output folder of the sweep plan.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--images", type=Path, default=Path("data/v2/images"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--note", default="", help="Text for the viewer (block, timestep, family).")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    splits = {n: pd.read_csv(args.splits / f"{n}.csv") for n in ("train", "val")}
    labels = {
        k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy()) for k, v in splits.items()
    }
    probe = run(
        {k: pixel_features(v, args.images, 64) for k, v in splits.items()}, labels, ProbeConfig()
    ).model

    plan = np.load(args.plan, allow_pickle=False)
    ids = [str(s) for s in plan["image_ids"]]
    shapes = [str(s) for s in plan["shapes"]]
    frame = pd.DataFrame({"image_id": ids, "shape": shapes})
    scenes = []
    for k, (image_id, shape) in enumerate(zip(ids, shapes, strict=True)):
        scenes.append(
            {
                "image_id": image_id,
                "shape": shape,
                "label_az": float(plan["azimuth"][k]),
                "render": jpeg(args.images / shape / f"{image_id}.png", args.size, args.quality),
                "outputs": [],
            }
        )
    for cond in (str(c) for c in plan["conditions"]):
        folder = args.outputs / cond.replace(":", "_")
        p = probe.predict(pixel_features(frame, folder, 64))
        read = decode_azimuth(p[:, 0], p[:, 1])
        m = re.search(r"_k([\d.]+)_r(-?[\d.]+)$", cond)
        k_val, r_val = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
        for i, scene in enumerate(scenes):
            scene["outputs"].append(
                {
                    "condition": cond,
                    "strength": k_val,
                    "rotation": r_val,
                    "read_az": float(read[i]),
                    "image": jpeg(folder / shapes[i] / f"{ids[i]}.png", args.size, args.quality),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"note": args.note, "scenes": scenes}))
    print(f"{len(scenes)} scenes -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
