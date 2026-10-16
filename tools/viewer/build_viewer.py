"""Build the light viewer: one HTML file with the sweep and patch data inside.

The inputs are the JSON files of scripts/experiments/sweep_package.py. The script
adds the direction of the dark shadow region of each output, a second measure
next to the pixel probe. The output is a single HTML file that works offline and
can also be published.

Usage: python tools/viewer/build_viewer.py --sweep sweep.json --patch patch.json --out viewer.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from dd3d.cues import shadow_azimuth, shadow_kept

TEMPLATE = Path(__file__).resolve().parent / "template.html"


def decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def add_shadow(block: dict) -> None:
    """Add two shadow measures to each output.

    shadow_az is the azimuth of the dark shadow region (dd3d.cues.shadow_azimuth).
    kept is the share of the crisp shadow of the base output that stays in place
    (dd3d.cues.shadow_kept); for a patch, shown is the share of the crisp shadow
    of the source output that appears.
    """
    for scene in block["scenes"]:
        base = next(o for o in scene["outputs"] if o["condition"] == "base")
        source = next((o for o in scene["outputs"] if o["condition"] == "source"), None)
        base_rgb = decode(base["image"])
        source_rgb = decode(source["image"]) if source else None
        for out in scene["outputs"]:
            rgb = decode(out["image"])
            az = shadow_azimuth(rgb)
            out["shadow_az"] = None if np.isnan(az) else round(float(az), 1)
            kept = shadow_kept(base_rgb, rgb)
            out["kept"] = None if np.isnan(kept) else round(kept, 2)
            if source_rgb is not None:
                shown = shadow_kept(source_rgb, rgb)
                out["shown"] = None if np.isnan(shown) else round(shown, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, action="append", default=[], help="One file for each block.")
    parser.add_argument("--patch", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = {"sweeps": []}
    for path in args.sweep:
        sweep = json.loads(path.read_text())
        add_shadow(sweep)
        m = re.search(r"b(\d+)", path.stem)
        sweep["block"] = int(m.group(1)) if m else None
        data["sweeps"].append(sweep)
    if args.patch is not None:
        data["patch"] = json.loads(args.patch.read_text())
        add_shadow(data["patch"])
    # A "</" inside the JSON would close the script element of the page.
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__DATA__", text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"{args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
