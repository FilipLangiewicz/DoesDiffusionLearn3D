"""Write three example images of each data set version and one overview sheet.

The new versions (v2, v2ns, v2r, v2rns, v2rso, v2rot) use the same scene
numbers. Thus each row of the sheet differs from the v2 row in one property.
v1 and v2pos have their own scenes. The images are 256 px JPEG files, so the
folder data/examples stays near 1 MB and can be in git.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCENES = ("Sphere_00076", "SmoothCylinder_00157", "SmoothCube_v2_00008")
ROWS = [
    ("v1", "original data (ZZSN): area lamp around the centre, sun lamp on", None),
    ("v2", "one sun lamp, flat floor, object at the centre", "v2"),
    ("v2ns", "as v2, the object casts no shadow", "v2ns"),
    ("v2pos", "as v2, object at a random position (test only, own scenes)", None),
    ("v2r", "as v2, object at a random position", "v2r"),
    ("v2rns", "as v2r, the object casts no shadow", "v2rns"),
    ("v2rso", "as v2r, only the shadow (the camera does not see the object)", "v2rso"),
    ("v2rot", "as v2, light azimuth + 90 deg (pairs for the patch test)", "v2rot"),
]
V1 = ("Sphere_new_000000", "SmoothCylinder_new_000000", "SmoothCube_v2_new_000000")
V2POS = ("v2pos_Sphere_00000", "v2pos_SmoothCylinder_00000", "v2pos_SmoothCube_v2_00000")


def source(name: str, scene: str, roots: dict) -> Path | None:
    shape = scene.rsplit("_", 1)[0]
    if name == "v1":
        i = SCENES.index(scene)
        return roots["v1"] / V1[i].rsplit("_new", 1)[0] / f"{V1[i]}.png"
    if name == "v2pos":
        i = SCENES.index(scene)
        return roots["v2pos"] / V2POS[i].split("_", 1)[1].rsplit("_", 1)[0] / f"{V2POS[i]}.png"
    path = roots[name] / shape / f"{name}_{scene}.png"
    return path if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--v2rot", type=Path, required=True, help="Image folder of v2rot.")
    parser.add_argument("--out", type=Path, default=Path("data/examples"))
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    roots = {n: args.data / n / "images" for n in ("v2", "v2ns", "v2pos", "v2r", "v2rns", "v2rso")}
    roots["v1"] = args.data / "final" / "images"
    roots["v2rot"] = args.v2rot
    cell, label_w, pad = 200, 330, 6
    sheet = Image.new("RGB", (label_w + 3 * (cell + pad), len(ROWS) * (cell + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=13)
    for r, (name, desc, _) in enumerate(ROWS):
        y = pad + r * (cell + pad)
        draw.text((10, y + 70), name, fill="black", font=font)
        words, line, lines = desc.split(), "", []
        for w in words:
            if len(line) + len(w) > 38:
                lines.append(line)
                line = ""
            line = f"{line} {w}".strip()
        lines.append(line)
        for k, text in enumerate(lines):
            draw.text((10, y + 96 + 16 * k), text, fill="#444444", font=small)
        (args.out / name).mkdir(parents=True, exist_ok=True)
        for c, scene in enumerate(SCENES):
            path = source(name, scene, roots)
            if path is None or not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            img.resize((args.size, args.size), Image.BICUBIC).save(
                args.out / name / f"{path.stem}.jpg", quality=85
            )
            sheet.paste(img.resize((cell, cell), Image.BICUBIC), (label_w + c * (cell + pad), y))
    sheet.save(args.out / "overview.jpg", quality=85)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
