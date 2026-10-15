"""Write the plan of the patch test with activations (input of dd3d.intervene.run_patch).

Each pair has a target render of v2 and a source render of v2rot. The source
has the same scene, but its light azimuth is the target azimuth + offset. The
plan copies the source tokens of one region in one block into the target pass.
The regions come from the scene geometry (see probe_maps.regions):

- all: every token (a positive control, in one block only);
- object: the tokens that show the object;
- shadow_src: the tokens of the source shadow, outside the object;
- shadow_tgt: the tokens of the target shadow, outside the object;
- floor: all other tokens.

Two more conditions patch the object tokens of one block only in the first half
or only in the second half of the steps.

The shadow of the output can turn towards the source light. Then the patched
tokens carry the light direction that the model uses to draw the shadow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from probe_maps import regions

from dd3d.intervene import cell_mask_to_tokens

REGIONS = ("all", "object", "shadow_src", "shadow_tgt", "floor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/v2/labels.csv"))
    parser.add_argument("--only", required=True, help="Comma list <shape>_<i> of the rendered pairs.")
    parser.add_argument("--offset", type=float, default=90.0, help="Source azimuth - target azimuth.")
    parser.add_argument("--blocks", default="0,2,4,6,8,10,12,14,16,18")
    parser.add_argument("--regions", default="object,shadow_src,floor", help="Regions for every block.")
    parser.add_argument("--control-block", type=int, default=10, help="Block of the all-token control.")
    parser.add_argument(
        "--time-block", type=int, default=10, help="Block of the early and late object patch."
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--grid", type=int, default=16, help="Grid of the region masks.")
    parser.add_argument("--side", type=int, default=32, help="Token grid of the transformer.")
    parser.add_argument("--parts", type=int, default=2, help="Number of plan files (one for each GPU).")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lab = pd.read_csv(args.labels).set_index("image_id")
    ids = [f"v2_{k}" for k in args.only.split(",")]
    blocks = [int(b) for b in args.blocks.split(",")]
    regions_of = [r for r in args.regions.split(",") if r]
    all_steps = np.ones(args.steps, dtype=bool)
    # Each entry: (name, block, region, step mask).
    plan = [("base", -1, "all", all_steps)]
    if args.control_block >= 0:
        plan.append((f"patch:b{args.control_block:02d}_all", args.control_block, "all", all_steps))
    for b in blocks:
        for r in regions_of:
            plan.append((f"patch:b{b:02d}_{r}", b, r, all_steps))
    if args.time_block >= 0:
        half = args.steps // 2
        early, late = np.arange(args.steps) < half, np.arange(args.steps) >= half
        plan.append((f"patch:b{args.time_block:02d}_object_early", args.time_block, "object", early))
        plan.append((f"patch:b{args.time_block:02d}_object_late", args.time_block, "object", late))
    conditions = [c[0] for c in plan]
    cond_blocks = [c[1] for c in plan]
    step_masks = np.stack([c[3] for c in plan])
    masks = np.zeros((len(ids), len(conditions), args.side * args.side), dtype=bool)
    stats = []
    for i, image_id in enumerate(ids):
        row = lab.loc[image_id]
        obj, sh_tgt = regions(row, args.grid)
        src = row.copy()
        src["light_azimuth_deg"] = (row["light_azimuth_deg"] + args.offset) % 360.0
        _, sh_src = regions(src, args.grid)
        cells = {
            "all": np.ones_like(obj),
            "object": obj,
            "shadow_src": sh_src & ~obj,
            "shadow_tgt": sh_tgt & ~obj,
            "floor": ~(obj | sh_src | sh_tgt),
        }
        for c, (_name, _block, region, _steps) in enumerate(plan):
            if c > 0:
                masks[i, c] = cell_mask_to_tokens(cells[region], args.side)
        stats.append({k: float(v.mean()) for k, v in cells.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    shapes = [lab.loc[k, "shape"] for k in ids]
    parts = np.array_split(np.arange(len(ids)), args.parts)
    for p, sel in enumerate(parts):
        np.savez(
            args.out / f"patch_part{p}.npz",
            target_ids=np.array(ids)[sel],
            source_ids=np.array([k.replace("v2_", "v2rot_", 1) for k in ids])[sel],
            shapes=np.array(shapes)[sel],
            azimuth=lab.loc[ids, "light_azimuth_deg"].to_numpy()[sel],
            conditions=np.array(conditions),
            blocks=np.array(cond_blocks),
            step_masks=step_masks,
            masks=masks[sel],
        )
    report = {
        "pairs": len(ids),
        "offset": args.offset,
        "conditions": conditions,
        "mean_area": pd.DataFrame(stats).mean().round(4).to_dict(),
    }
    (args.out / "patch_plan.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["mean_area"]), len(conditions), "conditions")


if __name__ == "__main__":
    main()
