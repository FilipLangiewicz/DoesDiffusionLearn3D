"""Write a causal plan that shifts the pattern direction in many blocks at the same time.

For each listed block, a least-squares regression of the grid features on
(sin az, cos az) over the train split gives the activation pattern of the light.
make_intervention_plan.py describes this pattern. The shift of a block is k times the mean feature
change between images whose light differs by the rotation angle. All blocks get
their shift in the same forward pass (dd3d.intervene, plan key shift_blocks).

The conditions are base and pattern_all_x<k> for each k in --scales. The last
condition, random_all, adds a random vector in each block. Its cell norms are the
same as in the largest pattern shift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from make_intervention_plan import pattern_of

from dd3d.data import pool_grid
from dd3d.metrics import encode_targets
from dd3d.store import FeatureStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Feature root (dd3d.store).")
    parser.add_argument("--run", default="sana")
    parser.add_argument("--blocks", default=",".join(str(b) for b in range(20)))
    parser.add_argument("--t", type=int, default=100)
    parser.add_argument("--grid", type=int, default=4, help="Grid of the features of every block.")
    parser.add_argument("--channels", type=int, default=2240)
    parser.add_argument("--splits", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-images", type=int, default=48)
    parser.add_argument("--scales", default="0.25,0.5,1")
    parser.add_argument("--rotate", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    blocks = [int(b) for b in args.blocks.split(",")]
    train = pd.read_csv(args.splits / "train.csv")
    sub = pd.read_csv(args.splits / f"{args.split}.csv")
    pick = sub.sample(min(args.n_images, len(sub)), random_state=args.seed)
    store = FeatureStore(args.features)
    z_train = encode_targets(train["light_azimuth_deg"], train["light_elevation_deg"])[:, :2]
    az = pick["light_azimuth_deg"].to_numpy()
    el = pick["light_elevation_deg"].to_numpy()
    delta_y = encode_targets(az + args.rotate, el)[:, :2] - encode_targets(az, el)[:, :2]

    d = args.grid * args.grid * args.channels
    natural = np.zeros((len(pick), len(blocks), d), dtype=np.float32)
    norms = {}
    for j, b in enumerate(blocks):
        key = f"{args.run}/block_{b:02d}/t_{args.t:03d}"
        native = store.meta(args.run)["grid_of_block"][str(b)]
        x = pool_grid(store.load(key, train["image_id"].tolist()), native, args.grid)
        pattern = pattern_of(x, z_train)
        natural[:, j] = delta_y @ pattern
        cells = natural[:, j].reshape(len(pick), -1, args.channels)
        norms[b] = float(np.linalg.norm(cells, axis=2).mean())
        print(f"block {b}: mean cell norm of the natural shift {norms[b]:.2f}", flush=True)
        del x

    scales = [float(v) for v in args.scales.split(",")]
    conditions, shifts = ["base"], [np.zeros_like(natural)]
    for k in scales:
        conditions.append(f"shift:pattern_all_x{k:g}")
        shifts.append(k * natural)
    big = max(scales) * natural
    rand = rng.normal(size=big.shape).astype(np.float32)
    cells_r = rand.reshape(len(pick), len(blocks), -1, args.channels)
    cells_b = big.reshape(len(pick), len(blocks), -1, args.channels)
    cells_r *= (np.linalg.norm(cells_b, axis=3) / np.linalg.norm(cells_r, axis=3))[..., None]
    conditions.append("shift:random_all")
    shifts.append(cells_r.reshape(big.shape))
    shifts = np.stack(shifts, axis=1).astype(np.float16)  # (n_img, n_cond, n_blocks, D)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out / "plan.npz",
        image_ids=pick["image_id"].to_numpy().astype(str),
        shapes=pick["shape"].to_numpy().astype(str),
        azimuth=az,
        conditions=np.array(conditions),
        shifts=shifts,
        shift_blocks=np.array(blocks),
    )
    report = {
        "blocks": blocks,
        "t": args.t,
        "n_images": len(pick),
        "conditions": conditions,
        "natural_cell_norm": norms,
    }
    (args.out / "plan.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
