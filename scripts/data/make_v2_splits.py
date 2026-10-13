"""Write the splits of data set v2 with the same rules as the splits of v1.

The rules of v1: spheres and cylinders go to train (80%), val (10%) and test
(10%), and all cubes go to test_cubes. The random position set (v2pos) is one
more test split, test_pos, with all three shapes. For v2 the label is exact for
the object, so the local columns are copies of the global columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def with_local(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["local_az"] = df["light_azimuth_deg"]
    df["local_el"] = df["light_elevation_deg"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/v2/labels.csv"))
    parser.add_argument("--pos-labels", type=Path, default=Path("data/v2pos/labels.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/v2/splits"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = with_local(pd.read_csv(args.labels))
    rng = np.random.default_rng(args.seed)
    parts = {"train": [], "val": [], "test": []}
    for shape in ("Sphere", "SmoothCylinder"):
        sub = df[df["shape"] == shape].iloc[rng.permutation((df["shape"] == shape).sum())]
        n = len(sub)
        n_train, n_val = int(0.8 * n), int(0.1 * n)
        parts["train"].append(sub.iloc[:n_train])
        parts["val"].append(sub.iloc[n_train : n_train + n_val])
        parts["test"].append(sub.iloc[n_train + n_val :])
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frames in parts.items():
        pd.concat(frames).to_csv(args.out / f"{name}.csv", index=False)
    df[df["shape"] == "SmoothCube_v2"].to_csv(args.out / "test_cubes.csv", index=False)
    if args.pos_labels.exists():
        with_local(pd.read_csv(args.pos_labels)).to_csv(args.out / "test_pos.csv", index=False)
    for f in sorted(args.out.glob("*.csv")):
        print(f, len(pd.read_csv(f)))


if __name__ == "__main__":
    main()
