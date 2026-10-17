"""Audit check: do the LinAlgWarning messages of the pixel probes change the results?

The pixel probes keep at most 256 PCA components. For the 8 x 8 grid this is every
component, so some components have almost no variance. With a small Ridge alpha
and float32 features, the solver reports an ill-conditioned matrix. This script
fits each pixel grid with float32 features (as in the experiments) and with
float64 features. It gives the selected alpha, the errors and the number of
warnings of each fit.

Output: precision.csv and precision.md in --out.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dd3d.data import pixel_features
from dd3d.probe import ProbeConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sets", default="v2,v2r,v2rns,v2rso")
    parser.add_argument("--grids", default="8,16,32,64")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for name in args.sets.split(","):
        folder = args.data_root / name
        splits = {
            s: pd.read_csv(folder / "splits" / f"{s}.csv") for s in ("train", "val", "test", "test_cubes")
        }
        labels = {
            k: (v["light_azimuth_deg"].to_numpy(), v["light_elevation_deg"].to_numpy())
            for k, v in splits.items()
        }
        for grid in (int(g) for g in args.grids.split(",")):
            base = {k: pixel_features(v, folder / "images", grid) for k, v in splits.items()}
            for dtype in (np.float32, np.float64):
                feats = {k: v.astype(dtype) for k, v in base.items()}
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    res = run(feats, labels, ProbeConfig(n_components=min(256, 3 * grid * grid)))
                rows.append(
                    {
                        "set": name,
                        "grid": grid,
                        "dtype": np.dtype(dtype).name,
                        "alpha": res.alpha,
                        "warnings": sum("ill-conditioned" in str(w.message).lower() for w in caught),
                        "val_az": res.splits["val"]["az_mae"],
                        "test_az": res.splits["test"]["az_mae"],
                        "cubes_az": res.splits["test_cubes"]["az_mae"],
                    }
                )
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "precision.csv", index=False, float_format="%.4f")
    (args.out / "precision.md").write_text(df.to_markdown(index=False, floatfmt=".3f") + "\n")


if __name__ == "__main__":
    main()
