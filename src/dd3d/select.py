"""Select the block, the timestep and the grid of each run from a probe scan.

The val split alone makes the selection, so the test splits stay unseen. The
criterion is the sum of the azimuth MAE and the elevation MAE.
"""

from __future__ import annotations

import pandas as pd

BASELINES = ("pixels", "vae_latent")


def best_rows(scan: pd.DataFrame) -> pd.DataFrame:
    """Return, for each run, the rows of the block, timestep and grid chosen on val."""
    out = []
    for _run, g in scan.groupby("run"):
        val = g[g["split"] == "val"]
        if val.empty:
            continue
        key = val.loc[(val["az_mae"] + val["el_mae"]).idxmin(), ["block", "t", "grid"]]
        sel = g[(g["block"] == key["block"]) & (g["t"] == key["t"]) & (g["grid"] == key["grid"])]
        out.append(sel)
    return pd.concat(out)


def selected(scan: pd.DataFrame, target: str = "global") -> pd.DataFrame:
    """Return one row for each run: run, block, t and grid chosen on val."""
    best = best_rows(scan[scan["target"] == target])
    return best.drop_duplicates("run")[["run", "block", "t", "grid"]].reset_index(drop=True)


def feature_key(run: str, block: int, t: int) -> str:
    """Return the dd3d.store key of a run, block and timestep."""
    return f"{run}/block_{int(block):02d}/t_{int(t):03d}"
