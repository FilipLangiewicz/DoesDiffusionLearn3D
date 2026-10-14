"""Write the main result table (RQ1) from the scan files of all data sets.

The rows are the data sets and the feature sources: pixels, VAE latent, SANA,
SANA with random weights, the old noise rule, SD 1.5 and SDXL. Each row gives
the azimuth MAE with its 95% interval on the test split (spheres and cylinders)
and on the cubes. For a model, the val split alone selects the block, the
timestep and the grid. The criterion is the sum of the azimuth and elevation MAE.

Output: a Markdown table and a LaTeX table (docs/article/tables/rq1.tex).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LABELS = {
    "pixels": "Pixels ($64\\times64$)",
    "vae_latent": "DC-AE latent",
    "sana": "SANA-1.6B",
    "sana_random_init": "SANA-1.6B, random weights",
    "sana_legacy_noise": "SANA-1.6B, original noise rule",
    "sd15": "Stable Diffusion 1.5",
    "sdxl": "SDXL",
}


def best_rows(scan: pd.DataFrame) -> pd.DataFrame:
    """Return, for each run, the rows of the block, timestep and grid chosen on val."""
    out = []
    for _run, g in scan.groupby("run"):
        val = g[g["split"] == "val"]
        key = val.loc[(val["az_mae"] + val["el_mae"]).idxmin(), ["block", "t", "grid"]]
        sel = g[(g["block"] == key["block"]) & (g["t"] == key["t"]) & (g["grid"] == key["grid"])]
        out.append(sel)
    return pd.concat(out)


def cell(rows: pd.DataFrame, split: str, latex: bool) -> str:
    r = rows[rows["split"] == split]
    if r.empty:
        return "--"
    r = r.iloc[0]
    if latex:
        return f"${r['az_mae']:.1f}$ \\scriptsize{{[{r['az_ci_low']:.1f}, {r['az_ci_high']:.1f}]}}"
    return f"{r['az_mae']:.1f} [{r['az_ci_low']:.1f}, {r['az_ci_high']:.1f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", default="v1,v2,v2r")
    parser.add_argument("--reports", type=Path, default=Path("reports/experiments"))
    parser.add_argument("--latex", type=Path, default=Path("docs/article/tables/rq1.tex"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/experiments/rq1.md"))
    args = parser.parse_args()

    md = ["| Data | Features | Block, t, grid | Test az MAE | Cubes az MAE |", "|---|---|---|---|---|"]
    tex = [
        "\\begin{tabular}{llrll}",
        "\\toprule",
        "Data & Features & Block, $t$ & Test (az.) & Cubes (az.) \\\\",
        "\\midrule",
    ]
    for name in args.sets.split(","):
        path = args.reports / name / "scan" / "scan.csv"
        if not path.exists():
            continue
        scan = pd.read_csv(path)
        scan = scan[scan["target"] == "global"]
        best = best_rows(scan)
        for run in LABELS:
            rows = best[best["run"] == run]
            if rows.empty:
                continue
            r0 = rows.iloc[0]
            where = "--" if r0["block"] < 0 else f"{int(r0['block'])}, {int(r0['t'])}, {int(r0['grid'])}"
            test, cubes = cell(rows, "test", False), cell(rows, "test_cubes", False)
            md.append(f"| {name} | {run} | {where} | {test} | {cubes} |")
            test, cubes = cell(rows, "test", True), cell(rows, "test_cubes", True)
            tex.append(f"{name} & {LABELS[run]} & {where} & {test} & {cubes} \\\\")
        tex.append("\\midrule")
    if tex[-1] == "\\midrule":
        tex[-1] = "\\bottomrule"
    tex.append("\\end{tabular}")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(md) + "\n")
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    args.latex.write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
