"""Write the main result table (RQ1) from the outputs of best_probes.py.

Each row is one data set and one feature source. The row gives the block,
timestep and grid that the val split selected. It gives the azimuth MAE with a
95% interval on the test split (spheres and cylinders) and on the cubes. The
last column is the paired difference to the best pixel grid on the same images.

Output: reports/experiments/rq1.md and docs/article/tables/rq1.tex.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dd3d.metrics import bootstrap_ci

LABELS = {
    "pixels_best": "Pixels (best grid)",
    "vae_latent": "DC-AE latent",
    "sana_random_init": "SANA-1.6B, random weights",
    "sana_legacy_noise": "SANA-1.6B, original noise",
    "sana": "SANA-1.6B",
    "sd15": "Stable Diffusion 1.5",
    "sdxl": "SDXL",
    "dinov2": "DINOv2-L (no generation)",
    "clip": "CLIP ViT-L/14 (no generation)",
}
UNET_BLOCKS = ["down0", "down1", "down2", "mid", "up0", "up1", "up2"]
SET_NAMES = {
    "v1": "v1",
    "v2": "v2",
    "v2r": "v2r",
    "v2rns": "v2r, no shadow",
    "v2rso": "v2r, only shadow",
}


def where(run: str, sel: pd.DataFrame, pix: str) -> str:
    if run == "pixels_best":
        return f"{pix.replace('pixels', '') or '64'} px"
    if run == "vae_latent":
        return "--"
    r = sel[sel["run"] == run].iloc[0]
    block = UNET_BLOCKS[int(r["block"])] if run in ("sd15", "sdxl") else str(int(r["block"]))
    return f"{block}, {int(r['t'])}, {int(r['grid'])}"


def cell(err: np.ndarray, latex: bool) -> str:
    lo, hi = bootstrap_ci(err)
    if latex:
        return f"${err.mean():.1f}$ \\scriptsize{{[{lo:.1f}, {hi:.1f}]}}"
    return f"{err.mean():.1f} [{lo:.1f}, {hi:.1f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", default="v1,v2,v2r,v2rns,v2rso")
    parser.add_argument("--reports", type=Path, default=Path("reports/experiments"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/experiments/rq1.md"))
    parser.add_argument("--latex", type=Path, default=Path("docs/article/tables/rq1.tex"))
    args = parser.parse_args()

    md = [
        "| Data | Features | Block, t, grid | Test az MAE | Cubes az MAE | Cubes: diff. to pixels |",
        "|---|---|---|---|---|---|",
    ]
    tex = [
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "Data & Features & Block, $t$, grid & Test & Cubes & Cubes $-$ pixels \\\\",
        "\\midrule",
    ]
    for name in args.sets.split(","):
        best = args.reports / name / "best"
        if not (best / "predictions.csv").exists():
            continue
        pred = pd.read_csv(best / "predictions.csv")
        sel = pd.read_csv(best / "selected.csv")
        pix = (best / "pixels_best.txt").read_text().strip()
        pred = pd.concat([pred, pred[pred["source"] == pix].assign(source="pixels_best")])
        comp = pd.read_csv(best / "compare.csv")
        for run in LABELS:
            p = pred[pred["source"] == run]
            if p.empty:
                continue
            test = p[p["split"] == "test"]["az_err"].to_numpy()
            cubes = p[p["split"] == "test_cubes"]["az_err"].to_numpy()
            c = comp[
                (comp["source"] == run)
                & (comp["reference"] == "pixels_best")
                & (comp["split"] == "test_cubes")
            ]
            if run == "pixels_best" or c.empty:
                d_md = d_tex = "--"
            else:
                r = c.iloc[0]
                d_md = f"{r['diff']:+.1f} [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]"
                d_tex = f"${r['diff']:+.1f}$ \\scriptsize{{[{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]}}"
            w = where(run, sel, pix)
            head = f"{SET_NAMES[name]} | {LABELS[run]} | {w}"
            md.append(f"| {head} | {cell(test, False)} | {cell(cubes, False)} | {d_md} |")
            head = f"{SET_NAMES[name]} & {LABELS[run]} & {w}"
            tex.append(f"{head} & {cell(test, True)} & {cell(cubes, True)} & {d_tex} \\\\")
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
