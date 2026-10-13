"""Draw the block scan of feature_scan.py and write a summary table.

The figure shows the azimuth MAE of each run against the block, with one line
for each timestep, on one test split. The baselines (pixels, VAE latent) are
horizontal lines. The table selects the best block and timestep of each run on
the val split only, and reports the test splits for this selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True, help="scan.csv of feature_scan.py")
    parser.add_argument("--split", default="test_cubes", help="Split of the figure.")
    parser.add_argument("--grid", type=int, default=4, help="Grid of the block scan.")
    parser.add_argument("--target", default="global")
    parser.add_argument("--title", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.scan)
    df = df[df["target"] == args.target]
    runs = [r for r in df["run"].unique() if r not in ("pixels", "vae_latent")]
    if not runs:
        runs = ["pixels"]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.4), squeeze=False, sharey=True)
    for ax, run in zip(axes[0], runs, strict=True):
        sub = df[(df["run"] == run) & (df["split"] == args.split) & (df["grid"] == args.grid)]
        for t, g in sub.groupby("t"):
            g = g.sort_values("block")
            ax.plot(g["block"], g["az_mae"], marker="o", ms=3, label=f"t={t}")
        for base, style in (("pixels", "k--"), ("vae_latent", "k:")):
            b = df[(df["run"] == base) & (df["split"] == args.split)]
            if len(b):
                ax.axhline(b["az_mae"].iloc[0], ls=style[1:], color="k", lw=1, label=base)
        ax.set_title(run, fontsize=9)
        ax.set_xlabel("block")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(f"azimuth MAE on {args.split} (deg)")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(args.title, fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=150)

    # Selection on val only: the test splits never choose the block or the timestep.
    rows = []
    for (run, grid), g in df.groupby(["run", "grid"]):
        val = g[g["split"] == "val"]
        if val.empty:
            continue
        best = val.loc[(val["az_mae"] + val["el_mae"]).idxmin()]
        row = {"run": run, "grid": grid, "block": best["block"], "t": best["t"], "val_az": best["az_mae"]}
        for split in sorted(set(g["split"]) - {"train", "val"}):
            s = g[(g["split"] == split) & (g["block"] == best["block"]) & (g["t"] == best["t"])].iloc[0]
            row[f"{split}_az"] = f"{s['az_mae']:.1f} [{s['az_ci_low']:.1f}, {s['az_ci_high']:.1f}]"
            row[f"{split}_el"] = round(s["el_mae"], 1)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    args.out.with_suffix(".md").write_text(table.to_markdown(index=False, floatfmt=".1f") + "\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
