"""Draw the result figures of the article from the files in reports/experiments.

- scan.png: the azimuth MAE of the SANA probe on each block, for each data set.
  Each timestep has one line, and the controls are horizontal lines.
- causal.png: the rotation of the output light for each condition of the shift
  and erase test, for each block;
- patch.png: the share of the light change that a patch of one region in one
  block moves into the output.

If the input files of a figure do not exist, the script skips the figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dd3d.select import selected  # noqa: E402

CONTROLS = {
    "pixels": ("pixels", "--"),
    "vae_latent": ("DC-AE latent", ":"),
    "sana_random_init": ("SANA, random weights", "-."),
}
TITLES = {
    "v1": "v1 (original data)",
    "v2": "v2 (object at the centre)",
    "v2r": "v2r (random position)",
    "v2rns": "v2rns (no cast shadow)",
    "v2rso": "v2rso (only the shadow)",
}


def scan_figure(reports: Path, sets: list[str], split: str, grid: int, out: Path) -> None:
    scans = {s: reports / s / "scan" / "scan.csv" for s in sets}
    scans = {s: p for s, p in scans.items() if p.exists()}
    if not scans:
        print("scan figure: no scan files")
        return
    fig, axes = plt.subplots(1, len(scans), figsize=(3.6 * len(scans), 3.1), sharey=True, squeeze=False)
    for ax, (name, path) in zip(axes[0], scans.items(), strict=True):
        df = pd.read_csv(path)
        df = df[(df["target"] == "global") & (df["split"] == split)]
        sana = df[(df["run"] == "sana") & (df["grid"] == grid)]
        cmap = plt.get_cmap("viridis")
        ts = sorted(sana["t"].unique())
        for i, t in enumerate(ts):
            g = sana[sana["t"] == t].sort_values("block")
            ax.plot(
                g["block"],
                g["az_mae"],
                marker="o",
                ms=2.5,
                lw=1.2,
                color=cmap(i / max(1, len(ts) - 1)),
                label=f"SANA t={t}",
            )
        sel = selected(pd.read_csv(path))
        for run, (label, style) in CONTROLS.items():
            row = sel[sel["run"] == run]
            if row.empty:
                continue
            r = row.iloc[0]
            v = df[
                (df["run"] == run)
                & (df["block"] == r["block"])
                & (df["t"] == r["t"])
                & (df["grid"] == r["grid"])
            ]
            if len(v):
                ax.axhline(v["az_mae"].iloc[0], color="k", ls=style, lw=1, label=label)
        ax.set_title(TITLES.get(name, name), fontsize=9)
        ax.set_xlabel("block")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel(f"azimuth MAE on {split} (deg)")
    axes[0][-1].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"-> {out}")


def causal_figure(reports: Path, out: Path) -> None:
    folders = sorted((reports / "causal").glob("b*/results.csv"))
    if not folders:
        print("causal figure: no results")
        return
    frames = []
    for f in folders:
        d = pd.read_csv(f)
        d["block"] = int(f.parent.name[1:])
        frames.append(d)
    df = pd.concat(frames)
    df = df[df["condition"] != "base"]
    conds = list(dict.fromkeys(df["condition"]))
    blocks = sorted(df["block"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    width = 0.8 / len(blocks)
    for j, b in enumerate(blocks):
        g = df[df["block"] == b].set_index("condition").reindex(conds)
        ci = np.array([json.loads(str(c).replace("(", "[").replace(")", "]")) for c in g["rotation_ci"]])
        x = np.arange(len(conds)) + (j - (len(blocks) - 1) / 2) * width
        err = np.abs(ci.T - g["rotation_mean"].to_numpy())
        ax.bar(x, g["rotation_mean"], width, yerr=err, capsize=2, label=f"block {b}")
    ax.axhline(90, color="k", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(
        range(len(conds)), [c.replace("shift:", "") for c in conds], rotation=25, ha="right", fontsize=8
    )
    ax.set_ylabel("rotation of the output (deg)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"-> {out}")


def signed(v: float) -> str:
    """Format a rotation with its sign; a value that rounds to zero is 0."""
    r = round(float(v))
    return "0" if r == 0 else f"{r:+d}"


def causal_table(reports: Path, out: Path, folder: str = "causal") -> None:
    """Write a LaTeX table: one row for each condition, two measures for each block.

    The first measure is the pixel probe (mean rotation with a 95% interval). The
    second is the dark shadow region (median rotation over the images).
    """
    files = sorted((reports / folder).glob("b*/results.csv"))
    if not files:
        return
    blocks = [int(f.parent.name[1:3]) for f in files]
    frames = {b: pd.read_csv(f).set_index("condition") for b, f in zip(blocks, files, strict=True)}
    order = sorted(blocks)
    conds = [c for c in frames[order[0]].index if c != "base"]
    lines = [
        "\\begin{tabular}{l" + "rr" * len(order) + "r}",
        "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{Block {b}}}" for b in order) + " & \\\\",
        "Condition" + " & probe & shadow" * len(order) + " & pixels \\\\",
        "\\midrule",
    ]
    for c in conds:
        cells = []
        for b in order:
            r = frames[b].loc[c]
            lo, hi = (float(v) for v in str(r["rotation_ci"]).strip("()").split(","))
            cells.append(f"${signed(r['rotation_mean'])}$ \\scriptsize{{[{signed(lo)}, {signed(hi)}]}}")
            sm = r.get("shadow_rotation_median", float("nan"))
            cells.append("--" if pd.isna(sm) else f"${signed(sm)}$")
        change = np.mean([frames[b].loc[c, "pixel_change"] for b in order])
        name = c.replace("shift:", "").replace("_", "\\_")
        lines.append(f"{name} & " + " & ".join(cells) + f" & {change:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out}")


def patch_figure(reports: Path, out: Path) -> None:
    path = reports / "patch" / "summary.csv"
    if not path.exists():
        print("patch figure: no results")
        return
    df = pd.read_csv(path)
    p = df[df["condition"].str.match(r"patch:b\d+_(object|shadow_src|floor)$")].copy()
    p["block"] = p["condition"].str.extract(r"b(\d+)_")[0].astype(int)
    p["region"] = p["condition"].str.extract(r"b\d+_(\w+)$")[0]
    names = {"object": "object tokens", "shadow_src": "source shadow tokens", "floor": "other floor tokens"}
    measures = (
        ("fraction_mean", "pixel probe (mean)"),
        ("shadow_fraction_median", "cast shadow region (median)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=True)
    for ax, (col, title) in zip(axes, measures, strict=True):
        for region, label in names.items():
            g = p[p["region"] == region].sort_values("block")
            ax.plot(g["block"], g[col], marker="o", ms=3, label=label)
        for cond, style, label in (
            ("source", "--", "full source pass"),
            ("patch:b10_all", ":", "all tokens, block 10"),
        ):
            v = df[df["condition"] == cond]
            if len(v):
                ax.axhline(v[col].iloc[0], color="k", ls=style, lw=1, label=label)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("patched block")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("rotation / 90 deg")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"-> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports/experiments"))
    parser.add_argument("--sets", default="v1,v2,v2r")
    parser.add_argument("--split", default="test_cubes")
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("docs/article/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scan_figure(args.reports, args.sets.split(","), args.split, args.grid, args.out / "scan.png")
    causal_figure(args.reports, args.out / "causal.png")
    tables = args.out.parent / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    causal_table(args.reports, tables / "causal.tex")
    patch_figure(args.reports, args.out / "patch.png")


if __name__ == "__main__":
    main()
