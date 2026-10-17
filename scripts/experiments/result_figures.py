"""Draw the result figures of the article from the files in reports/experiments.

- scan.png: the azimuth MAE of the SANA probe on each block, for each data set.
  Each timestep has one line, and the controls are horizontal lines.
- causal.tex: the four measures of the output light for each condition of the
  shift test, for each block;
- erase.tex: the probe on the erased features and the output light for each eraser;
- patch.png: the share of the light change that a patch of one region in one
  block moves into the output, and the source shadow outside the patched cells.

If the input files of a figure do not exist, the script skips the figure.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dd3d.select import selected  # noqa: E402

# The models without generation get a colour, so their lines differ from the black input controls.
CONTROLS = {
    "vae_latent": ("DC-AE latent", ":", "k"),
    "sana_random_init": ("SANA, random weights", "-.", "k"),
    "dinov2": ("DINOv2-L (no generation)", "--", "tab:red"),
    "clip": ("CLIP ViT-L/14 (no generation)", ":", "tab:red"),
}


def best_pixel_line(reports: Path, name: str, split: str) -> tuple[float, str] | None:
    """Return the MAE and the label of the pixel grid that the val split selects (as in Table 4)."""
    best = reports / name / "best"
    if not (best / "pixels_best.txt").exists():
        return None
    source = (best / "pixels_best.txt").read_text().strip()
    pred = pd.read_csv(best / "predictions.csv")
    err = pred[(pred["source"] == source) & (pred["split"] == split)]["az_err"]
    return float(err.mean()), "pixels, grid selected on val"


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
        line = best_pixel_line(reports, name, split)
        if line is not None:
            ax.axhline(line[0], color="k", ls="--", lw=1, label=line[1])
        for run, (label, style, colour) in CONTROLS.items():
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
                ax.axhline(v["az_mae"].iloc[0], color=colour, ls=style, lw=1.2, label=label)
        ax.set_title(TITLES.get(name, name), fontsize=9)
        ax.set_xlabel("block")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel(f"azimuth MAE on {split} (deg)")
    handles = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out, dpi=180)
    print(f"-> {out}")


def signed(v: float) -> str:
    """Format a rotation with its sign; a value that rounds to zero is 0."""
    r = round(float(v))
    return "0" if r == 0 else f"{r:+d}"


def causal_table(reports: Path, out: Path, folder: str = "causal") -> None:
    """Write a LaTeX table: one row for each condition, one column for each measure.

    Each cell gives the values of the blocks in the order of their numbers. The
    rotations are the mean for the two probes and the median for the dark region.
    """
    files = sorted((reports / folder).glob("b*/results.csv"))
    if not files:
        return
    blocks = [int(f.parent.name[1:3]) for f in files]
    frames = {b: pd.read_csv(f).set_index("condition") for b, f in zip(blocks, files, strict=True)}
    order = sorted(blocks)
    conds = [c for c in frames[order[0]].index if c != "base"]
    columns = (
        ("rotation_mean", signed),
        ("crop_rotation_mean", signed),
        ("shadow_rotation_median", signed),
        ("shadow_kept", lambda v: f"{v:.2f}"),
        ("pixel_change", lambda v: f"{v:.0f}"),
    )
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Condition & Pixel probe & Object probe & Dark region & Kept share & Pixels \\\\",
        "\\midrule",
    ]
    for c in conds:
        cells = []
        for col, fmt in columns:
            vals = [frames[b].loc[c].get(col, float("nan")) for b in order]
            cells.append(" / ".join("--" if pd.isna(v) else f"${fmt(v)}$" for v in vals))
        name = c.replace("shift:", "").replace("_", "\\_")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out}")


def abs_turn_median(per_image: pd.DataFrame, condition: str) -> float:
    """Return the median absolute change of the pixel-probe azimuth against the base output."""
    base = per_image[per_image["condition"] == "base"].set_index("image_id")["pred_az"]
    cond = per_image[per_image["condition"] == condition].set_index("image_id")["pred_az"]
    turn = (cond - base.reindex(cond.index) + 180.0) % 360.0 - 180.0
    return float(turn.abs().median())


def erase_table(reports: Path, out: Path) -> None:
    """Write a LaTeX table of the erasure test: one block of rows for each plan.

    The pooled eraser comes from the first run. The cell eraser and its control
    come from the second run, which has both, with the same images and noise.
    """
    names = {"erase": "pooled", "erase_cell": "cell", "erase_cell_random": "cell, control"}
    rows = []
    for tag in ("b04t400", "b10t400", "b18t400", "b05t100"):
        runs = {
            "erase": (reports / "causal_erase" / tag, reports / "plans" / f"{tag}e" / "plan.json"),
            "erase_cell": (reports / "causal_erase_ctrl" / tag, reports / "plans" / f"{tag}c" / "plan.json"),
        }
        runs["erase_cell_random"] = runs["erase_cell"]
        if not all((d / "results.csv").exists() and j.exists() for d, j in runs.values()):
            print(f"erase table: no results for {tag}")
            return
        first = True
        for cond, (folder, plan_file) in runs.items():
            res = pd.read_csv(folder / "results.csv").set_index("condition")
            plan = json.loads(plan_file.read_text())
            if first:
                base_err = res.loc["base", "label_az_error"]
                probe = plan["probe"]["test"]["az_mae"]
                block = f"{int(tag[1:3])}, $t={int(tag[4:])}$"
                rows.append(
                    f"{block} & none & ${probe:.1f}$ & -- & ${base_err:.1f}$ & $0$ & $1.00$ & $0$ \\\\"
                )
                first = False
            after = plan["probe_after_erasure"][cond]["test"]
            change = plan.get("erasure_change_norm", {}).get(cond)
            change_txt = "--" if change is None else f"${100 * change / plan['mean_token_norm']:.1f}$"
            r = res.loc[cond]
            turn = abs_turn_median(pd.read_csv(folder / "per_image.csv"), cond)
            rows.append(
                f" & {names[cond]} & ${after:.1f}$ & {change_txt} & ${r['label_az_error']:.1f}$"
                f" & ${turn:.0f}$ & ${r['shadow_kept']:.2f}$ & ${r['pixel_change']:.1f}$ \\\\"
            )
        rows.append("\\midrule")
    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Features} & \\multicolumn{4}{c}{Output} \\\\",
        "Block & Eraser & Probe & Change (\\%) & Error & $|$Turn$|$ & Kept & Pixels \\\\",
        "\\midrule",
        *rows[:-1],
        "\\bottomrule",
        "\\end{tabular}",
    ]
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
    # The source shadow outside the patched cells, minus its value without a patch.
    p["outside_gain"] = p["source_shadow_outside"] - p["source_shadow_outside_base"]
    measures = (
        ("fraction_mean", "pixel probe: share of the rotation"),
        ("crop_fraction_mean", "object probe: share of the rotation"),
        ("outside_gain", "source shadow outside the patched cells\n(gain against no patch)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, (col, title) in zip(axes, measures, strict=True):
        for region, label in names.items():
            g = p[p["region"] == region].sort_values("block")
            ax.plot(g["block"], g[col], marker="o", ms=3, label=label)
        for cond, style, label in (
            ("source", "--", "full source pass"),
            ("patch:b10_all", ":", "all tokens, block 10"),
        ):
            v = df[df["condition"] == cond]
            if len(v) and col in v and pd.notna(v[col].iloc[0]):
                ax.axhline(v[col].iloc[0], color="k", ls=style, lw=1, label=label)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("patched block")
        ax.set_xticks(range(0, 19, 2))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("share")
    axes[0].legend(fontsize=7)
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
    tables = args.out.parent / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    causal_table(args.reports, tables / "causal.tex")
    erase_table(args.reports, tables / "erase.tex")
    patch_figure(args.reports, args.out / "patch.png")
    examples = args.reports / "causal_b05" / "b05t100" / "examples.png"
    if examples.exists():
        shutil.copy(examples, args.out / "causal_examples.png")
        print(f"-> {args.out / 'causal_examples.png'}")


if __name__ == "__main__":
    main()
