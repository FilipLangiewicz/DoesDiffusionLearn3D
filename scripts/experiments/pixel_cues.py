"""Probe the light direction from pixel cue variants (no diffusion model).

The script fits the same probe (PCA, standard scores, Ridge, alpha from val) on
four pixel variants and on two label types. It writes a CSV and a Markdown table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dd3d.cues import cue_features
from dd3d.data import add_scene_columns, load_splits, target_columns
from dd3d.probe import ProbeConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/final/images"))
    parser.add_argument("--scenes", type=Path, default=Path("data/final/scenes"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--out", type=Path, default=Path("reports/experiments/pixel_cues"))
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    splits = {k: add_scene_columns(v, args.scenes) for k, v in load_splits(args.splits).items()}
    feats = {k: cue_features(v, args.images, args.resolution) for k, v in splits.items()}
    rows = []
    for variant in feats["train"]:
        for target in ("global", "local"):
            az_col, el_col = target_columns(target)
            labels = {k: (v[az_col].to_numpy(), v[el_col].to_numpy()) for k, v in splits.items()}
            n_comp = 5 if variant.startswith("floor highlight") else 256
            res = run({k: f[variant] for k, f in feats.items()}, labels, ProbeConfig(n_components=n_comp))
            for split in ("test", "test_cubes"):
                s = res.splits[split]
                rows.append(
                    {
                        "variant": variant,
                        "target": target,
                        "split": split,
                        "alpha": res.alpha,
                        "az_mae": s["az_mae"],
                        "az_ci_low": s["az_mae_ci"][0],
                        "az_ci_high": s["az_mae_ci"][1],
                        "el_mae": s["el_mae"],
                        "el_ci_low": s["el_mae_ci"][0],
                        "el_ci_high": s["el_mae_ci"][1],
                    }
                )
            print(
                f"{variant:40s} {target:6s} test {res.splits['test']['az_mae']:6.2f} "
                f"cubes {res.splits['test_cubes']['az_mae']:6.2f}"
                f" / el {res.splits['test_cubes']['el_mae']:5.2f}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "results.csv", index=False, float_format="%.3f")
    (args.out / "results.md").write_text(df.to_markdown(index=False, floatfmt=".2f") + "\n")
    (args.out / "config.json").write_text(json.dumps({"resolution": args.resolution}, indent=1))


if __name__ == "__main__":
    main()
