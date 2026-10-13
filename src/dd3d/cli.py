"""Command line interface: ``dd3d <command> --help`` shows the options."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

app = typer.Typer(add_completion=False, help="Light direction probes and audits for SANA-1.6B.")


def _load_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


@app.command()
def extract(
    config: Path = typer.Option(Path("configs/extract.yaml"), help="YAML file with ExtractConfig fields."),
    task_id: int = typer.Option(-1, help="Array task id; -1 reads SLURM_ARRAY_TASK_ID."),
    num_tasks: int = typer.Option(0, help="Number of array tasks; 0 keeps the config value."),
    limit: int = typer.Option(0, help="Only the first N images of the task (0: all)."),
    device: str = typer.Option("", help="cuda or cpu; empty keeps the config value."),
    out_dir: Path = typer.Option(None, help="Output folder; empty keeps the config value."),
    legacy_noise: bool = typer.Option(False, help="Use the ZZSN noise (only to reproduce old data)."),
    random_init: bool = typer.Option(False, help="Use a transformer with random weights (control)."),
) -> None:
    """Extract pooled activations (flow noise, fixed seeds, prompt mask)."""
    from dd3d.extract import ExtractConfig, env_task_id, run

    values = _load_yaml(config)
    for key in ("images_dir", "out_dir", "prompt_cache"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    cfg = ExtractConfig(**values)
    cfg.task_id = task_id if task_id >= 0 else env_task_id(cfg.task_id)
    if num_tasks:
        cfg.num_tasks = num_tasks
    if limit:
        cfg.limit = limit
    if device:
        cfg.device = device
    if out_dir is not None:
        cfg.out_dir = out_dir
    cfg.legacy_noise = cfg.legacy_noise or legacy_noise
    cfg.random_init = cfg.random_init or random_init
    run(cfg)


@app.command()
def merge(feature_root: Path) -> None:
    """Join the chunk files of an extraction run."""
    from dd3d.extract import merge as merge_chunks

    for d in merge_chunks(feature_root):
        typer.echo(f"merged {d}")


@app.command()
def probe(
    features: Path = typer.Argument(..., help="Folder with features.npy and ids.txt, or 'pixels'."),
    target: str = typer.Option("global", help="global (lamp around the origin) or local (light at object)."),
    n_components: int = typer.Option(256),
    pixel_resolution: int = typer.Option(64, help="Only for features=pixels."),
    splits_dir: Path = typer.Option(Path("data/splits")),
    scenes_dir: Path = typer.Option(Path("data/final/scenes")),
    images_dir: Path = typer.Option(Path("data/final/images")),
    out: Path = typer.Option(Path("reports/probes/run")),
    tb_dir: Path = typer.Option(None, help="TensorBoard log folder (optional)."),
) -> None:
    """Fit a probe on train, select alpha on val, report all splits with 95% intervals."""
    import numpy as np

    from dd3d.data import add_scene_columns, load_feature_dir, load_splits, pixel_features, target_columns
    from dd3d.probe import ProbeConfig, run

    splits = {k: add_scene_columns(v, scenes_dir) for k, v in load_splits(splits_dir).items()}
    if str(features) == "pixels":
        feats = {k: pixel_features(v, images_dir, pixel_resolution) for k, v in splits.items()}
    else:
        feats = {k: load_feature_dir(features, v["image_id"].tolist()) for k, v in splits.items()}
    az_col, el_col = target_columns(target)
    labels = {k: (v[az_col].to_numpy(), v[el_col].to_numpy()) for k, v in splits.items()}
    res = run(feats, labels, ProbeConfig(n_components=n_components))
    out.mkdir(parents=True, exist_ok=True)
    report = {"features": str(features), "target": target, "alpha": res.alpha, "splits": res.splits}
    (out / "results.json").write_text(json.dumps(report, indent=2))
    for name, (az_err, el_err) in res.errors.items():
        np.save(out / f"errors_{name}.npy", np.stack([az_err, el_err]))
    if tb_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(tb_dir))
        for i, row in enumerate(res.val_curve):
            writer.add_scalar("val/az_mae_vs_alpha_index", row["val_az_mae"], i)
            writer.add_scalar("val/el_mae_vs_alpha_index", row["val_el_mae"], i)
        for name, s in res.splits.items():
            writer.add_scalar(f"{name}/az_mae", s["az_mae"], 0)
            writer.add_scalar(f"{name}/el_mae", s["el_mae"], 0)
        writer.close()
    for name, s in res.splits.items():
        typer.echo(
            f"{name:11s} az {s['az_mae']:6.2f} [{s['az_mae_ci'][0]:.2f}, {s['az_mae_ci'][1]:.2f}]"
            f"  el {s['el_mae']:5.2f}"
        )
    typer.echo(f"alpha={res.alpha:g} -> {out}")


@app.command("audit-arrows")
def audit_arrows(
    scenes_dir: Path = typer.Option(Path("data/final/scenes")),
    out: Path = typer.Option(Path("reports/audit/arrow_audit.csv")),
) -> None:
    """Compare the ZZSN arrows and labels with the exact scene geometry."""
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / "audit" / "arrow_audit.py"
    subprocess.run([sys.executable, str(script), "--scenes", str(scenes_dir), "--out", str(out)], check=True)


@app.command()
def viz(
    predictions: Path = typer.Argument(..., help="CSV with image_id, pred_azimuth_deg, pred_elevation_deg."),
    out_dir: Path = typer.Option(Path("reports/viz")),
    images_dir: Path = typer.Option(Path("data/final/images")),
    scenes_dir: Path = typer.Option(Path("data/final/scenes")),
    n: int = typer.Option(12, help="Number of images: n/2 best and n/2 worst azimuth errors."),
) -> None:
    """Draw label and prediction arrows with the exact camera of each image."""
    import pandas as pd

    from dd3d.metrics import circular_abs_error
    from dd3d.viz import draw_prediction, load_scene

    df = pd.read_csv(predictions)
    if "az_error_deg" not in df:
        df["az_error_deg"] = circular_abs_error(df["pred_azimuth_deg"], df["light_azimuth_deg"])
    df = df.sort_values("az_error_deg")
    pick = pd.concat([df.head(n // 2), df.tail(n - n // 2)])
    for row in pick.itertuples():
        scene = load_scene(scenes_dir, row.image_id)
        shape = row.image_id.rsplit("_new_", 1)[0]
        draw_prediction(
            images_dir / shape / f"{row.image_id}.png",
            scene,
            float(row.pred_azimuth_deg),
            float(row.pred_elevation_deg),
            out_dir / f"{row.image_id}_azerr{row.az_error_deg:.1f}.png",
        )
    typer.echo(f"saved {len(pick)} images to {out_dir}")


if __name__ == "__main__":
    app()
