"""Write Kaggle job folders from the templates in tools/kaggle/templates.

Each job folder gets the code file run.py and kernel-metadata.json. The kernel
ID is jankombajn/<slug>. A push of the same slug makes a new version.

Example: python tools/kaggle/make_jobs.py render --name v2 --shards 6 --ref main
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OWNER = "jankombajn"


def write_job(
    out_root: Path,
    slug: str,
    template: str,
    params: dict,
    gpu: bool,
    kernel_sources=(),
    dataset_sources=(),
) -> Path:
    job = out_root / slug
    job.mkdir(parents=True, exist_ok=True)
    code = (ROOT / "templates" / template).read_text().replace("__PARAMS__", json.dumps(params))
    (job / "run.py").write_text(code)
    meta = {
        "id": f"{OWNER}/{slug}",
        "title": slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": list(dataset_sources),
        "competition_sources": [],
        "kernel_sources": list(kernel_sources),
    }
    (job / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return job


# Settings of the SANA runs on a T4: float32 (no bfloat16 on the T4), text encoder on the CPU.
T4 = {"device": "cuda", "dtype": "float32", "text_encoder_device": "cpu", "transformer_variant": "bf16"}
MAIN_BLOCKS = [4, 10, 18]
RUNS = {
    # All blocks with a 4 x 4 grid and the main blocks with an 8 x 8 grid.
    "sana": {
        **T4,
        "blocks": list(range(20)),
        "timesteps": [100, 200, 400, 600, 800],
        "grid": 4,
        "fine_blocks": MAIN_BLOCKS,
        "fine_grid": 8,
    },
    # The ZZSN noise rule, only for the comparison with the old features.
    "sana_legacy_noise": {
        **T4,
        "blocks": MAIN_BLOCKS,
        "timesteps": [200, 500, 800],
        "grid": 8,
        "legacy_noise": True,
        "save_vae_latents": False,
    },
    # The same architecture with random weights (control).
    "sana_random_init": {
        **T4,
        "blocks": MAIN_BLOCKS,
        "timesteps": [100, 200, 400, 600, 800],
        "grid": 8,
        "random_init": True,
        "save_vae_latents": False,
    },
    # Two UNet models for the check of generality (see dd3d.extract_unet).
    "sd15": {"kind": "unet", "model": "sd15"},
    "sdxl": {"kind": "unet", "model": "sdxl"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render")
    p.add_argument("--name", required=True, help="Data set name, also the prefix of the slugs.")
    p.add_argument("--total", type=int, default=3000)
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--spec-args", default="")
    p.add_argument("--ref", default="main")
    p.add_argument("--device", default="CPU")
    p.add_argument("--out-root", type=Path, default=Path("runs/kaggle"))
    e = sub.add_parser("extract")
    e.add_argument("--name", required=True, help="Feature set name, also the prefix of the slugs.")
    e.add_argument("--tasks", type=int, default=6)
    e.add_argument("--only-task", type=int, default=None)
    e.add_argument("--runs", default="sana,sana_legacy_noise,sana_random_init")
    e.add_argument("--limit", type=int, default=None, help="Only the first N images of each task.")
    e.add_argument("--max-per-shape", type=int, default=1000)
    e.add_argument("--dataset", action="append", default=[], help="Kaggle data set with image tar files.")
    e.add_argument("--kernel", action="append", default=[], help="Kaggle job with image tar files.")
    e.add_argument("--ref", default="main")
    e.add_argument("--dtype", default=None, help="Override the dtype of the SANA runs (float16 or float32).")
    e.add_argument("--out-root", type=Path, default=Path("runs/kaggle"))
    args = parser.parse_args()

    if args.cmd == "extract":
        tasks = [args.only_task] if args.only_task is not None else range(args.tasks)
        for task in tasks:
            runs = {}
            for r in args.runs.split(","):
                runs[r] = {**RUNS[r], "max_per_shape": args.max_per_shape}
                if args.limit:
                    runs[r]["limit"] = args.limit
                if args.dtype and runs[r].get("kind", "sana") == "sana":
                    runs[r]["dtype"] = args.dtype
            params = {
                "name": args.name,
                "ref": args.ref,
                "task_id": task,
                "num_tasks": args.tasks,
                "runs": runs,
            }
            slug = f"dd3d-extract-{args.name}-{task}"
            job = write_job(
                args.out_root,
                slug,
                "extract.py",
                params,
                gpu=True,
                kernel_sources=args.kernel,
                dataset_sources=args.dataset,
            )
            print(job)
        return

    if args.cmd == "render":
        size = -(-args.total // args.shards)
        for i in range(args.shards):
            start, end = i * size, min((i + 1) * size, args.total)
            params = {
                "name": args.name,
                "ref": args.ref,
                "spec_args": args.spec_args,
                "start": start,
                "end": end,
                "device": args.device,
            }
            slug = f"dd3d-render-{args.name}-{i}"
            job = write_job(args.out_root, slug, "render.py", params, gpu=args.device == "GPU")
            print(job)


if __name__ == "__main__":
    main()
