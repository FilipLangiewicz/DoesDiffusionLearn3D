"""Write Kaggle job folders from the templates in tools/kaggle/templates.

Each job folder gets the code file run.py and kernel-metadata.json. The kernel
ID is jankombajn/<slug>. A push of the same slug makes a new version.

Example: python tools/kaggle/make_jobs.py render --name v2 --shards 6 --ref main
Analysis: python tools/kaggle/make_jobs.py analyze --spec configs/kaggle/scan_v2r.json --ref <commit>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OWNER = "jankombajn"


def check_ref_on_github(ref: str) -> None:
    """Stop if no branch on origin contains ``ref``: the job clones the code from GitHub."""
    subprocess.run(["git", "fetch", "-q", "origin"], check=True)
    out = subprocess.run(
        ["git", "branch", "-r", "--contains", ref], capture_output=True, text=True, check=False
    ).stdout
    if not out.strip():
        raise SystemExit(f"commit {ref} is not on origin: push it before you write the jobs")


def split_digest(folder: Path) -> str | None:
    """Return the md5 of the split names and their image ids (same code as analyze.py)."""
    if not folder.exists():
        return None
    h = hashlib.md5()
    for p in sorted(folder.glob("*.csv")):
        ids = [line.split(",")[0] for line in p.read_text().splitlines()[1:]]
        h.update((p.name + ":" + ",".join(ids) + ";").encode())
    return h.hexdigest()


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


# Settings of the SANA runs on a T4: no bfloat16 on the T4, text encoder on the CPU.
# A test on 12 images gave the same features in float16 and float32 (relative
# difference 0.1%), and float16 is 4 times faster. The ZZSN noise gives inputs
# that are 35 times too large, which overflow in float16, so that run uses float32.
T4 = {"device": "cuda", "dtype": "float16", "text_encoder_device": "cpu", "transformer_variant": "bf16"}
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
        "dtype": "float32",
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
    # Two vision encoders that do not generate images (see dd3d.extract_vision).
    "dinov2": {"kind": "vision", "model": "dinov2"},
    "clip": {"kind": "vision", "model": "clip"},
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
    e.add_argument("--gpus", type=int, default=1, help="GPUs of the Kaggle session; each runs one half.")
    e.add_argument("--out-root", type=Path, default=Path("runs/kaggle"))
    c = sub.add_parser("intervene")
    c.add_argument("--name", required=True)
    c.add_argument("--plan", default=None, help="File name of the plan in the plan data set.")
    c.add_argument("--spec", type=Path, default=None, help="JSON file with plans, kernels and data sets.")
    c.add_argument("--plan-dataset", default=f"{OWNER}/dd3d-plans")
    c.add_argument("--kernel", action="append", default=[], help="Kaggle job with image tar files.")
    c.add_argument("--config", default="{}", help="JSON with InterveneConfig values (block, sigma0, steps).")
    c.add_argument("--ref", default="main")
    c.add_argument("--out-root", type=Path, default=Path("runs/kaggle"))
    a = sub.add_parser("analyze")
    a.add_argument("--spec", type=Path, required=True, help="JSON file in configs/kaggle (see analyze.py).")
    a.add_argument("--ref", default="main")
    a.add_argument("--out-root", type=Path, default=Path("runs/kaggle/analyze"))
    args = parser.parse_args()
    check_ref_on_github(args.ref)

    if args.cmd == "analyze":
        spec = json.loads(args.spec.read_text())
        # The job checks that its splits have the same image ids as the local splits.
        md5 = {s: split_digest(Path("data") / s / "splits") for s in spec.get("sets", []) if s != "v1"}
        params = {
            "name": spec["name"],
            "ref": args.ref,
            "sets": spec.get("sets", []),
            "features": spec.get("features", {}),
            "split_md5": {k: v for k, v in md5.items() if v},
            "commands": spec["commands"],
        }
        job = write_job(
            args.out_root,
            f"dd3d-analyze-{spec['name']}",
            "analyze.py",
            params,
            gpu=False,
            kernel_sources=spec.get("kernels", []),
            dataset_sources=spec.get("datasets", []),
        )
        print(job)
        return

    if args.cmd == "intervene":
        if args.spec is not None:
            # Several plans in one job; the plan files can come from an analysis job.
            spec = json.loads(args.spec.read_text())
            params = {"name": args.name, "ref": args.ref, "plans": spec["plans"]}
            kernels, datasets = args.kernel + spec.get("kernels", []), spec.get("datasets", [])
        else:
            params = {
                "name": args.name,
                "ref": args.ref,
                "plan": args.plan,
                "config": json.loads(args.config),
            }
            kernels, datasets = args.kernel, [args.plan_dataset]
        job = write_job(
            args.out_root,
            f"dd3d-intervene-{args.name}",
            "intervene.py",
            params,
            gpu=True,
            kernel_sources=kernels,
            dataset_sources=datasets,
        )
        print(job)
        return

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
                "gpus": args.gpus,
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
