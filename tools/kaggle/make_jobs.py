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
    args = parser.parse_args()

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
