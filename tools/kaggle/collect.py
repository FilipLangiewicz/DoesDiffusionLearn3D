"""Unpack the downloaded tar files of Kaggle jobs and merge the feature chunks.

kjob.py run downloads the output of a job into <job folder>/output. This script
unpacks the tar files of all jobs with a slug prefix into one folder.

Examples:

    python tools/kaggle/collect.py runs/kaggle/render dd3d-render-v2- data/v2
    python tools/kaggle/collect.py runs/kaggle/extract dd3d-extract-v1- data/features/v1 --merge
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs_root", type=Path)
    parser.add_argument("prefix")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--merge", action="store_true", help="Join the chunks of each run folder.")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    tars = sorted(args.jobs_root.glob(f"{args.prefix}*/output/*.tar"))
    if not tars:
        raise SystemExit(f"no tar files for {args.jobs_root}/{args.prefix}*")
    for t in tars:
        print(f"unpack {t} ({t.stat().st_size / 1e9:.2f} GB)", flush=True)
        subprocess.run(["tar", "xf", str(t), "-C", str(args.dest)], check=True)
    if args.merge:
        from dd3d.extract import merge

        for run_dir in sorted(p for p in args.dest.iterdir() if p.is_dir()):
            print(f"merge {run_dir}: {len(merge(run_dir))} folders", flush=True)


if __name__ == "__main__":
    main()
