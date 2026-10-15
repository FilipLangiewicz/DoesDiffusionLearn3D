"""Copy the small result files of finished analysis jobs into reports/experiments/.

The queue downloads the output of each analysis job to
runs/kaggle/analyze/<slug>/output/ (outside git). The job writes its results
under out/<set>/... or out/transfer/..., and this script copies that tree to
reports/experiments/ (in git). A file larger than --max-mb stays out of git.
The log of the job goes to reports/experiments/logs/<slug>.txt.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=Path("runs/kaggle/analyze"))
    parser.add_argument("--reports", type=Path, default=Path("reports/experiments"))
    parser.add_argument("--max-mb", type=float, default=5.0)
    args = parser.parse_args()

    for job in sorted(args.jobs.glob("dd3d-analyze-*")):
        # The light test jobs check the pipeline only; their results stay out of the reports.
        if not (job / ".done").exists() or "smoke" in job.name:
            continue
        out = job / "output" / "out"
        copied = 0
        for f in sorted(out.rglob("*")) if out.exists() else []:
            if not f.is_file():
                continue
            if f.stat().st_size > args.max_mb * 1e6:
                print(f"{f}: larger than {args.max_mb} MB, not copied")
                continue
            dest = args.reports / f.relative_to(out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            copied += 1
        log = job / "output" / "results.txt"
        if log.exists():
            (args.reports / "logs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(log, args.reports / "logs" / f"{job.name}.txt")
        print(f"{job.name}: {copied} files")


if __name__ == "__main__":
    main()
