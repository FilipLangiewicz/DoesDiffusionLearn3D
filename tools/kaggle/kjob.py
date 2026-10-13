"""Push, wait for and download Kaggle jobs (script kernels).

Commands: run JOB_DIR, wait KERNEL_ID, fetch KERNEL_ID OUT_DIR.

JOB_DIR has a kernel-metadata.json file and the code file. The job must write
only a small number of files to its output folder. The Kaggle API gives
HTTP 429 when a download has many files. Pack many files into one tar file.

The script retries the API calls, because the Kaggle API fails often for a
short time. It writes a status line to stdout for each change of the status.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

KAG = "/home/flang/tools/kag"
DONE = ("COMPLETE", "ERROR", "CANCEL")


def kaggle(*args: str, timeout: int = 600) -> tuple[int, str]:
    try:
        proc = subprocess.run([KAG, *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    return proc.returncode, proc.stdout + proc.stderr


def retry(*args: str, tries: int = 8, timeout: int = 600) -> str:
    delay = 30
    for i in range(tries):
        code, out = kaggle(*args, timeout=timeout)
        if code == 0:
            return out
        print(f"[kjob] attempt {i + 1} failed ({code}): {out.strip()[-300:]}", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 600)
    raise SystemExit(f"[kjob] gave up: kaggle {' '.join(args)}")


def status(kernel_id: str) -> str:
    code, out = kaggle("kernels", "status", kernel_id, timeout=120)
    if code != 0:
        return "UNKNOWN"
    for word in (*DONE, "RUNNING", "QUEUED"):
        if word in out.upper():
            return word
    return "UNKNOWN"


def wait(kernel_id: str, max_hours: float) -> str:
    # The first status after a push can still be the one of the old version.
    time.sleep(60)
    last, start = "", time.time()
    while time.time() - start < max_hours * 3600:
        s = status(kernel_id)
        if s != last:
            print(f"[kjob] {time.strftime('%H:%M:%S')} {kernel_id}: {s}", flush=True)
            last = s
        if s in DONE:
            return s
        time.sleep(60)
    return "WAIT_TIMEOUT"


def fetch(kernel_id: str, out_dir: Path, pattern: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", kernel_id, "-p", str(out_dir), "-o", "-q"]
    if pattern:
        args += ["--file-pattern", pattern]
    retry(*args, timeout=3600)
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            print(f"[kjob] {f.relative_to(out_dir)} {f.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("job_dir", type=Path)
    p_run.add_argument("--accelerator", default=None)
    p_run.add_argument("--max-hours", type=float, default=11.5)
    p_run.add_argument("--out", type=Path, default=None)
    p_run.add_argument("--pattern", default=None)
    p_wait = sub.add_parser("wait")
    p_wait.add_argument("kernel_id")
    p_wait.add_argument("--max-hours", type=float, default=11.5)
    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("kernel_id")
    p_fetch.add_argument("out_dir", type=Path)
    p_fetch.add_argument("--pattern", default=None)
    args = parser.parse_args()

    if args.cmd == "fetch":
        fetch(args.kernel_id, args.out_dir, args.pattern)
        return
    if args.cmd == "wait":
        print(wait(args.kernel_id, args.max_hours))
        return

    kernel_id = json.loads((args.job_dir / "kernel-metadata.json").read_text())["id"]
    push = ["kernels", "push", "-p", str(args.job_dir)]
    if args.accelerator:
        push += ["--accelerator", args.accelerator]
    print(retry(*push).strip().splitlines()[-1], flush=True)
    final = wait(kernel_id, args.max_hours)
    print(f"[kjob] final status: {final}", flush=True)
    out = args.out or args.job_dir / "output"
    if final in DONE:
        fetch(kernel_id, out, args.pattern)
        log = out / "results.txt"
        if log.exists():
            print("===== results.txt (tail)")
            print("\n".join(log.read_text().splitlines()[-60:]))
    sys.exit(0 if final == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
