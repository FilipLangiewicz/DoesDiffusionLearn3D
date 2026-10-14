"""Run a queue of Kaggle jobs within the session limits of Kaggle.

The queue file has one job on each line, in the order of priority:

    <job folder> <CPU|GPU> [after=<slug prefix>] [acc=<Kaggle accelerator>]

A job with after=<prefix> starts only when all jobs whose slug starts with this
prefix are done. Empty lines and lines that start with # are ignored.

Marker files in the job folder keep the state of each job. Thus the queue
continues after a restart of the computer. The markers are:

- .pushed: the job is on Kaggle.
- .complete: the job ended on Kaggle; its output waits for the download.
- .fetching: a download process runs (the file has its process id).
- .done: the output is downloaded.
- .failed: Kaggle gave ERROR or CANCEL.

The queue never pushes a job again that has a .pushed file. To run a job again,
delete its marker files.

Kaggle runs at most 5 CPU jobs and 2 GPU jobs at the same time.

Usage: python tools/kaggle/kqueue.py runs/kaggle/queue.txt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kjob import fetch, retry, status  # noqa: E402

LIMITS = {"CPU": 5, "GPU": 2}
ACCELERATOR = {"CPU": None, "GPU": "NvidiaTeslaT4"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_queue(path: Path) -> list[dict]:
    jobs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        job = {"dir": Path(parts[0]), "kind": parts[1], "after": None, "acc": ACCELERATOR[parts[1]]}
        for p in parts[2:]:
            if p.startswith("after="):
                job["after"] = p.split("=", 1)[1]
            if p.startswith("acc="):
                job["acc"] = p.split("=", 1)[1]
        job["slug"] = job["dir"].name
        job["id"] = json.loads((job["dir"] / "kernel-metadata.json").read_text())["id"]
        jobs.append(job)
    return jobs


def state(job: dict) -> str:
    d = job["dir"]
    if (d / ".done").exists():
        return "done"
    if (d / ".failed").exists():
        return "failed"
    if (d / ".complete").exists():
        return "complete"
    if (d / ".pushed").exists():
        return "pushed"
    return "new"


def push(job: dict) -> str:
    args = ["kernels", "push", "-p", str(job["dir"])]
    acc = job["acc"]
    if acc:
        args += ["--accelerator", acc]
    out = retry(*args).strip()
    if "error" in out.lower():
        return "refused" if "session count" in out.lower() else f"error: {out[-200:]}"
    (job["dir"] / ".pushed").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return "pushed"


def fetching(job: dict) -> bool:
    """Return True if a download process for this job still runs."""
    f = job["dir"] / ".fetching"
    if not f.exists():
        return False
    try:
        os.kill(int(f.read_text().strip()), 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        f.unlink(missing_ok=True)
        return False


def start_fetch(job: dict) -> None:
    """Start the download in its own process, so a large file never stops the queue."""
    kfetch = Path(__file__).resolve().parent / "kfetch.py"
    out = job["dir"] / "output"
    done = job["dir"] / ".done"
    cmd = f"{sys.executable} -u {kfetch} {job['id']} {out} && date > {done}"
    logf = open(job["dir"] / "fetch.log", "a")
    proc = subprocess.Popen(
        ["bash", "-c", cmd], stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
    )
    (job["dir"] / ".fetching").write_text(str(proc.pid))
    log(f"{job['slug']}: download started (pid {proc.pid})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()

    while True:
        jobs = read_queue(args.queue)
        by_state = {}
        for job in jobs:
            by_state.setdefault(state(job), []).append(job)

        # Check the jobs on Kaggle. A complete job frees its session at once; its
        # download runs in a separate process (one at a time, see start_fetch).
        running = {"CPU": 0, "GPU": 0}
        for job in by_state.get("pushed", []):
            s = status(job["id"])
            if s == "COMPLETE":
                (job["dir"] / ".complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
                log(f"{job['slug']}: complete on Kaggle")
            elif s in ("ERROR", "CANCEL"):
                try:
                    fetch(job["id"], job["dir"] / "output", r"results\.txt|.*\.log")
                except SystemExit:
                    pass
                (job["dir"] / ".failed").write_text(s + "\n")
                log(f"{job['slug']}: {s}")
            else:
                # RUNNING, QUEUED or a short API failure: the job keeps its session.
                running[job["kind"]] += 1

        # Two downloads at a time, so a small file does not wait for a large one.
        jobs = read_queue(args.queue)
        waiting = [j for j in jobs if state(j) == "complete"]
        active = sum(fetching(j) for j in waiting)
        for job in waiting:
            if active >= 2:
                break
            if not fetching(job):
                start_fetch(job)
                active += 1
        for job in jobs:
            if state(job) == "done" and (job["dir"] / ".fetching").exists():
                (job["dir"] / ".fetching").unlink(missing_ok=True)
                log(f"{job['slug']}: done, output in {job['dir'] / 'output'}")

        # Push new jobs while Kaggle has free sessions.
        done_slugs = {j["slug"] for j in jobs if state(j) == "done"}
        refused = set()
        for job in jobs:
            kind = job["kind"]
            if state(job) != "new" or kind in refused or running[kind] >= LIMITS[kind]:
                continue
            if job["after"]:
                deps = [j for j in jobs if j["slug"].startswith(job["after"])]
                if not deps or any(j["slug"] not in done_slugs for j in deps):
                    continue
            result = push(job)
            if result == "pushed":
                running[kind] += 1
                log(f"{job['slug']}: pushed ({kind})")
            elif result == "refused":
                refused.add(kind)
            else:
                log(f"{job['slug']}: push {result}")

        counts = {
            s: sum(state(j) == s for j in jobs) for s in ("new", "pushed", "complete", "done", "failed")
        }
        log(f"state {counts} running {running}")
        if counts["new"] == 0 and counts["pushed"] == 0 and counts["complete"] == 0:
            log("queue finished")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
