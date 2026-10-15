"""Kaggle job: causal tests on SANA (shift and LEACE erase, see dd3d.intervene).

tools/kaggle/make_jobs.py writes PARAMS. The inputs are the image tar files of
the render jobs and the plan files (in a data set or in the output of an
analysis job). PARAMS["plans"] lists the plans; each plan runs in its own
worker process, and the workers share the GPUs of the session. The job writes
the output images of each plan as one tar file.
"""

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PARAMS = json.loads("""__PARAMS__""")
WORK = "/kaggle/working"
TMP = "/kaggle/temp"
LOG = open(f"{WORK}/results.txt", "a")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def sh(cmd, quiet=False):
    log(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out and not quiet:
        log(out[-3000:])
    if r.returncode != 0:
        log(out[-3000:])
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return out


os.makedirs(TMP, exist_ok=True)
log(f"params: {json.dumps(PARAMS)}")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
# The preinstalled torchao is too old for diffusers 0.41.
sh("pip uninstall -y -q torchao", quiet=True)
sh("pip install -q 'diffusers==0.41.0' transformers accelerate sentencepiece", quiet=True)

GIT = "https://github.com/FilipLangiewicz/DoesDiffusionLearn3D"
repo = f"{TMP}/repo"
sh(f"git clone -q {GIT} {repo} && cd {repo} && git checkout -q {PARAMS['ref']} && git log -1 --format=%h")

img_root = f"{TMP}/img"
os.makedirs(img_root, exist_ok=True)
for t in sorted(glob.glob("/kaggle/input/**/*.tar", recursive=True)):
    sh(f"tar xf '{t}' -C {img_root}", quiet=True)
found = glob.glob(f"{img_root}/**/Sphere", recursive=True)
candidates = sorted({Path(p).parent for p in found if glob.glob(f"{p}/*.png")})
if not candidates:
    raise SystemExit("no image folder with a Sphere subfolder")
images_dir = candidates[0]
plans = PARAMS.get("plans") or [{"name": PARAMS["name"], "plan": PARAMS["plan"], "config": PARAMS["config"]}]
for p in plans:
    found = sorted(glob.glob(f"/kaggle/input/**/{p['plan']}", recursive=True))
    if not found:
        raise SystemExit(f"no plan file {p['plan']} in the inputs")
    p["path"] = found[0]
log(f"images: {images_dir}, plans: {[p['path'] for p in plans]}")

WORKER = r"""
import json, sys, time
from pathlib import Path
cfg = json.load(open(sys.argv[1]))
sys.path.insert(0, cfg["repo"] + "/src")
from dd3d.intervene import InterveneConfig, run
for p in cfg["plans"]:
    t0 = time.time()
    c = InterveneConfig(images_dir=Path(cfg["images_dir"]), out_dir=Path(cfg["out"]) / p["name"],
                        plan=Path(p["path"]), **p["config"])
    run(c)
    print(f"plan {p['name']}: done in {time.time() - t0:.0f} s", flush=True)
"""
gpus = max(1, min(len(sh("nvidia-smi -L", quiet=True).splitlines()), len(plans)))
Path(f"{TMP}/worker.py").write_text(WORKER)
procs = []
t0 = time.time()
for k in range(gpus):
    cfg = {"repo": repo, "images_dir": str(images_dir), "out": f"{TMP}/out", "plans": plans[k::gpus]}
    Path(f"{TMP}/worker_{k}.json").write_text(json.dumps(cfg))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
    logf = open(f"{TMP}/worker_{k}.log", "w")
    procs.append(
        subprocess.Popen(
            [sys.executable, f"{TMP}/worker.py", f"{TMP}/worker_{k}.json"],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    )
codes = [p.wait() for p in procs]
log(f"workers ended with codes {codes} after {time.time() - t0:.0f} s")
for k in range(gpus):
    tail = Path(f"{TMP}/worker_{k}.log").read_text().splitlines()[-12:]
    log(f"worker {k} log tail:\n" + "\n".join(tail))
for p in plans:
    if Path(f"{TMP}/out/{p['name']}").exists():
        sh(f"cd {TMP}/out && tar cf {WORK}/{p['name']}.tar {p['name']}")
sh(f"ls -la {WORK}")
