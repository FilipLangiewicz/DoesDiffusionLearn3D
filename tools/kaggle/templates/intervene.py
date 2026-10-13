"""Kaggle job: causal tests on SANA (shift and LEACE erase, see dd3d.intervene).

tools/kaggle/make_jobs.py writes PARAMS. The inputs are the image tar files of
the render jobs and a data set with the plan file (plan.npz). The job writes
the output images as one tar file.
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
plans = sorted(glob.glob(f"/kaggle/input/**/{PARAMS['plan']}", recursive=True))
if not plans:
    raise SystemExit(f"no plan file {PARAMS['plan']} in the inputs")
log(f"images: {images_dir}, plan: {plans[0]}")

sys.path.insert(0, f"{repo}/src")
from dd3d.intervene import InterveneConfig, run  # noqa: E402

out_dir = Path(f"{TMP}/out/{PARAMS['name']}")
cfg = InterveneConfig(images_dir=images_dir, out_dir=out_dir, plan=Path(plans[0]), **PARAMS["config"])
t0 = time.time()
run(cfg)
log(f"done in {time.time() - t0:.0f} s")
sh(f"cd {TMP}/out && tar cf {WORK}/{PARAMS['name']}.tar . && ls -la {WORK}")
