"""Kaggle job: extract SANA features for one task of a data set on one GPU.

tools/kaggle/make_jobs.py writes PARAMS. The job finds the image tar files in
the inputs (a data set or the outputs of the render jobs). It writes the
features as one tar file, because a download of many files fails with HTTP 429.
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
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2")
# The preinstalled torchao is too old for diffusers 0.41.
sh("pip uninstall -y -q torchao", quiet=True)
sh("pip install -q 'diffusers==0.41.0' transformers accelerate sentencepiece", quiet=True)
sh("pip list 2>/dev/null | grep -E '^(torch|diffusers|transformers) '")

GIT = "https://github.com/FilipLangiewicz/DoesDiffusionLearn3D"
repo = f"{TMP}/repo"
sh(f"git clone -q {GIT} {repo} && cd {repo} && git checkout -q {PARAMS['ref']} && git log -1 --format=%h")

# Unpack all image tar files of the inputs into one folder.
img_root = f"{TMP}/img"
os.makedirs(img_root, exist_ok=True)
tars = sorted(glob.glob("/kaggle/input/**/*.tar", recursive=True))
log(f"{len(tars)} tar files in the inputs")
for t in tars:
    sh(f"tar xf '{t}' -C {img_root}", quiet=True)
# Kaggle unpacks a tar file of a data set itself, so also search the inputs.
# Only a Sphere folder with PNG files counts; the scenes folder has JSON files.
found = glob.glob(f"{img_root}/**/Sphere", recursive=True)
found += glob.glob("/kaggle/input/**/Sphere", recursive=True)
candidates = sorted({Path(p).parent for p in found if glob.glob(f"{p}/*.png")})
if not candidates:
    raise SystemExit("no image folder with a Sphere subfolder")
images_dir = candidates[0]
counts = {s: len(list((images_dir / s).glob("*.png"))) for s in ("Sphere", "SmoothCylinder", "SmoothCube_v2")}
log(f"images: {images_dir} {counts}")

sys.path.insert(0, f"{repo}/src")
from dd3d.extract import ExtractConfig, run  # noqa: E402
from dd3d.extract_unet import UNetExtractConfig  # noqa: E402
from dd3d.extract_unet import run as run_unet  # noqa: E402

name = PARAMS["name"]
prompt_cache = Path(f"{TMP}/prompt_empty.pt")
for run_name, overrides in PARAMS["runs"].items():
    overrides = dict(overrides)
    common = {
        "images_dir": images_dir,
        "out_dir": Path(f"{TMP}/out/{run_name}"),
        "task_id": PARAMS["task_id"],
        "num_tasks": PARAMS["num_tasks"],
    }
    if overrides.pop("kind", "sana") == "unet":
        cfg, runner = UNetExtractConfig(**common, **overrides), run_unet
    else:
        cfg, runner = ExtractConfig(**common, prompt_cache=prompt_cache, **overrides), run
    t0 = time.time()
    log(f"run {run_name}: {overrides}")
    runner(cfg)
    log(f"run {run_name}: done in {time.time() - t0:.0f} s")
    stats = sorted(glob.glob(f"{cfg.out_dir}/stats/*.csv"))
    if stats:
        import pandas as pd

        df = pd.concat(pd.read_csv(s) for s in stats)
        table = df.groupby(["block", "t"])["mean_token_norm"].mean().unstack("t").round(1)
        log(f"mean token norm ({run_name}):\n{table.to_string()}")

out_tar = f"{WORK}/{name}_task{PARAMS['task_id']:02d}.tar"
sh(f"cd {TMP}/out && tar cf {out_tar} . && ls -la {WORK} && du -sh {TMP}/out")
