"""Kaggle job: render one shard of a data set with Blender 2.78c.

tools/kaggle/make_jobs.py writes PARAMS. The job packs the images into one tar
file, because a download of many files fails with HTTP 429.
"""

import json
import os
import subprocess
import time

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
sh("nproc; nvidia-smi --query-gpu=name --format=csv,noheader || true")
sh(
    "apt-get -qq update && apt-get -qq install -y libglu1-mesa libxi6 libxrender1 libxxf86vm1 libxfixes3 libgl1 bzip2",
    quiet=True,
)
url = "https://download.blender.org/release/Blender2.78/blender-2.78c-linux-glibc219-x86_64.tar.bz2"
sh(f"cd {TMP} && wget -q {url} && tar xjf blender-2.78c-linux-glibc219-x86_64.tar.bz2")
blender = f"{TMP}/blender-2.78c-linux-glibc219-x86_64/blender"
sh(f"{blender} -b -noaudio --version | head -1")
repo = f"{TMP}/repo"
sh(
    f"git clone -q https://github.com/FilipLangiewicz/DoesDiffusionLearn3D {repo} && cd {repo} && git checkout -q {PARAMS['ref']} && git log -1 --format=%h"
)

out_dir = f"{TMP}/{PARAMS['name']}"
sh(f"cd {repo} && python scripts/data/make_v2_spec.py --out-dir {out_dir} {PARAMS['spec_args']}")
jobs = json.load(open(f"{out_dir}/render_spec.json"))
shard = jobs[PARAMS["start"] : PARAMS["end"]]
json.dump(shard, open(f"{out_dir}/shard.json", "w"))
log(f"{len(shard)} of {len(jobs)} jobs in this shard")

env = f"DD3D_DEVICE={PARAMS.get('device', 'CPU')}"
t0 = time.time()
proc = subprocess.Popen(
    f"{env} {blender} -b -noaudio --python {repo}/tools/blender/render_spec.py -- {out_dir}/shard.json {repo}",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
done = 0
tail = []
for line in proc.stdout:
    tail = (tail + [line.rstrip()])[-40:]
    if line.startswith("RENDERED"):
        done += 1
        if done in (1, 2, 5, 10) or done % 50 == 0:
            log(f"{done}/{len(shard)} rendered, {(time.time() - t0) / done:.1f} s per image")
proc.wait()
log("\n".join(tail[-15:]))
log(f"blender exit {proc.returncode}: {done}/{len(shard)} images in {time.time() - t0:.0f} s")

sh(f"cp {out_dir}/labels.csv {out_dir}/images/")
name = f"{PARAMS['name']}_{PARAMS['start']:05d}_{PARAMS['end']:05d}"
sh(f"cd {out_dir} && tar cf {WORK}/{name}.tar images && ls -la {WORK}")
if done != len(shard):
    raise SystemExit("not all images rendered")
