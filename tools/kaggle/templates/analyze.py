"""Kaggle job: run analysis scripts of the repository on stored features (CPU).

tools/kaggle/make_jobs.py writes PARAMS. The job does these steps:

1. It clones the repository at the commit PARAMS["ref"].
2. It makes the image folder, the labels and the splits of each render set.
3. It links the feature sources of each set into one folder for each set.
   A source is a data set or the output of an extraction job.
4. dd3d.store reads the chunks in place, so the job unpacks nothing.
5. It runs the commands of PARAMS["commands"]. The output folder gets only
   small result files.

The commands can use these names:

- {REPO}: the repository; {OUT}: the result folder;
- {DATA}: the render sets; {FEAT}: the feature folders;
- {PREV}: the outputs of earlier analysis and intervention jobs, one folder each.
"""

import glob
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

PARAMS = json.loads("""__PARAMS__""")
WORK = "/kaggle/working"
TMP = "/kaggle/temp"
LOG = open(f"{WORK}/results.txt", "a")
SHAPES = ("Sphere", "SmoothCylinder", "SmoothCube_v2")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def sh(cmd, quiet=False, check=True, cwd=None, env=None):
    log(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, env=env)
    out = (r.stdout + r.stderr).strip()
    if out and not quiet:
        log(out[-6000:])
    if r.returncode != 0:
        log(out[-3000:])
        if check:
            raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode, out


def inputs(prefix):
    """Return the input folders whose name starts with prefix (data sets and job outputs)."""
    found = set()
    for depth in ("*", "*/*", "*/*/*"):
        for p in glob.glob(f"/kaggle/input/{depth}"):
            if os.path.basename(p).startswith(prefix) and os.path.isdir(p):
                found.add(p)
    return sorted(found)


def split_digest(folder):
    """Return the md5 of the split names and their image ids (same code as make_jobs.py)."""
    h = hashlib.md5()
    for p in sorted(Path(folder).glob("*.csv")):
        ids = [line.split(",")[0] for line in p.read_text().splitlines()[1:]]
        h.update((p.name + ":" + ",".join(ids) + ";").encode())
    return h.hexdigest()


os.makedirs(TMP, exist_ok=True)
log(f"params: {json.dumps(PARAMS)}")
sh("nproc; free -g | head -2; df -h /kaggle/temp /kaggle/working | tail -2")
sh("ls /kaggle/input /kaggle/input/* | head -60")
sh("pip install -q tabulate", quiet=True)

GIT = "https://github.com/FilipLangiewicz/DoesDiffusionLearn3D"
repo = f"{TMP}/repo"
sh(f"git clone -q {GIT} {repo} && cd {repo} && git checkout -q {PARAMS['ref']} && git log -1 --format=%h")
data, feat, out = Path(f"{TMP}/data"), Path(f"{TMP}/feat"), Path(f"{WORK}/out")
out.mkdir(parents=True, exist_ok=True)

# Render sets: v1 comes from its own data set; the new sets come from dd3d-renders-tar.
renders = (inputs("dd3d-renders-tar") or [None])[0]
for name in PARAMS.get("sets", []):
    d = data / name
    d.mkdir(parents=True, exist_ok=True)
    if name == "v1":
        src = [f"{r}/final/images" for r in inputs("dd3d-v1-images") if os.path.isdir(f"{r}/final/images")]
        if not src:
            raise SystemExit("no v1 images in the inputs")
        os.symlink(src[0], d / "images")
        os.symlink(str(Path(src[0]).parent / "scenes"), d / "scenes")
        os.symlink(f"{repo}/data/splits", d / "splits")
        log(f"v1: images {src[0]}")
        continue
    if renders is None:
        raise SystemExit("no input dd3d-renders-tar")
    shards = sorted(glob.glob(f"{renders}/{name}/{name}_*/images"))
    labels = []
    for shape in SHAPES:
        (d / "images" / shape).mkdir(parents=True, exist_ok=True)
    for s in shards:
        for shape in SHAPES:
            for png in glob.glob(f"{s}/{shape}/*.png"):
                os.symlink(png, d / "images" / shape / os.path.basename(png))
        lines = Path(f"{s}/labels.csv").read_text().splitlines()
        labels = labels or [lines[0]]
        labels += lines[1:]
    head, rows = labels[0], list(dict.fromkeys(labels[1:]))
    (d / "labels.csv").write_text("\n".join([head, *rows]) + "\n")
    counts = {shape: len(os.listdir(d / "images" / shape)) for shape in SHAPES}
    log(f"{name}: {len(shards)} shards, {len(rows)} labels, images {counts}")

for name in PARAMS.get("sets", []):
    if name == "v1":
        continue
    d = data / name
    pos = data / "v2pos" / "labels.csv" if name == "v2" else "none"
    sh(
        f"python scripts/data/make_v2_splits.py --labels {d}/labels.csv --pos-labels {pos} --out {d}/splits",
        cwd=repo,
    )
    digest = split_digest(d / "splits")
    want = PARAMS.get("split_md5", {}).get(name)
    log(f"{name}: splits md5 {digest} (expected {want})")
    if want and digest != want:
        raise SystemExit(f"{name}: the splits differ from the local splits")

# Feature sources: one folder for each set, with a link to each source.
for name, prefixes in PARAMS.get("features", {}).items():
    (feat / name).mkdir(parents=True, exist_ok=True)
    for prefix in prefixes:
        for src in inputs(prefix):
            os.symlink(src, feat / name / os.path.basename(src))
            log(f"features {name}: {src}")

# Outputs of earlier jobs (for example a scan.csv or the images of a causal test),
# one link for each job.
prev = Path(f"{TMP}/prev")
prev.mkdir(parents=True, exist_ok=True)
for src in inputs("dd3d-analyze-") + inputs("dd3d-intervene-") + inputs("dd3d-plans"):
    os.symlink(src, prev / os.path.basename(src))
    log(f"earlier output: {src}")

env = dict(os.environ, PYTHONPATH=f"{repo}/src", MPLBACKEND="Agg")
names = {"REPO": repo, "DATA": str(data), "FEAT": str(feat), "OUT": str(out), "PREV": str(prev)}
failed = []
for i, cmd in enumerate(PARAMS["commands"]):
    cmd = cmd.format(**names)
    t0 = time.time()
    code, _ = sh(cmd, check=False, cwd=repo, env=env)
    log(f"command {i} ended with code {code} after {time.time() - t0:.0f} s")
    if code != 0:
        failed.append(i)
log(f"FAILED commands: {failed}" if failed else "all commands ended with code 0")
sh(f"cd {out} && find . -type f | sort | head -300 && du -sh .")
