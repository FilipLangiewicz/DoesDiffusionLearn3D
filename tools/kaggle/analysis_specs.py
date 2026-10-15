"""Write the job specs of the Kaggle analysis jobs to configs/kaggle/.

Stage 1 (scan-<set>): the probe on every block and timestep of each run.
Stage 2 (deep-<set>, transfer-*, plans-v2): jobs that read the scan.csv of
stage 1 (input {PREV}) and select the block and timestep on the val split.

Usage: python tools/kaggle/analysis_specs.py, then
python tools/kaggle/make_jobs.py analyze --spec configs/kaggle/<spec>.json --ref <commit>
"""

from __future__ import annotations

import json
from pathlib import Path

O = "jankombajn"
# The job output with the render tar files of all new sets (faster to read than
# the unpacked data set dd3d-renders-tar).
RENDERS = f"{O}/dd3d-renders"
OUT = Path("configs/kaggle")

# The feature sources of each set: data sets for v1 and v2r, job outputs for the others.
DATASETS = {"v1": [f"{O}/dd3d-features-v1-{i}" for i in range(6)] + [f"{O}/dd3d-v1-images"]}
DATASETS["v2r"] = [f"{O}/dd3d-features-v2r-{i}" for i in range(6)]
KERNELS = {s: [f"{O}/dd3d-extract-{s}-{i}" for i in range(6)] for s in ("v2", "v2rns", "v2rso")}
KERNELS["v2"] = KERNELS["v2"] + [f"{O}/dd3d-extract-v2pos-0"]
PREFIXES = {
    "v1": ["dd3d-features-v1-"],
    "v2": ["dd3d-extract-v2-", "dd3d-extract-v2pos-"],
    "v2r": ["dd3d-features-v2r-"],
    "v2rns": ["dd3d-extract-v2rns-"],
    "v2rso": ["dd3d-extract-v2rso-"],
}
RENDER_SETS = {"v1": ["v1"], "v2": ["v2pos", "v2"]}
RUNS_WITH_MAPS = "sana,sana_random_init,sd15,sdxl"


def sources(*sets: str) -> dict:
    kernels, datasets = [], []
    for s in sets:
        kernels += KERNELS.get(s, [])
        datasets += DATASETS.get(s, [])
    if any(s != "v1" for s in sets):
        kernels.append(RENDERS)
    render = []
    for s in sets:
        render += [r for r in RENDER_SETS.get(s, [s]) if r not in render]
    return {
        "sets": render,
        "features": {s: PREFIXES[s] for s in sets},
        "kernels": kernels,
        "datasets": datasets,
    }


def scan_csv(s: str) -> str:
    return f"{{PREV}}/dd3d-analyze-scan-{s}/out/{s}/scan/scan.csv"


def data_args(s: str) -> str:
    return f"--splits {{DATA}}/{s}/splits --images {{DATA}}/{s}/images"


def scan_spec(s: str) -> dict:
    extra = " --scenes {DATA}/v1/scenes --targets global,local" if s == "v1" else ""
    return {
        "name": f"scan-{s}",
        **sources(s),
        "commands": [
            f"python scripts/experiments/feature_scan.py --features {{FEAT}}/{s} --runs all {data_args(s)}"
            f" --out {{OUT}}/{s}/scan{extra}",
            f"python scripts/experiments/plot_scan.py --scan {{OUT}}/{s}/scan/scan.csv --title {s}"
            f" --out {{OUT}}/{s}/scan_summary",
        ],
    }


def deep_spec(s: str) -> dict:
    spec = sources(s)
    spec["kernels"] = spec["kernels"] + [f"{O}/dd3d-analyze-scan-{s}"]
    labels = f"{{DATA}}/{s}/labels.csv"
    cmds = [
        f"python scripts/experiments/best_probes.py --scan {scan_csv(s)} --features {{FEAT}}/{s}"
        f" {data_args(s)}"
        f" --out {{OUT}}/{s}/best",
    ]
    if s != "v1":
        cmds.append(
            f"python scripts/experiments/probe_maps.py --features {{FEAT}}/{s} --scan {scan_csv(s)}"
            f" --runs {RUNS_WITH_MAPS} {data_args(s)} --labels {labels} --title {s} --out {{OUT}}/{s}/maps"
        )
    return {"name": f"deep-{s}", **spec, "commands": cmds}


def transfer_spec(name: str, fit: str, tests: list[str], runs: str) -> dict:
    spec = sources(fit, *tests)
    spec["kernels"] = spec["kernels"] + [f"{O}/dd3d-analyze-scan-{fit}"]
    cmd = (
        f"python scripts/experiments/transfer.py --fit {fit} --tests {','.join(tests)} --data-root {{DATA}}"
        f" --feature-root {{FEAT}} --scan {scan_csv(fit)} --runs {runs} --pooled-grids 1,2"
        f" --out {{OUT}}/transfer/{name}"
    )
    return {"name": f"transfer-{name}", **spec, "commands": [cmd]}


def plans_spec(keys: list[str]) -> dict:
    spec = sources("v2")
    cmds = []
    for key in keys:
        tag = key.replace("sana/", "").replace("block_", "b").replace("/t_", "t")
        cmds.append(
            f"python scripts/experiments/make_intervention_plan.py --features {{FEAT}}/v2 --key {key}"
            f" --splits {{DATA}}/v2/splits --n-images 60 --out {{OUT}}/plans/{tag}"
        )
    return {"name": "plans-v2", **spec, "commands": cmds}


def unpack(job: str, to: str) -> str:
    """Return a shell command that unpacks the image tar files of a GPU job."""
    return f"mkdir -p {to} && for f in {{PREV}}/{job}/*.tar; do tar xf $f -C {to}; done"


def causal_spec(blocks: list[int]) -> dict:
    spec = sources("v2")
    spec["features"] = {}
    spec["kernels"] = [RENDERS, f"{O}/dd3d-intervene-causal", f"{O}/dd3d-analyze-plans-v2"]
    cmds = [unpack("dd3d-intervene-causal", "/kaggle/temp/causal")]
    for b in blocks:
        plan = f"{{PREV}}/dd3d-analyze-plans-v2/out/plans/b{b:02d}t400/plan.npz"
        cmds.append(
            "python scripts/experiments/analyze_interventions.py"
            f" --outputs /kaggle/temp/causal/causal_b{b:02d} --plan {plan}"
            f" --splits {{DATA}}/v2/splits --images {{DATA}}/v2/images --out {{OUT}}/causal/b{b:02d}"
        )
    return {"name": "causal-v2", **spec, "commands": cmds}


def patch_spec() -> dict:
    spec = sources("v2")
    spec["features"] = {}
    spec["kernels"] = [RENDERS, f"{O}/dd3d-intervene-patch"]
    spec["datasets"] = spec["datasets"] + [f"{O}/dd3d-plans"]
    parts = (0, 1)
    outs = " ".join(f"/kaggle/temp/patch/patch_part{p}" for p in parts)
    plans = " ".join(f"{{PREV}}/dd3d-plans/plan_patch_part{p}.dat" for p in parts)
    cmds = [
        unpack("dd3d-intervene-patch", "/kaggle/temp/patch"),
        f"python scripts/experiments/analyze_patching.py --outputs {outs} --plans {plans}"
        f" --splits {{DATA}}/v2/splits --images {{DATA}}/v2/images --out {{OUT}}/patch",
    ]
    return {"name": "patch-v2", **spec, "commands": cmds}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {}
    for s in ("v1", "v2", "v2r", "v2rns", "v2rso"):
        specs[f"scan_{s}"] = scan_spec(s)
        specs[f"deep_{s}"] = deep_spec(s)
    specs["transfer_pos"] = transfer_spec("pos", "v2", ["v2r"], "sana,sd15")
    specs["transfer_cue"] = transfer_spec("cue", "v2r", ["v2rns", "v2rso"], "sana")
    specs["plans_v2"] = plans_spec(["sana/block_10/t_400", "sana/block_04/t_400", "sana/block_18/t_400"])
    specs["causal_v2"] = causal_spec([10, 4, 18])
    specs["patch_v2"] = patch_spec()
    for name, spec in specs.items():
        (OUT / f"{name}.json").write_text(json.dumps(spec, indent=1) + "\n")
        print(OUT / f"{name}.json")


if __name__ == "__main__":
    main()
