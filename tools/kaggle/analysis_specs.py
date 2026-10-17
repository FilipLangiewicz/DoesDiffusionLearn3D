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


def causal_sigma_spec(sigmas: list[int]) -> dict:
    """The causal test of block 4 with more noise (sigma0 in hundredths)."""
    spec = sources("v2")
    spec["features"] = {}
    spec["kernels"] = [RENDERS, f"{O}/dd3d-intervene-causal-sigma", f"{O}/dd3d-analyze-plans-v2"]
    cmds = [unpack("dd3d-intervene-causal-sigma", "/kaggle/temp/causal")]
    plan = "{PREV}/dd3d-analyze-plans-v2/out/plans/b04t400/plan.npz"
    for sg in sigmas:
        cmds.append(
            "python scripts/experiments/analyze_interventions.py"
            f" --outputs /kaggle/temp/causal/causal_b04_s{sg:03d} --plan {plan}"
            " --splits {DATA}/v2/splits --images {DATA}/v2/images"
            f" --out {{OUT}}/causal_sigma/b04_s{sg:03d}"
        )
    return {"name": "causal-sigma-v2", **spec, "commands": cmds}


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


SWEEP_BLOCKS = (4, 10)


def sweep_plans_spec() -> dict:
    spec = sources("v2")
    cmds = [
        f"python scripts/experiments/make_sweep_plan.py --features {{FEAT}}/v2 --key sana/block_{b:02d}/t_400"
        f" --splits {{DATA}}/v2/splits --per-shape 3,3,2 --family pattern --scales 0.5,1,1.5"
        f" --out {{OUT}}/sweep/sweep_b{b:02d}"
        for b in SWEEP_BLOCKS
    ]
    return {"name": "sweep-plans", **spec, "commands": cmds}


def viewer_spec() -> dict:
    spec = sources("v2")
    spec["sets"] = spec["sets"] + ["v2rot"]
    spec["features"] = {}
    spec["kernels"] = [
        RENDERS,
        f"{O}/dd3d-render-v2rot-0",
        f"{O}/dd3d-intervene-sweep",
        f"{O}/dd3d-intervene-patch",
        f"{O}/dd3d-analyze-sweep-plans",
    ]
    spec["datasets"] = [f"{O}/dd3d-plans"]
    data = "--splits {DATA}/v2/splits --images {DATA}/v2/images"
    cmds = [
        unpack("dd3d-intervene-sweep", "/kaggle/temp/sweep"),
        unpack("dd3d-intervene-patch", "/kaggle/temp/patch"),
    ]
    for b in SWEEP_BLOCKS:
        cmds.append(
            f"python scripts/experiments/sweep_package.py --outputs /kaggle/temp/sweep/sweep_b{b:02d}"
            f" --plan {{PREV}}/dd3d-analyze-sweep-plans/out/sweep/sweep_b{b:02d}/plan.npz {data}"
            f" --note 'SANA-1.6B, blok {b}, wzorzec z cech t = 400' --out {{OUT}}/viewer/sweep_b{b:02d}.json"
        )
    cmds.append(
        "python scripts/experiments/sweep_package.py --outputs /kaggle/temp/patch/patch_part0"
        f" --plan {{PREV}}/dd3d-plans/plan_patch_part0.dat {data} --source-images {{DATA}}/v2rot/images"
        " --max-images 8 --note 'SANA-1.6B, przeszczep tokenów z renderu ze światłem +90 stopni'"
        " --out {OUT}/viewer/patch.json"
    )
    return {"name": "viewer-data", **spec, "commands": cmds}


# Features of two vision encoders that do not generate images (dd3d.extract_vision).
VISION_SETS = ("v1", "v2", "v2r", "v2rns", "v2rso")
VISION_JOBS = {s: [f"dd3d-extract-vis-{s}-"] for s in VISION_SETS}
VISION_JOBS["v2"] = VISION_JOBS["v2"] + ["dd3d-extract-vis-v2pos-"]


def vision_spec(s: str) -> dict:
    """Add the DINOv2 and CLIP runs to the scan of a set, then select the probes again.

    The job starts from the scan.csv of stage 1, so only the new runs are fitted.
    """
    spec = sources(s)
    spec["kernels"] = spec["kernels"] + [f"{O}/{p}0" for p in VISION_JOBS[s]] + [f"{O}/dd3d-analyze-scan-{s}"]
    spec["features"] = {s: spec["features"][s] + VISION_JOBS[s]}
    extra = " --scenes {DATA}/v1/scenes --targets global,local" if s == "v1" else ""
    scan_dir = f"{{OUT}}/{s}/scan"
    cmds = [
        f"mkdir -p {scan_dir} && cp {scan_csv(s)} {scan_dir}/scan.csv",
        f"python scripts/experiments/feature_scan.py --features {{FEAT}}/{s} --runs dinov2,clip"
        f" {data_args(s)} --out {scan_dir}{extra}",
        f"python scripts/experiments/plot_scan.py --scan {scan_dir}/scan.csv --title {s}"
        f" --out {{OUT}}/{s}/scan_summary",
        f"python scripts/experiments/best_probes.py --scan {scan_dir}/scan.csv --features {{FEAT}}/{s}"
        f" {data_args(s)} --out {{OUT}}/{s}/best",
    ]
    return {"name": f"vis-{s}", **spec, "commands": cmds}


def vision_transfer_spec() -> dict:
    """The position transfer (fit on v2, test on v2r) for the vision encoders."""
    spec = sources("v2", "v2r")
    spec["kernels"] = (
        spec["kernels"]
        + [f"{O}/{p}0" for s in ("v2", "v2r") for p in VISION_JOBS[s]]
        + [f"{O}/dd3d-analyze-vis-v2"]
    )
    spec["features"] = {s: spec["features"][s] + VISION_JOBS[s] for s in ("v2", "v2r")}
    cmd = (
        "python scripts/experiments/transfer.py --fit v2 --tests v2r --data-root {DATA} --feature-root {FEAT}"
        " --scan {PREV}/dd3d-analyze-vis-v2/out/v2/scan/scan.csv --runs dinov2,clip --pooled-grids 1,2"
        " --out {OUT}/transfer/pos_vision"
    )
    return {"name": "transfer-pos-vision", **spec, "commands": [cmd]}


def reanalyze_spec() -> dict:
    """Analyse the images of all earlier GPU tests again with the measures of the audit fix.

    The new columns: the rotation that a probe on the object crop reads, and the
    shadow shares outside the patched cells. The job also calibrates the measures.
    """
    spec = sources("v2")
    spec["sets"] = spec["sets"] + ["v2rot"]
    spec["features"] = {}
    gpu_jobs = ["causal", "causal-b05", "multi", "causal-sigma", "patch"]
    spec["kernels"] = (
        [RENDERS, f"{O}/dd3d-render-v2rot-0"]
        + [f"{O}/dd3d-intervene-{j}" for j in gpu_jobs]
        + [f"{O}/dd3d-analyze-plans-v2", f"{O}/dd3d-analyze-plans-v2-b05", f"{O}/dd3d-analyze-plans-multi"]
    )
    spec["datasets"] = [f"{O}/dd3d-plans"]
    data = "--splits {DATA}/v2/splits --images {DATA}/v2/images"
    cmds = [unpack(f"dd3d-intervene-{j}", "/kaggle/temp/gpu") for j in gpu_jobs]
    runs = [
        ("causal_b04", "dd3d-analyze-plans-v2/out/plans/b04t400", "causal/b04"),
        ("causal_b10", "dd3d-analyze-plans-v2/out/plans/b10t400", "causal/b10"),
        ("causal_b18", "dd3d-analyze-plans-v2/out/plans/b18t400", "causal/b18"),
        ("causal_b05t100", "dd3d-analyze-plans-v2-b05/out/plans/b05t100", "causal_b05/b05t100"),
        ("causal_multi", "dd3d-analyze-plans-multi/out/plans/multi_all", "causal_multi/all"),
        ("causal_b04_s085", "dd3d-analyze-plans-v2/out/plans/b04t400", "causal_sigma/b04_s085"),
        ("causal_b04_s095", "dd3d-analyze-plans-v2/out/plans/b04t400", "causal_sigma/b04_s095"),
    ]
    for folder, plan, out in runs:
        cmds.append(
            f"python scripts/experiments/analyze_interventions.py --outputs /kaggle/temp/gpu/{folder}"
            f" --plan {{PREV}}/{plan}/plan.npz {data} --out {{OUT}}/{out}"
        )
    parts = (0, 1)
    outs = " ".join(f"/kaggle/temp/gpu/patch_part{p}" for p in parts)
    plans = " ".join(f"{{PREV}}/dd3d-plans/plan_patch_part{p}.dat" for p in parts)
    cmds.append(
        f"python scripts/experiments/analyze_patching.py --outputs {outs} --plans {plans} {data}"
        " --out {OUT}/patch"
    )
    cmds.append(
        f"python scripts/experiments/calibrate_measures.py {data} --rot-images {{DATA}}/v2rot/images"
        " --pairs configs/kaggle/patch_pairs.txt --out {OUT}/measures"
    )
    return {"name": "reanalyze-v2", **spec, "commands": cmds}


def erase_spec() -> dict:
    """Analyse the erasure test with one eraser for each grid cell."""
    spec = sources("v2")
    spec["features"] = {}
    spec["kernels"] = [RENDERS, f"{O}/dd3d-intervene-erase", f"{O}/dd3d-analyze-plans-erase-v2"]
    data = "--splits {DATA}/v2/splits --images {DATA}/v2/images"
    cmds = [unpack("dd3d-intervene-erase", "/kaggle/temp/erase")]
    for tag in ("b04t400", "b10t400", "b18t400", "b05t100"):
        cmds.append(
            f"python scripts/experiments/analyze_interventions.py --outputs /kaggle/temp/erase/erase_{tag}"
            f" --plan {{PREV}}/dd3d-analyze-plans-erase-v2/out/plans/{tag}e/plan.npz {data}"
            f" --out {{OUT}}/causal_erase/{tag}"
        )
    return {"name": "erase-v2", **spec, "commands": cmds}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {}
    for s in VISION_SETS:
        specs[f"vis_{s}"] = vision_spec(s)
    specs["transfer_pos_vision"] = vision_transfer_spec()
    specs["reanalyze_v2"] = reanalyze_spec()
    specs["erase_v2"] = erase_spec()
    for s in ("v1", "v2", "v2r", "v2rns", "v2rso"):
        specs[f"scan_{s}"] = scan_spec(s)
        specs[f"deep_{s}"] = deep_spec(s)
    specs["transfer_pos"] = transfer_spec("pos", "v2", ["v2r"], "sana,sd15")
    specs["transfer_cue"] = transfer_spec("cue", "v2r", ["v2rns", "v2rso"], "sana")
    specs["plans_v2"] = plans_spec(["sana/block_10/t_400", "sana/block_04/t_400", "sana/block_18/t_400"])
    specs["causal_v2"] = causal_spec([10, 4, 18])
    specs["patch_v2"] = patch_spec()
    specs["causal_sigma_v2"] = causal_sigma_spec([85, 95])
    specs["sweep_plans"] = sweep_plans_spec()
    specs["viewer_data"] = viewer_spec()
    for name, spec in specs.items():
        (OUT / f"{name}.json").write_text(json.dumps(spec, indent=1) + "\n")
        print(OUT / f"{name}.json")


if __name__ == "__main__":
    main()
