# Usage manual

This manual tells you how to install the project, how to run each step and where the results go.

## 1. Requirements

- Linux or WSL2, Python 3.10 or newer.
- For the probes, the audit and the tests: a CPU and 4 GB of free RAM.
- For the activation extraction: a GPU with 16 GB of memory (an A100 on the Athena cluster). A CPU can run a smoke test on a few images.
- For new renders: Blender 2.78c (the CLEVR generator does not work with newer versions).

## 2. Installation

```bash
git clone <repository> && cd DoesDiffusionLearn3D
make venv        # makes .venv
make install     # CPU PyTorch, the package dd3d and the dev tools
make test        # the unit tests must pass
```

On the cluster, use your own environment (for example conda) and install the package with `pip install -e ".[model,track]"`.

## 3. Data

| Path | Content | In git |
|---|---|---|
| `data/final/images/<shape>/` | 3000 renders (512 x 512 PNG) | no (1.1 GB) |
| `data/final/scenes/<shape>/` | one JSON file for each render | no |
| `data/final_example/` | 10 renders and JSON files for each shape | yes |
| `data/master_labels.csv` | labels of all renders | yes |
| `data/splits/` | train, val, test (spheres and cylinders), test_cubes | yes |
| `data/features/` | extracted features (output of `dd3d extract`) | no |

Do not change the files in `data/final`, `data/master_labels.csv` and `data/splits`. All new outputs go to `data/features`, `reports` or `data/audit_renders`.

## 4. Commands

All commands have `--help`.

### 4.1 Audit of the labels and the arrows

```bash
dd3d audit-arrows                      # writes reports/audit/arrow_audit.csv and .txt
python scripts/audit/shadow_audit.py   # needs the renders of section 5
```

### 4.2 Pixel probes (no diffusion model)

```bash
dd3d probe pixels --out reports/probes/pixels64_global
dd3d probe pixels --target local --out reports/probes/pixels64_local
make cues                              # four image variants, writes reports/experiments/pixel_cues/
```

The option `--target global` uses the label of the dataset (lamp position around the scene centre). The option `--target local` uses the light direction at the object.

### 4.3 Activation extraction (GPU)

```bash
dd3d extract --config configs/extract.yaml --task-id 0 --num-tasks 30
dd3d merge data/features/v1_sana
```

The configuration file sets the blocks, the timesteps, the pool grid and the seed. On the cluster, the SLURM jobs in `scripts/slurm/` call these commands (see `scripts/slurm/README.md`). The controls use `configs/extract_random_init.yaml` (random weights) and `configs/extract_legacy_noise.yaml` (the old noise rule).

The output for block `b` and timestep `t` is `data/features/v1_sana/block_<bb>/t_<ttt>/features.npy` with `ids.txt`. Each row has 8 x 8 x 2240 float16 values.

### 4.4 Probe on extracted features

```bash
dd3d probe data/features/v1_sana/block_10/t_200 --out reports/probes/sana_b10_t200 --tb-dir runs/sana_b10_t200
tensorboard --logdir runs
```

The command fits the probe on the train split and selects the Ridge alpha on the val split. It reports all splits with 95% bootstrap intervals in `results.json`.

### 4.5 Figures with exact arrows

```bash
dd3d viz probe_exp/test_cubes_eval_all/layer_10_t_200/test_cubes_predictions.csv --out-dir reports/viz/b10_t200
```

The input CSV needs the columns `image_id`, `light_azimuth_deg`, `pred_azimuth_deg` and `pred_elevation_deg`.

### 4.6 CPU smoke test of SANA-1.6B

```bash
python scripts/cpu/make_prompt_cache.py            # Gemma-2 with disk offload, about 3 min
python scripts/cpu/smoke_noise_comparison.py --images data/final_example/images/Sphere/Sphere_new_000000.png
```

The test needs about 4 GB of free RAM and 17 GB of disk for the model files. It compares the old noise rule with the flow rule.

## 5. Renders with Blender 2.78c

1. Download Blender 2.78c for Linux from `download.blender.org/release/Blender2.78/`.
2. If Blender does not start, get the necessary libraries with `apt-get download libglu1-mesa libopengl0` and `dpkg -x <file>.deb <folder>`.
3. Put the library folder on `LD_LIBRARY_PATH`.
4. Write a spec file with `python scripts/audit/make_render_specs.py --spec-dir specs`.
5. Render it with `blender -b -noaudio --python tools/blender/render_spec.py -- specs/conventions.json "$PWD"`.

One render with 64 samples takes about 25 s on an 8-core CPU.

## 6. Development rules

- Format and lint: `make format` and `make lint` (ruff, line length 110).
- Tests: `make test`. Add a unit test for each new function in `src/dd3d`.
- Commits: conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- Configuration goes to `configs/`, not into the code.
- Record the progress each week in `docs/worklog.md`.
