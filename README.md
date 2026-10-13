# Does Diffusion Learn 3D?

Do text-to-image models build a 3D model of the world inside? We test this with one simple property: the direction of the light.

We render simple objects with a known light direction and give the images to **SANA-1.6B**. A linear probe on the frozen transformer activations predicts the light azimuth and elevation. The probe learns on spheres and cylinders only. Then we test it on cubes.

> **Status (2026-10-13).** An audit found that the first result (June 2026) does not show what it claimed. A probe on raw pixels gives the same error as the probe on SANA. The project now continues as an explainable AI (XAI) study with controls and causal tests. See [the audit](docs/audit/labels_and_arrows.md) and [the design proposal](docs/design_proposal.md).

<p align="center">
  <img src="docs/figures/readme_good.png" width="45%">
  <img src="docs/figures/readme_bad.png" width="45%">
</p>

The arrows show the cast shadow of the lamp. Each arrow goes from the object base to the shadow of the object centre, with the exact camera of the image. Green: label. Red: probe prediction (block 10, t = 200). Left: a correct prediction. Right: one of the worst errors.

## Results

| Method (on cubes, not seen in the train split) | Azimuth MAE | Elevation MAE |
|---|---:|---:|
| Random guess | 89.86° | 14.61° |
| Always the mean of the train set | 88.87° | 11.35° |
| Probe on SANA-1.6B (`block 10`, `t = 200`), first result | 14.01° | 6.94° |
| **Same probe on raw 64×64 pixels (control)** | **14.09°** | **6.25°** |

What the audit found (details in [`docs/audit/labels_and_arrows.md`](docs/audit/labels_and_arrows.md)):

1. **No gain over pixels.** The control probe on pixels is as good as the SANA probe. So the first result does not show that SANA knows more than the image.
2. **Wrong noise level.** The first extraction used `add_noise` of the scheduler before `set_timesteps`. For t = 200 this gives `-24.5·x0 + 25.5·ε` instead of `0.8·x0 + 0.2·ε`. The new code in `src/dd3d/noise.py` uses the flow rule. An upstream report for the library is in [`tools/diffusers_upstream/`](tools/diffusers_upstream/).
3. **The label is the lamp position, not the light at the object.** The lamp turns around the scene centre. At the object, the light direction differs from the label by 35° (mean). The probe follows the label, probably through the bright floor highlight under the lamp.
4. **The old arrows were not exact.** They used a wrong camera and no perspective. About 18% of the arrows were not in the figures at all. The arrows above use the exact geometry.

## Method

1. **Dataset.** A modified [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen) script renders one purple rubber object in each scene with Blender 2.78c. The key lamp has a random azimuth (0°–360°) and elevation (30°–75°) on a sphere around the scene centre. There are 1000 images for each shape.
2. **Activations.** We add noise to the DC-AE latent of each image and record the output of transformer blocks. The new code stores pooled float16 features (8×8 grid).
3. **Probe.** PCA, standard scores and a Ridge regression predict `[sin φ, cos φ, θ]`, so the cyclic azimuth has no jump at 0°/360°. The Ridge alpha comes from the validation split only. All results have 95% bootstrap intervals.

## Quick start

```bash
make venv install      # Python 3.10+ virtual environment in .venv, CPU PyTorch
make test              # unit tests (CPU, no model weights)
make lint              # ruff
make audit             # label and arrow audit of all 3000 scenes
make cues              # pixel probes: full image, floor only, object only, floor highlight
.venv/bin/dd3d --help  # all commands: extract, merge, probe, viz, audit-arrows
```

The usage manual is in [`docs/usage.md`](docs/usage.md). The cluster jobs are in [`scripts/slurm/`](scripts/slurm/README.md).

## Repository structure

```
src/dd3d/            Python package: geometry, noise, extraction, probes, cues, plots, CLI
configs/             YAML configurations of the extraction runs
scripts/             audit, experiment, CPU and SLURM scripts
tools/blender/       Blender 2.78c script that renders scenes from a JSON spec
tools/diffusers_upstream/  bug report, patch and tests for the diffusers library
tests/unit/          unit tests (pytest)
docs/                design proposal, audit, literature table, worklog, article
clevr-dataset-gen/   CLEVR generator with control of the light direction
dataset_generation/  PowerShell scripts that rendered the dataset
data/                dataset sample, labels and train/val/test splits
probe_exp/           results of the first probe (June 2026)
legacy/              code and SLURM jobs of the first probe (June 2026)
reports/             new results: audit tables, experiments, figures
```

The full dataset and the activations are not in this repository. The first code ran on the Athena supercomputer (PLGrid), and the paths in `legacy/` point there.

## Authors

Natalia Choszczyk and Filip Langiewicz
