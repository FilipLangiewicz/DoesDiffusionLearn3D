# Does Diffusion Learn 3D?

Do text-to-image models build a 3D model of the world inside? We test this with one simple property: the direction of the light.

We render simple objects with a known light direction and give the images to **SANA-1.6B**. A linear probe on the frozen transformer activations predicts the light azimuth and elevation. The probe learns on spheres and cylinders only. Then we test it on cubes.

> **Status (2026-10-16).** An audit found that the first result (June 2026) does not show what it claimed. With a corrected pipeline, new data and controls, the answer is now clearer: SANA-1.6B carries the light direction far beyond the pixels, and it uses this code to light the object. The cast shadow follows only large changes. See [the status report (Polish)](docs/STAN_PRAC.md), [the article draft](docs/article/main.tex) and [the audit](docs/audit/labels_and_arrows.md).

## Results

Azimuth MAE on cubes, a shape that the probe did not see. The val split selects the block, the timestep and the grid of each source.

| Data | Best pixel grid | DC-AE latent | SANA, random weights | **SANA-1.6B** | SD 1.5 | SDXL |
|---|---:|---:|---:|---:|---:|---:|
| v1 (original data, flow noise) | 12.8° | 8.9° | 11.9° | **4.8°** | 6.9° | 6.0° |
| v1 with the original noise rule | | | | 14.5° | | |
| v2r (one sun lamp, random object position) | 20.1° | 19.0° | 15.9° | **7.6°** | 11.1° | 7.3° |
| v2r without the cast shadow | 23.3° | 30.7° | | **9.9°** | | |
| v2r with only the cast shadow | 16.9° | 17.8° | | **7.9°** | | |

The full table with 95% intervals is in [`reports/experiments/rq1.md`](reports/experiments/rq1.md). Other results:

1. **The first result was right for the wrong reasons.** The floor highlight made the pixels as good as the model, and a noise pitfall (`add_noise` before `set_timesteps`) hid the information in the model. With the flow noise, the same probe on the same data gives 4.8° instead of 14.5°.
2. **The code does not depend on the position.** A probe on the mean of all tokens, fitted on centred objects, gives 9.0° (17.6° on cubes) on objects at random positions. The pixel probe gives 71°.
3. **The model uses the code to light the object.** A shift along the activation pattern of the light (2% of the token norm) turns the shading by 60° to 65° towards a target of 90°. A random shift of the same size does nothing.
4. **The shift does not draw a new cast shadow.** With a small shift the crisp shadow stays in place. A large shift removes part of it and darkens the floor towards the target. A patch of the object tokens from a scene with another light moves the shading and about a third of the cast shadow.

The images of each data set version are in [`data/examples/`](data/examples/README.md). An interactive viewer of the interventions is in [`tools/viewer/`](tools/viewer/).

The old figures below show the arrows of the cast shadow with the exact camera. Green: label. Red: probe prediction of the first study (block 10, t = 200).

<p align="center">
  <img src="docs/figures/readme_good.png" width="45%">
  <img src="docs/figures/readme_bad.png" width="45%">
</p>

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
make cues              # pixel probes on image regions (needs the v1 images of the Kaggle data set dd3d-v1-images)
.venv/bin/dd3d --help  # all commands: extract, merge, probe, viz, audit-arrows
```

The usage manual is in [`docs/usage.md`](docs/usage.md). All GPU and analysis jobs run on Kaggle: see [`docs/KAGGLE_RUNBOOK.md`](docs/KAGGLE_RUNBOOK.md) and the list of the data in [`docs/KAGGLE_DATA.md`](docs/KAGGLE_DATA.md).

## Repository structure

```
src/dd3d/            Python package: geometry, noise, extraction, probes, cues, plots, CLI
configs/             YAML configurations of the extraction runs; configs/kaggle: job specs
scripts/             audit, experiment, CPU and SLURM scripts
tools/blender/       Blender 2.78c script that renders scenes from a JSON spec
tools/kaggle/        Kaggle job templates, the job queue and the analysis job specs
tools/viewer/        interactive viewer of the interventions (one HTML file)
tools/diffusers_upstream/  bug report, patch and tests for the diffusers library
tests/unit/          unit tests (pytest)
docs/                design proposal, audit, literature table, worklog, article
clevr-dataset-gen/   CLEVR generator with control of the light direction
dataset_generation/  PowerShell scripts that rendered the dataset
data/                labels and splits of all data sets, examples of each version
probe_exp/           results of the first probe (June 2026)
legacy/              code and SLURM jobs of the first probe (June 2026)
reports/             new results: audit tables, experiments, figures
```

The images and the activations are not in this repository. They are on Kaggle (see `docs/KAGGLE_DATA.md`). The first code ran on the Athena supercomputer (PLGrid), and the paths in `legacy/` point there.

## Authors

Natalia Choszczyk and Filip Langiewicz
