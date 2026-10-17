# Design proposal

> This is the plan of 2026-10-13. It does not change. The results are in `docs/RESULTS.md`, and the deviations from this plan are in `docs/worklog.md`.

Project: Does SANA encode the light direction of a 3D scene, and does it use this information?

Date of this version: 2026-10-13. Branch: `siwy-xai`.

## 1. Summary

This project continues a project of the summer 2026. The old project trained a linear probe on the activations of SANA-1.6B. The probe predicted the light direction of simple Blender scenes. The old report says that SANA encodes the light direction.

In week 1 we audited the old pipeline. The audit shows that the old result is not valid as stated. A probe on raw pixels gives the same error. The noise injection had an error. The label does not describe the light at the object.

The new project asks the question again, with correct data, baselines and causal tests. The goal is an explainable AI (XAI) study of one model and one property. The study can give a negative result, and we still report it with care.

What is new compared to the old project:

- A corrected extraction of the activations (new package `dd3d`).
- Baselines: pixels, DC-AE latent, and a SANA transformer with random weights.
- A new data set (v2) that separates the visual cues of the light.
- Causal tests: concept erasure and activation shift.
- A scientific article, and a public upstream fix for a library error.

We must state this difference to the course coordinator, because the old project belongs to another course. We ask for approval in week 2.

## 2. Background and the state at the start

### 2.1 The old pipeline

- Data: 3000 Blender renders from the CLEVR generator. There are 1000 images for each shape (sphere, cylinder, cube). Each scene has one purple rubber object. The lamp azimuth is U(0, 360) deg. The lamp elevation is U(30, 75) deg.
- Method: we add noise to the DC-AE latent of the image. We record the activations of transformer blocks 4, 10 and 18 at timesteps 200, 500 and 800. A 4x4 average pool, PCA with 256 components and Ridge regression follow. The target is `[sin az, cos az, el]`.
- Split: the train split has spheres and cylinders (1600 images). The validation split has 200 images and the test split has 200 images. The set `test_cubes` has 1000 cubes.
- Old result: on cubes, the azimuth MAE is 14.0 deg and the elevation MAE is 6.9 deg (block 10, t = 200).

### 2.2 Audit findings (2026-10-12 and 2026-10-13)

| Id | Finding |
|---|---|
| F1 | The same probe on raw 64x64 pixels gives 14.09 deg and 6.25 deg on cubes. On the test split it gives 11.45 and 5.60 deg. SANA gives 13.01 and 6.88 deg. The old result does not show more than the pixels. |
| F2 | The old extraction called `add_noise` of `DPMSolverMultistepScheduler` before `set_timesteps`. The result is x_t = -24.5 x0 + 25.5 eps at t = 200. It is -2.4 x0 + 3.4 eps at t = 500 and 0.28 x0 + 0.72 eps at t = 800. The correct form is (1 - t/1000) x0 + (t/1000) eps. The stored activations agree with this error. The same library problem affects the VP schedules of four schedulers. |
| F3 | The label is the lamp position on a sphere of radius 5 around the world origin. The light direction at the object differs by 35 deg in azimuth and 16 deg in elevation (mean). The SANA probe follows the label (14.0 deg), not the light at the object (35.5 deg). |
| F4 | The CLEVR base scene has a fourth lamp "Area" (type SUN, strength 0.45, azimuth 334.9 deg, elevation 62.7 deg). The render script did not disable it. It lights every image from a constant direction. |
| F5 | The Ridge alpha had no effect. The MAE is the same for alpha 0.001 to 1000. |
| F6 | The arrows in the old figures used a wrong camera and an orthographic projection. The mean direction error is 5.2 deg. In 16% of the scenes it is above 10 deg. The maximum is 34 deg. |
| F7 | The old report says that the azimuth turns clockwise. In the image it turns counter-clockwise. The value 0 deg is light from the front right, 90 deg from the back right, 180 deg from the back left and 270 deg from the front left. Blender renders confirm this. |
| F8 | The floor highlight of the key lamp is large and bright. It shows the lamp azimuth directly. It is a likely shortcut cue. |

### 2.3 State after week 1

- The package `dd3d` has the modules geometry, metrics, noise, extract and viz.
- The corrected extraction uses flow noise, a fixed seed for each image and the prompt attention mask. It stores pooled float16 features (287 KB for each image) for a chosen block and timestep.
- Blender 2.78c runs on Linux, so we can make controlled renders.
- The literature table is in `docs/literature.md`. The list of venues is in `docs/article/venues.md`.
- An upstream issue and a patch with tests for the library error are ready in `tools/diffusers_upstream/`. We did not send them yet.

## 3. Research questions and hypotheses

**RQ1.** Does SANA encode the light direction beyond what a linear probe gets from the pixels or from the DC-AE latent?

- H1a: A probe on SANA is not better than a probe on pixels. This is the null hypothesis, and the old data support it.
- H1b: A probe on SANA is better than the pixel probe in some blocks and timesteps, on a data set without shortcut cues.

**RQ2.** What does the probe read: the lamp position (global label) or the light at the object (local label)? Where in the image does it read?

- H2: The probe follows the global label because the floor highlight and the lamp position give a stable cue (F3, F8).

**RQ3.** Which cue carries the information: the floor highlight, the shadow or the shade on the object?

- H3: The floor highlight is the main cue. When we remove it, the error rises. The shadow is the second cue.

**RQ4.** Does the model use the representation? If we remove or move the direction, does the generated light change?

- H4: If the model uses the light direction, erasure lowers the match of the light in the output. An activation shift moves the shadow.

## 4. Planned experiments

All errors use the circular distance for the azimuth. All results have bootstrap intervals (1000 samples over images). The probe code is the same for all models and baselines.

### 4.1 E0: corrected repeat of the old experiment

| Item | Plan |
|---|---|
| Question | What is the result of the old experiment with a correct pipeline? |
| Method | Flow noise with a fixed seed for each image. Prompt attention mask. All 20 blocks. Timesteps 100, 200, 400, 600 and 800. Standardized features. 4x4 average pool, PCA 256, Ridge. We select alpha on the validation split only. |
| Data | Data set v1 (3000 images). Train on spheres and cylinders. Test on the test split and on cubes. |
| Metric | MAE of the azimuth and the elevation in deg, with bootstrap intervals. |
| Success criterion | A table of 20 blocks and 5 timesteps for the trained model. The result shows if the F1 conclusion holds with correct noise. |

### 4.2 RQ1: baselines and controls

| Item | Plan |
|---|---|
| Method | Run the same probe on four inputs: raw pixels, the DC-AE latent, SANA activations, and activations of a SANA transformer with random weights (model randomization check). Run control tasks with shuffled labels and report selectivity. |
| Data | Data set v1 and data set v2 (section 4.4). |
| Metric | MAE on the test split and on cubes. Selectivity = MAE of the control task minus MAE of the real task. |
| Success criterion | SANA is better than the pixels and the latent by more than the bootstrap interval, and better than random weights. If not, we report that SANA does not encode more than the input. |

### 4.3 RQ2: what the probe reads

| Item | Plan |
|---|---|
| Method | Train probes on the global label and on the local label (light direction at the object). Train a probe for the object position. Compute the exact spatial decomposition of the linear probe on the 8x8 grid. Occlude the object, the shadow and the floor highlight, and measure the change of the prediction. Run the sanity checks of the literature (random weights, shuffled labels). |
| Data | Data set v1 with both labels. Masks come from Blender renders of object, shadow and floor. |
| Metric | MAE for each label. Share of the probe output for each grid cell. Change of the MAE after each occlusion. |
| Success criterion | A clear answer for each label: which one has the lower error, and which image region carries the weight. The map must differ from the map of the random model. |

### 4.4 RQ3: dataset v2 and cue conflicts

| Item | Plan |
|---|---|
| Method | Render a new data set. The sun lamp (F4) is off. A directional key light replaces the point lamp, so the floor highlight is absent and the local direction equals the global direction. Make four variants: key light with shadow, without shadow, with the floor highlight, and cue-conflict renders (the shadow and the shade point to different directions). Train and test the probes on each variant. |
| Data | Blender 2.78c renders with the render spec of `dd3d`. Size: 3000 images for each main variant, fewer for the conflict set. |
| Metric | MAE for each variant. In the conflict set: the share of predictions that follow the shadow, the shade or the lamp label. |
| Success criterion | The error change between variants identifies the cue. A difference above the bootstrap interval is a result. |

### 4.5 RQ4: causal tests

| Item | Plan |
|---|---|
| Method | Concept erasure with LEACE in blocks 4, 10 and 18. Activation shift (steer): add a vector along the probe direction to the activations during the denoise process. Generate images with and without the change. Measure the change of the shadow with the pixel probe. |
| Data | Prompts on the same scenes as in data set v2. Fixed seeds. |
| Metric | Change of the predicted azimuth of the output image (pixel probe). Image quality change, with a simple check of the pixel difference outside the shadow. |
| Success criterion | The shadow azimuth moves in the direction of the shift vector, with a dose response. Erasure lowers the match to the input light. If nothing changes, we report that the representation is not used in this way. |

## 5. Planned functionality

The package `dd3d` is a library. All logic is in `src/dd3d/`. The command line is a thin layer on top. We use no notebooks for the main code.

### 5.1 CLI commands

| Command | Function |
|---|---|
| `dd3d extract` | Run SANA on the images and store pooled activations. It runs as a SLURM array job. |
| `dd3d merge` | Merge the parts of an array job into one file for each block and timestep. |
| `dd3d probe` | Train and evaluate the probe. Write metrics and bootstrap intervals. |
| `dd3d baseline` | Run the probe on pixels, on the DC-AE latent, or on random weights (`pixels`, `vae`, `random-init`). |
| `dd3d audit-arrows` | Check the arrows in the figures with the Blender camera. |
| `dd3d viz` | Draw the figures: error tables, prediction arrows, spatial maps. |
| `dd3d render-spec` | Write the Blender render specification for a data set variant. |
| `dd3d erase` | (Later) Apply concept erasure and generate images. |
| `dd3d steer` | (Later) Apply the activation shift during the denoise process. |

### 5.2 Configuration

- All settings are in YAML files in `configs/`. A file defines the data set, the blocks, the timesteps, the seeds and the probe options.
- The code has no hard-coded paths. The command line takes a config file and can override single values.
- The original renders in `data/` are immutable. The commands write only to `data/features/`, `data/audit_renders/` and `reports/`.

### 5.3 Outputs

- Feature files (float16, one file for each block and timestep).
- Metric tables in CSV and JSON, with the config and the commit hash.
- Figures in PNG.
- TensorBoard logs for the probes and for the steer runs.
- Weekly notes in `docs/worklog.md`.

### 5.4 Quality rules

- `ruff` for format and lint. PEP8.
- A virtual environment, and a script (`Makefile` or `scripts/`) for build, test and run.
- Unit tests in `tests/unit/` for geometry, metrics, noise and the probe. The noise test checks that x_t equals (1 - t/1000) x0 + (t/1000) eps.
- Conventional commits.
- Documentation: the README, this proposal, the usage manual in `docs/usage.md`, and the literature file.

## 6. Technology stack

| Area | Tool | Use |
|---|---|---|
| Language | Python 3.10 or newer | All code |
| Model | SANA-1.6B (Hugging Face `diffusers`) | Source of the activations |
| Deep learning | PyTorch | Forward pass, hooks, steer |
| Hugging Face | `diffusers`, `transformers`, `huggingface_hub` | Model, scheduler, text encoder |
| Probes | scikit-learn | PCA, Ridge, metrics |
| Erasure | `concept-erasure` (LEACE) | Concept removal |
| Data | NumPy, pandas | Features and tables |
| Render | Blender 2.78c with the CLEVR generator | Data sets v1 and v2 |
| Tracking | TensorBoard | Curves and images of the runs |
| Config | YAML files, with a small loader | Config separate from code |
| Quality | `ruff`, `pytest`, `pre-commit` | Format, lint, tests |
| Environment | `venv` and `pyproject.toml` | Install of the package |
| Compute | Kaggle jobs (T4 GPU, 4-core CPU), tools in `tools/kaggle/` | GPU and CPU jobs |
| Figures | Matplotlib | Plots |
| Article | LaTeX | Paper, in `docs/article/` |

## 7. Compute plan

The GPU allocation of the Athena cluster is exhausted (2026-10-13). All GPU and render work runs on Kaggle. Kaggle runs at most 5 CPU jobs and 2 GPU jobs at the same time, with 30 GPU hours each week. A queue program (`tools/kaggle/kqueue.py`) pushes the jobs and downloads the results.

| Task | Resource | Measured speed | Estimate |
|---|---|---|---|
| Render v2, v2r and the variants (3000 images each) | Kaggle CPU jobs | 7 s for each image | 6 CPU hours for each set |
| SANA extraction (20 blocks, 5 timesteps) | Kaggle T4, float16 | 1 s for each image | 1 GPU hour for each set |
| Old noise rule (control, v1 only) | Kaggle T4, float32 | 2.5 s for each image | 2 GPU hours |
| Random weights (control) | Kaggle T4, float16 | 1 s for each image | 1 GPU hour for each set |
| SD 1.5 and SDXL (generality) | Kaggle T4, float16 | 0.5 s and 2 s for each image | 2.5 GPU hours for each set |
| Causal tests (shift, erase) | Kaggle T4, float16 | 20 steps for each image | 1 to 2 GPU hours |
| Probes and baselines | Laptop CPU | Minutes for each block and timestep | Continuous |
| Total | GPU | | About 20 GPU hours |

Notes:

- SANA in float16 gives the same features as in float32 (relative difference 0.1%). The old noise rule overflows in float16, so that run uses float32.
- The features of one data set take about 50 GB. The laptop has enough disk for all sets.

## 8. Schedule

The semester runs from October 2026 to mid-February 2027. The date of the last lecture (public presentation) is not known. We plan for the last week of January 2027. This date is to be confirmed.

| Week | Dates (Monday to Sunday) | Planned progress | Deliverable |
|---|---|---|---|
| 1 | 2026-10-12 to 2026-10-18 | Audit of the old pipeline (F1 to F10). Package `dd3d` with corrected extraction. Kaggle jobs and queue. Render of v2. Start of the extraction of v1 and v2. Upstream issue. | Audit results, package `dd3d`, `docs/literature.md`, this proposal, data set v2 |
| 2 | 2026-10-19 to 2026-10-25 | Ask the coordinator about the team size and the old project. Finish the extraction of v1, v2, v2r and the variants. E0 and RQ1 tables. First causal tests. | Approval from the coordinator. Feature files. Tables of E0 and RQ1. |
| 3 | 2026-10-26 to 2026-11-01 | Run extraction v1 for all blocks and 5 timesteps. Run the random-weight model. Pixel and latent baselines. | Feature files v1. Baseline tables. |
| 4 | 2026-11-02 to 2026-11-08 | E0 results. Prototype demo: one CLI run from image to light prediction with a figure. Finish v2 render. Start extraction v2. | Prototype demo. Table of E0. |
| 5 | 2026-11-09 to 2026-11-15 | Finish extraction v2. Run erasure and steer on the GPU. Last planned GPU job on 2026-11-13. | All feature files. Erased and steered images. End of GPU work. |
| 6 | 2026-11-16 to 2026-11-22 | RQ1 analysis: baselines, random weights, selectivity. Send the upstream issue and patch. | RQ1 tables and figures |
| 7 | 2026-11-23 to 2026-11-29 | RQ2 analysis: local and global labels, spatial decomposition, occlusion. | RQ2 tables and figures |
| 8 | 2026-11-30 to 2026-12-06 | RQ3 analysis on v2. Article draft with all sections. | Article draft v1 |
| 9 | 2026-12-07 to 2026-12-13 | RQ4 analysis: effect of erasure and steer on the shadow. Revise the article. | RQ4 figures. Article draft v2. |
| 10 | 2026-12-14 to 2026-12-20 | Small extra runs on Kaggle or Colab if needed. Finish the usage manual and the tests. Internal review of the article. | Usage manual. Reviewed draft. |
| 11 | 2026-12-21 to 2026-12-27 | Buffer. Holiday. No planned work. | None |
| 12 | 2026-12-28 to 2027-01-03 | Buffer. Holiday. Read the article only. | None |
| 13 | 2027-01-04 to 2027-01-10 | Final figures. Write the article in the format of the venue. Reproduction check from a clean environment. | Article draft v3 |
| 14 | 2027-01-11 to 2027-01-17 | Check the call of the venue. Final text. Post to arXiv if the venue allows it. Write the video script. | Final article text. Video script. |
| 15 | 2027-01-18 to 2027-01-24 | Submit the article to TMLR (open at any time), or prepare the IJCNN version. Record the 3 to 5 minute video. | Article submission. Video. |
| 16 | 2027-01-25 to 2027-01-31 | Public presentation (date to be confirmed). IJCNN deadline is 2027-01-31 if we choose it. | Presentation. Slides. |
| 17 | 2027-02-01 to 2027-02-07 | Workshop option: ICLR 2027 workshops (about 2027-02-01). Fix open items in the repository. | Final repository state |
| 18 | 2027-02-08 to 2027-02-14 | Final report to the course. Tag the release. Answer review questions. | Release tag. Course report. |

## 9. Roles

| Role | Tasks |
|---|---|
| Member A | Extraction, Kaggle jobs, probes and baselines, tests |
| Member B | Blender renders, data set v2, figures, spatial analysis |
| Member C (if a third member joins) | Erasure and steer, article text, video |

All members write the weekly note in `docs/worklog.md` and review the code of the others. The tasks of Member C pass to Members A and B if no third member joins.

## 10. Risks and mitigations

| Risk | Effect | Mitigation |
|---|---|---|
| The GPU allocation of the cluster is exhausted (2026-10-13). | No cluster GPU. | All GPU work runs on Kaggle (30 GPU hours each week, plan about 20). Ask for a new PLGrid grant if more is necessary. |
| The course asks for teams of 3 persons and the team has 2. | The grade or scope can change. | Ask the coordinator in week 2. Or add a member. Section 9 shows the plan for both cases. |
| The project continues a project of another course. | A claim of double credit. | State what is new. Ask for approval in week 2. |
| Negative results (SANA is not better than pixels). | The story is less strong. | Plan the study as a careful negative result. TMLR accepts such work. |
| Render time is long. | v2 is late. | Use array jobs on CPU nodes. Start in week 2. Reduce the size of the conflict set if needed. |
| Extraction is slower than planned. | GPU hours run out. | Extract fewer blocks first (4, 10, 18). Use a smaller image subset. |
| The date of the last lecture changes. | Video and slides are late. | Finish the video in week 15, before the planned date. |
| The article deadline is missed. | No submission in the semester. | TMLR accepts papers at any time. IJCNN and the workshops are fall-back options. |

## 11. Bibliography

The full table with venue, summary, code, pretrained models, metrics and compute is in `docs/literature.md`. The list below gives the main references. Links were checked on 2026-10-13.

1. Alain, Bengio. Understanding intermediate layers using linear classifier probes. https://arxiv.org/abs/1610.01644
2. Hewitt, Liang. Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP 2019. https://arxiv.org/abs/1909.03368
3. Belinkov. Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics 2022. https://arxiv.org/abs/2102.12452
4. Belrose et al. LEACE: Perfect linear concept erasure in closed form. NeurIPS 2023. https://arxiv.org/abs/2306.03819
5. Elazar et al. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. TACL 2021. https://arxiv.org/abs/2006.00995
6. Kim et al. Interpretability Beyond Feature Attribution: TCAV. ICML 2018. https://arxiv.org/abs/1711.11279
7. Adebayo et al. Sanity Checks for Saliency Maps. NeurIPS 2018. https://arxiv.org/abs/1810.03292
8. Geirhos et al. ImageNet-trained CNNs are biased towards texture. ICLR 2019. https://arxiv.org/abs/1811.12231
9. Chen, Viegas, Wattenberg. Beyond Surface Statistics: Scene Representations in a Latent Diffusion Model. 2023. https://arxiv.org/abs/2306.05720
10. Zhan, Zheng, Xie, Zisserman. A General Protocol to Probe Large Vision Models for 3D Physical Understanding. NeurIPS 2024. https://arxiv.org/abs/2310.06836
11. El Banani et al. Probing the 3D Awareness of Visual Foundation Models. CVPR 2024. https://arxiv.org/abs/2404.08636
12. Kwon, Jeong, Uh. Diffusion Models already have a Semantic Latent Space. ICLR 2023. https://arxiv.org/abs/2210.10960
13. Helbling et al. ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features. ICML 2025. https://arxiv.org/abs/2502.04320
14. Xie et al. SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers. ICLR 2025. https://arxiv.org/abs/2410.10629
15. Johnson et al. CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning. CVPR 2017. https://arxiv.org/abs/1612.06890
16. Lipman et al. Flow Matching for Generative Modeling. ICLR 2023. https://arxiv.org/abs/2210.02747
