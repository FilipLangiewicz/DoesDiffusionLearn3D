# Worklog

This file is the weekly standup log of the team. Each week has one entry. We write the entry at the end of the week. The entry has the sections Done, Findings, Next steps and Blockers. The newest entry is at the end. Each member reads the entry before the team meets each week.

## Week 1 (2026-10-12 to 2026-10-18)

### Done

- We audited the old pipeline of the summer 2026 project.
- We wrote the package `dd3d` with the modules geometry, metrics, noise, extract and viz.
- The corrected extraction uses flow noise, a fixed seed for each image and the prompt attention mask. It stores pooled float16 features (287 KB for each image) for a block and a timestep.
- Blender 2.78c runs on Linux. We can make controlled renders.
- We wrote the literature table (`docs/literature.md`) and the venue list (`docs/article/venues.md`).
- We prepared an upstream issue and a patch with tests for the library error in `tools/diffusers_upstream/`. We did not send them yet.
- We wrote the design proposal (`docs/design_proposal.md`).
- Update 2026-10-13: the GPU allocation of the cluster is exhausted. We moved the renders and the GPU work to Kaggle. The tools are in `tools/kaggle/`, and `docs/usage.md` section 6 tells how to use them.
- We rendered data set v2 (3000 images) on 6 Kaggle CPU jobs. It has one sun lamp, a flat ground, the object at the scene centre and an elevation from 30 to 75 deg.
- We added two render variants with the same scenes as v2. In the first variant, the object casts no shadow. In the second variant, the camera sees only the shadow.
- We started the corrected extraction of v1 and v2 on Kaggle T4 GPUs. The runs are SANA, SANA with random weights, the old noise rule (v1 only), SD 1.5 and SDXL.
- Update 2026-10-14: all renders (v2, v2pos, v2ns, v2r, v2rns, v2rso) and all feature extractions are complete on Kaggle.
- Update 2026-10-15: the probes run as Kaggle CPU jobs next to the features (`tools/kaggle/templates/analyze.py`). `dd3d.store` reads the feature chunks in place, so no job unpacks the 10 GB tar files.
- We ran the probe scan on all five data sets: every block, five timesteps, four grids, pixels at four resolutions, the latent, random weights, SD 1.5 and SDXL.
- We ran the causal test on 60 images of v2 (shift along the probe and along the activation pattern, a random shift, LEACE erasure) in blocks 4, 10 and 18.
- We rendered 48 pairs of v2 scenes with the light turned by 90 deg (v2rot) and ran the patch test in 10 blocks and 3 regions.
- We added a second measure of the output light: the direction of the dark shadow region, with no fitted model.
- We built an interactive viewer (`tools/viewer/`) and rewrote the article and the status report with all results.
- We removed the local copies of the Kaggle data from the laptop. The labels, splits and three example images of each version (`data/examples/`) stay in git.

### Findings

- F1: The same probe on raw pixels gives 14.09 / 6.25 deg on cubes. SANA gives 13.01 / 6.88 deg on the test split, and the pixels give 11.45 / 5.60 deg. The old result does not show more than the pixels.
- F2: The old extraction called `add_noise` before `set_timesteps`. The noise level was wrong. The same problem affects the VP schedules of four schedulers.
- F3: The label is the lamp position. The light at the object differs by 35 deg in azimuth and 16 deg in elevation (mean). The probe follows the label.
- F4: The base scene has a fourth lamp "Area" (SUN, strength 0.45). It lights every image from a constant direction.
- F5: The Ridge alpha had no effect on the MAE.
- F6: The old arrows had a mean direction error of 5.2 deg because of a wrong camera and an orthographic projection.
- F7: The azimuth turns counter-clockwise in the image. The old report says clockwise.
- F8: The floor highlight of the key lamp shows the azimuth directly. It is a likely shortcut cue.
- F9: With the old noise rule, the new extraction gives a mean token norm of 11,147 in block 4 at t = 200. The old features have about 11,100. With the flow rule, the norm is 718. Thus the new pipeline agrees with the old pipeline, except for the noise.
- F10: SANA in float16 gives the same features as SANA in float32 (relative difference 0.1%) and is 4 times faster on a T4. The old noise rule overflows in float16 in block 18, so that run uses float32.
- F11: On v1 with the flow noise, SANA gives 4.8 deg on cubes and the best pixel grid gives 12.8 deg. With the old noise rule SANA gives 14.5 deg. The wrong noise hid the information.
- F12: On v2r (random position), SANA gives 7.6 deg on cubes. The best pixel grid gives 20.1 deg, the latent 19.0 deg and random weights 15.9 deg. SDXL gives 7.3 deg and SD 1.5 gives 11.1 deg.
- F13: A probe on the mean of all tokens, fitted on v2, gives 9.0 / 17.6 deg on v2r. The pixel probe gives 71 deg. The code does not depend on the position.
- F14: SANA reads the light from the shading alone (9.9 deg) and from the shadow alone (7.9 deg).
- F15: A shift along the activation pattern (2% of the token norm) turns the pixel probe reading by 60 to 65 deg towards a target of 90 deg. A random shift of the same size does nothing. The crisp cast shadow stays (68% to 80% of it is kept). With 3 times the shift, only 32% to 40% of the old shadow stays, and a diffuse dark region appears towards the target. The images show no new crisp shadow. More noise (sigma0 0.85, 0.95) does not change this.
- F16: A patch of the object tokens moves the shading (0.59 of the change) and about a third of the crisp cast shadow (33% to 38% of the source shadow appears; the reference is 12%). Only a patch of all tokens moves the shadow fully (84%).
- F17: The pixel probe on the outputs reads mostly the object. A claim about the shadow needs a measure of the shadow.
- F18: A shift of the pattern in all 20 blocks together moves every measure strongly (+120 deg at a quarter of the natural size). The images show dark and bright blocks on the grid of the shift, not a new cast shadow. No shift drew a new crisp shadow.

### Audit of 2026-10-17

- A validation audit checked the article, the documents, the code and the course rules. The report is `docs/audit/validation_2026-10-17/RAPORT.md`.
- Two independent readers read only the article or only the documents. Their notes are in the same folder.
- Four checks ran on Kaggle (CPU). They changed these findings:
  - F16 is not correct. Most of the source shadow that a patch of the object tokens shows lies in the patched cells. The patch copies these pixels. One token or more away from these cells, the source shadow does not appear (5% against 3% without a patch).
  - The LEACE eraser of the causal test removed only the code that is the same at every position. A probe on the grid read the azimuth after the eraser as well as before (4.6 deg and 4.7 deg). Thus the erasure result in F15 does not show that the code is unused.
  - The shift along the pattern turns the light and dark sides of the object. A probe on a crop of the object, without the floor and the shadow, shows this (F15 stays). The values "60 to 65 deg" are readings of the probe on the whole image.
  - Two calibration values of the shadow measures were wrong. The dark-region error on cubes is 8.7 deg (not 10.3 deg). The kept share for a light turned by 90 deg is 0.07 to 0.09 (not 0.00 to 0.01).
- The fixes are on the branch `fix/audit-findings`:
  - one LEACE eraser for each grid cell;
  - DINOv2 and CLIP as encoders that do not generate images;
  - a measure on the object crop, and shadow shares outside the patched cells;
  - a calibration of the output measures;
  - tests of the probe and of the measures;
  - `docs/REPRODUCE.md`.
- The new runs gave these findings (all values in `docs/RESULTS.md`):
  - F19: DINOv2 and CLIP do not generate images, but they read the light direction equally well or better on the known shapes (v2r: 3.3 and 3.9 deg, SANA 5.4 deg). On cubes, SANA is better only when the object moves and its shading is visible (v2r: 7.6 deg, DINOv2 10.0 deg, CLIP 16.0 deg). A readable light direction is thus not specific to a generative model.
  - F20: The probe on the object crop confirms F15 for the shading: the pattern shift (x1) turns the object by 50 to 52 deg.
  - F21: A patch of the object tokens turns the shading of the object (0.36 to 0.45 of the turn by the object probe, against 0.56 for the full source pass). Outside the patched cells, 3% to 9% of the source shadow appears, against 3% without a patch. This replaces F16.
  - F22: After the cell eraser, a new probe on the grid is at chance (88 deg). In blocks 4, 10 and 18 at t = 400, the output error of the light grows from 7.6 deg to 21.9, 30.4 and 44.2 deg. A control with cell erasers of shuffled labels gives 8.7 to 9.6 deg. But the control changes the features about three times less, and the cell eraser adds grey smears near the object. Thus the later blocks probably use the code, but the evidence is not final. At t = 100, the control also breaks the output (31.1 deg).
- `docs/RESULTS.md` (Polish) is now the only summary of the results. It replaces `docs/STAN_PRAC.md` and `docs/PODSUMOWANIE.md`. The explanation of the library error is now in `tools/diffusers_upstream/WYJASNIENIE.md`.

### Deviations from the plan

- The plan (`docs/design_proposal.md`) put E0 in week 4, RQ1 in weeks 6 and 7, RQ3 in week 8 and RQ4 in week 9. Kaggle GPU jobs ran in parallel, so all four studies ran in week 1.
- The design proposal named the CLI commands `baseline`, `render-spec`, `erase` and `steer`. The work used scripts in `scripts/experiments/` and Kaggle job specs in their place.
- The design proposal planned TensorBoard logs. The experiments record their settings in the Kaggle job specs, with the commit of each run, and their results in `reports/experiments/`.
- The upstream issue for the library error is sent (diffusers issue 15021).

### Next steps

- Week 2 (2026-10-19 to 2026-10-25): finish the fixes of the audit and change the article to the new results.
- Ask the coordinator about the team size and the old project.
- Review the language of the article and select the venue.
- Show the viewer and `docs/RESULTS.md` at the next meeting.

### Blockers

- The cluster has no GPU hours left. All GPU work runs on Kaggle (30 GPU hours each week).
- The team size: the course rules ask for 3 persons and the team has 2. We need an answer from the coordinator.
