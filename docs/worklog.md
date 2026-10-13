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

### Next steps

- Week 2 (2026-10-19 to 2026-10-25): ask the coordinator about the team size and the old project.
- Get the cluster access, and test the extraction on 100 images.
- Start the render jobs for data set v2.
- Add unit tests for the noise and geometry modules.

### Blockers

- The cluster access: we need it for the corrected extraction. The access ends in November 2026.
- The team size: the course rules ask for 3 persons and the team has 2. We need an answer from the coordinator.

## Week N (YYYY-MM-DD to YYYY-MM-DD)

### Done

-

### Findings

-

### Next steps

-

### Blockers

-
