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

### Findings

- F1: The same probe on raw pixels gives 14.09 / 6.25 deg on cubes. SANA gives 13.01 / 6.88 deg on the test split, and the pixels give 11.45 / 5.60 deg. The old result does not show more than the pixels.
- F2: The old extraction called `add_noise` before `set_timesteps`. The noise level was wrong. The same problem affects the VP schedules of four schedulers.
- F3: The label is the lamp position. The light at the object differs by 35 deg in azimuth and 16 deg in elevation (mean). The probe follows the label.
- F4: The base scene has a fourth lamp "Area" (SUN, strength 0.45). It lights every image from a constant direction.
- F5: The Ridge alpha had no effect on the MAE.
- F6: The old arrows had a mean direction error of 5.2 deg because of a wrong camera and an orthographic projection.
- F7: The azimuth turns counter-clockwise in the image. The old report says clockwise.
- F8: The floor highlight of the key lamp shows the azimuth directly. It is a likely shortcut cue.

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
