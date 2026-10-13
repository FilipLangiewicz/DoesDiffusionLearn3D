# Audit of the light labels, the predictions and the arrows

Date: 2026-10-13. Scope: the dataset in `data/final` (3000 scenes), the ZZSN probe outputs in `probe_exp/`, and the figure script `legacy/src/test_cubes/visualize_predictions.py`.

The audit answers three questions:

1. How does the dataset define the azimuth and the elevation?
2. Does the code decode and score the predictions correctly?
3. Do the arrows in the figures show the correct directions?

## 1. Method

We copied the render setup of `clevr-dataset-gen/image_generation/render_images.py` into code that we can test:

- `src/dd3d/geometry.py`: the camera, the lamp, the shadow and the projection rules of Blender 2.78c.
- `tools/blender/render_spec.py`: renders of any scene with the same rules, with Blender 2.78c on Linux.

Two checks show that the copy is exact:

- The camera model gives the `pixel_coords` of all 3000 scene JSON files with a maximum error of 0.56 px. The fitted camera jitter has the expected uniform distribution in [-0.5, 0.5].
- A new render of scene `SmoothCube_v2_new_000004` with the values from its JSON file is equal to the dataset image. The mean absolute difference is 0.14 of 255, and the object masks have an IoU of 0.98.

The file `base_scene.blend` is identical to the original CLEVR file (same MD5 sum).

## 2. How the dataset defines the light direction

| Item | Fact | Source |
|---|---|---|
| Lamp | `Lamp_Key` is an AREA lamp (size 0.5, strength 100) with a TRACK_TO constraint to the world origin. | base_scene.blend |
| Lamp position | x = 5 cos(el) cos(az), y = 5 cos(el) sin(az), z = 5 sin(el), around the world origin. | render_images.py |
| Label | `light_azimuth_deg` and `light_elevation_deg` are the angles of the lamp position around the world origin. | render_images.py |
| Azimuth direction | The azimuth turns counter-clockwise from +x when you look down from +z. | geometry, renders |
| Azimuth in the image | 0 deg: light from the front right (lower right of the image). 90 deg: back right. 180 deg: back left. 270 deg: front left. The angle turns counter-clockwise in the image too. | renders `conv_az*.png` |
| Other lamps | `Lamp_Fill` and `Lamp_Back` are off. A fourth lamp `Area` (type SUN, strength 0.45, from azimuth 334.9 deg and elevation 62.7 deg) stays on. | base_scene.blend, renders |
| World light | White background light with strength 0.2, used as a light source. | render_images.py |
| Camera | TRACK_TO constraint to the origin, lens 35 mm, sensor 32 mm, 512 x 512 px. Each image adds a random jitter in [-0.5, 0.5] to each axis. The JSON file does not store the jitter. | base_scene.blend, render_images.py |

### Findings about the labels

- **L1. The report figure is wrong about the direction.** Figure 4.1 of the ZZSN report says that the azimuth turns clockwise. It turns counter-clockwise, in the world and in the image. The Blender renders in `data/audit_renders/conv_*.png` (make them with `scripts/audit/make_render_specs.py`) confirm this.
- **L2. The label is not the light direction at the object.** The lamp turns around the world origin, but the object stands up to 2.8 units away from the origin. The light direction at the object differs from the label by 35.0 deg in azimuth (mean). The median is 27.2 deg and the maximum is 180 deg. In elevation, the mean absolute difference is 16.2 deg. The light elevation at the object is in the range 8.4 to 88.8 deg, not 30 to 75 deg.
- **L3. The floor highlight shows the label directly.** The area lamp faces the origin, so it makes a large bright spot on the floor between the lamp and the origin. The position of this spot follows the label, not the object. See `docs/audit/figures/` and section 5.
- **L4. An extra lamp lights every image.** The sun lamp `Area` of the base scene stays on. Together with the world light, it gives 19% of the floor brightness (the key lamp gives 81%). Its shadow is almost invisible, because it falls behind the object as the camera sees it. Thus its effect is a constant front light on the object.
- **L5. The JSON field `directions` is the same for all scenes.** The script reads the camera matrix before Blender applies the jitter. The field `pixel_coords` comes later and includes the jitter.
- **L6. The JSON field `rotation` is not in degrees.** CLEVR writes the value in degrees to a field in radians. The real rotation is `rotation` radians. This does not change the light labels.
- **L7. The first image of each shape has the elevation 45.0 deg.** The argument `--light_elevation` has the default 45.0, so the random value applies only from the second image. This affects 3 of 3000 images.

## 3. How the code decodes and scores the predictions

- The probe predicts `[sin az, cos az, el]`. The code decodes the azimuth with `atan2(sin, cos) mod 360`. This is correct.
- The azimuth MAE uses `min(|d|, 360 - |d|)`. This is correct.
- The rows of the features and the labels come from the same split table, so they agree. A pixel probe with the same split files gives errors of the same size, which confirms the order.
- The scores are correct for the label as defined. The label itself is the problem (L2, L3).

## 4. The arrows in the ZZSN figures

The script `legacy/src/test_cubes/visualize_predictions.py` draws the arrows. We compared its arrows with the exact geometry for all 3000 scenes (`scripts/audit/arrow_audit.py`, output in `reports/audit/arrow_audit.csv`).

- **A1. Wrong camera.** The script uses the default camera position of Blender (7.36, -6.93, 4.96), not the CLEVR camera (7.48, -6.51, 5.34). It also ignores the jitter.
- **A2. Orthographic projection.** The script projects the ground vector on the camera axes without perspective. With A1, the direction error is 5.2 deg on average (median 2.9 deg, 90th percentile 13.2 deg, maximum 34.5 deg). In 15.8% of the scenes the error is more than 10 deg.
- **A3. Missing arrows.** The arrows have a fixed length of 143 px from the object centre. Matplotlib does not draw an annotation arrow that ends outside the image. Thus 18 to 19% of the label arrows and of the prediction arrows are not in the figures (test and test_cubes splits).
- **A4. The arrows show the shadow, not the light.** The arrows point from the object to its shadow, away from the lamp. The README caption said "light direction". The README now says "shadow direction".
- **A5. No elevation in the arrows.** All arrows have the same length, so an elevation error is not visible.
- **A6. The arrow starts at the object centre.** The shadow starts at the object base, on average 56 px lower in the image.

### Check with real shadows

For 8 scenes we rendered the scene without the object and with the key lamp only (`scripts/audit/shadow_audit.py`). The difference of two renders gives the real cast shadow.

- For the 6 scenes with long shadows, the exact arrows point to the measured shadow with a mean error of 4.7 deg. The legacy arrows have a mean error of 7.6 deg.
- For 2 spheres the shadow is short (13 and 20 px, light at the object from 66 and 71 deg elevation). The measurement is not reliable there.
- The extra shadow of the sun lamp is 0.1 to 5% of the key lamp shadow.

Figure: `reports/audit/shadow_audit.png`.

### Corrected arrows

`src/dd3d/viz.py` (command `dd3d viz`) draws the arrows from the object base to the shadow of the object centre. It uses the exact camera of each image (fitted jitter) and the full perspective projection. The arrow length shows the elevation, and the arrow stops at the image border. Examples: `reports/viz/`.

## 5. Consequences for the experiment

- The ZZSN probe follows the label (block 10, t = 200, cubes). Its azimuth MAE is 14.0 deg for the label and 35.5 deg for the light direction at the object.
- A probe on raw pixels gives the same error as the SANA probe. Section "Pixel cues" in `reports/experiments/pixel_cues/results.md` shows which image region carries the information.
- The floor highlight (L3) is a shortcut that does not depend on the object shape. This explains why the probe "generalizes" to cubes.

## 6. Recommendations for dataset v2

1. Switch off the sun lamp `Area` (`hide_render = True`), or record it in the JSON file.
2. Use a SUN key lamp. Then the light direction is the same at every point, and there is no floor highlight.
3. If the key lamp stays an area lamp, store the light direction at the object as the label.
4. Set the camera jitter to 0, or store the camera position in the JSON file.
5. Call `bpy.context.scene.update()` before the script reads the camera matrix.
6. Convert `theta` to radians before the script writes it to `rotation_euler`.
7. Store the lamp type, size and strength in the JSON file.

## 7. How to repeat the audit

```bash
make install                                   # once
python scripts/audit/arrow_audit.py            # all scenes, about 2 min
python scripts/audit/make_render_specs.py --spec-dir /tmp/specs
blender -b -noaudio --python tools/blender/render_spec.py -- /tmp/specs/conventions.json "$PWD"
blender -b -noaudio --python tools/blender/render_spec.py -- /tmp/specs/shadows.json "$PWD"
python scripts/audit/shadow_audit.py
dd3d viz probe_exp/test_cubes_eval_all/layer_10_t_200/test_cubes_predictions.csv
```

Blender 2.78c needs the libraries `libGLU.so.1` and `libOpenGL.so.0`. Without root access, download the Ubuntu packages `libglu1-mesa` and `libopengl0` with `apt-get download` and unpack them with `dpkg -x`. Then put the folder on `LD_LIBRARY_PATH`.

## 8. Test of the light options (2026-10-13)

`scripts/audit/lighting_study.py` renders a sphere at the scene centre with four light setups. Each setup has six light directions. Means over the six directions:

| Setup | Floor highlight | Shadow depth | Object contrast | Shadow direction error |
|---|---:|---:|---:|---:|
| Current dataset (AREA key lamp, sun lamp Area on) | 48.9 | 36.3 | 48.7 | 10.7 deg |
| Sun lamp Area off | 58.5 | 47.2 | 52.4 | 10.0 deg |
| SUN key lamp, flat ground | 0.1 | 86.2 | 77.1 | 7.1 deg |
| SUN key lamp, flat ground, world light 0.05 | 0.0 | 87.8 | 89.5 | 5.3 deg |

The values are luminance differences in the range 0 to 255. Figure: `reports/audit/lighting_study.png`.

- If you only switch off the sun lamp Area, the floor highlight stays. The image becomes darker, and the shadow becomes a little deeper.
- A SUN key lamp removes the floor highlight completely. The shadow is sharp and deep, and the object shows more contrast.
- **The SUN key lamp needs a flat ground.** The CLEVR ground is a curved backdrop with a wall behind the scene. The wall blocks a sun lamp from behind (azimuth near 100 to 200 deg), so the whole visible floor goes into shadow. The render option `flat_ground` replaces the backdrop with a flat plane of the same material.
- With a SUN key lamp, the mean floor brightness depends on the elevation only. This is correct physics, but a probe can read the elevation from the mean brightness. The pixel control shows this effect.

`scripts/data/make_v2_spec.py` writes the render spec and the labels of dataset v2 with these rules. Sample: `docs/figures/v2_sample.png`.
