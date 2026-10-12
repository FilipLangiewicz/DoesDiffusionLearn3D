"""Render scenes from a JSON spec with the same rules as render_images.py.

Run inside Blender 2.78c:

    blender -b -noaudio --python tools/blender/render_spec.py -- SPEC.json REPO_ROOT

The spec is a list of jobs. Each job has these keys:

- out: the output PNG path.
- shape: Sphere, SmoothCylinder, SmoothCube_v2 or null (no object).
- size: the object scale (1.0 for large, 1.4 for xlarge).
- xy: the object position on the ground.
- theta: the JSON rotation value (CLEVR writes it to a radian field).
- az, el: the Lamp_Key angles in degrees (radius 5 around the origin).
- jitter: the camera jitter [dx, dy, dz].
- key_on: if true, Lamp_Key renders.
- sun_on: if true, the sun lamp Area of the base scene renders.
- samples: the Cycles samples (the dataset used 64).

The script copies render_scene() and add_random_objects() of render_images.py,
so that a job with the values of a dataset scene gives the same image.
"""

import json
import math
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
SPEC_PATH, REPO = argv[0], argv[1]
GEN_DIR = os.path.join(REPO, "clevr-dataset-gen", "image_generation")
sys.path.insert(0, GEN_DIR)
import utils  # noqa: E402  (CLEVR helper module, needs bpy)

DATA = os.path.join(GEN_DIR, "data")
PURPLE = [129 / 255.0, 38 / 255.0, 192 / 255.0, 1.0]

with open(SPEC_PATH) as f:
    jobs = json.load(f)

for job in jobs:
    bpy.ops.wm.open_mainfile(filepath=os.path.join(DATA, "base_scene.blend"))
    utils.load_materials(os.path.join(DATA, "materials"))

    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.filepath = job["out"]
    render.resolution_x = job.get("width", 512)
    render.resolution_y = job.get("height", 512)
    render.resolution_percentage = 100
    render.tile_x = 256
    render.tile_y = 256
    bpy.data.worlds["World"].cycles.sample_as_light = True
    scene.cycles.blur_glossy = 2.0
    scene.cycles.samples = job.get("samples", 64)
    scene.cycles.transparent_min_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.film_transparent = True

    cam = bpy.data.objects["Camera"]
    for i in range(3):
        cam.location[i] += job.get("jitter", [0.0, 0.0, 0.0])[i]

    key = bpy.data.objects["Lamp_Key"]
    az, el = math.radians(job["az"]), math.radians(job["el"])
    key.location[0] = 5.0 * math.cos(el) * math.cos(az)
    key.location[1] = 5.0 * math.cos(el) * math.sin(az)
    key.location[2] = 5.0 * math.sin(el)
    key.hide_render = not job.get("key_on", True)

    for name in ("Lamp_Fill", "Lamp_Back"):
        bpy.data.objects[name].data.energy = 0.0
        bpy.data.objects[name].hide_render = True
    bpy.data.objects["Area"].hide_render = not job.get("sun_on", True)

    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = job.get("world_strength", 0.2)
    bpy.data.objects["Ground"].cycles.is_shadow_catcher = False

    if job.get("shape"):
        size = job["size"]
        utils.add_object(
            os.path.join(DATA, "shapes"), job["shape"], size, tuple(job["xy"]), theta=job["theta"]
        )
        utils.add_material("Rubber", Color=job.get("color", PURPLE))

    scene.update()
    bpy.ops.render.render(write_still=True)
    print("RENDERED", job["out"])
