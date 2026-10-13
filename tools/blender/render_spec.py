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
- key_type: AREA (dataset) or SUN (direction only, no floor highlight).
- key_strength, key_soft: the strength and the shadow softness of a SUN key lamp.
- world_strength: the strength of the white world light (dataset: 0.2).
- flat_ground: if true, a flat plane replaces the curved CLEVR backdrop.
- object_shadow: if false, the object casts no shadow (only its shade is left).
- object_camera: if false, the camera does not see the object, but the object
  still casts its shadow (only the shadow is left).

The environment variable DD3D_DEVICE=GPU renders with CUDA. Use one device for
a full data set, because the CPU and the GPU give different noise.

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

USE_GPU = os.environ.get("DD3D_DEVICE", "CPU").upper() == "GPU"
# Small tiles keep all CPU cores busy; the GPU is fastest with one large tile.
TILE = 256 if USE_GPU else 32

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
    render.tile_x = TILE
    render.tile_y = TILE
    if USE_GPU:
        bpy.context.user_preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        scene.cycles.device = "GPU"
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
    if job.get("key_type", "AREA") == "SUN":
        # A sun lamp has only a direction. The TRACK_TO constraint on the origin makes
        # this direction equal to the label direction at every point of the scene.
        key.data.type = "SUN"
        key.data.shadow_soft_size = job.get("key_soft", 0.02)
        key.data.node_tree.nodes["Emission"].inputs["Strength"].default_value = job.get("key_strength", 3.0)

    for name in ("Lamp_Fill", "Lamp_Back"):
        bpy.data.objects[name].data.energy = 0.0
        bpy.data.objects[name].hide_render = True
    bpy.data.objects["Area"].hide_render = not job.get("sun_on", True)

    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = job.get("world_strength", 0.2)
    ground = bpy.data.objects["Ground"]
    ground.cycles.is_shadow_catcher = False
    if job.get("flat_ground", False):
        # The CLEVR ground is a curved backdrop; its back wall blocks a sun lamp from
        # behind. A large flat plane with the same material removes the wall.
        bpy.ops.mesh.primitive_plane_add(radius=200.0, location=(0.0, 0.0, 0.0))
        bpy.context.object.data.materials.append(ground.active_material)
        ground.hide_render = True
        # CLEVR's add_object moves all selected objects, so nothing may stay selected.
        bpy.ops.object.select_all(action="DESELECT")

    if job.get("shape"):
        size = job["size"]
        utils.add_object(
            os.path.join(DATA, "shapes"), job["shape"], size, tuple(job["xy"]), theta=job["theta"]
        )
        utils.add_material("Rubber", Color=job.get("color", PURPLE))
        # add_object leaves the new object as the active object.
        obj = bpy.context.scene.objects.active
        obj.cycles_visibility.shadow = job.get("object_shadow", True)
        obj.cycles_visibility.camera = job.get("object_camera", True)

    scene.update()
    if job.get("debug"):
        for ob in bpy.data.objects:
            print("POS", ob.name, tuple(round(v, 3) for v in ob.location), ob.hide_render)
    bpy.ops.render.render(write_still=True)
    print("RENDERED", job["out"])
