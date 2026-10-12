"""Scene geometry of the rendered light dataset.

The functions copy the camera, lamp and projection rules of
``clevr-dataset-gen/image_generation/render_images.py`` and of Blender 2.78c.
With the camera jitter from :func:`fit_camera_jitter`, :func:`project` gives
the ``pixel_coords`` of the scene JSON files to 0.6 px.

Conventions (world frame of Blender, z up):

* The azimuth ``az`` turns counter-clockwise from the +x axis when you look
  down from +z. In the image, the azimuth also turns counter-clockwise.
* The elevation ``el`` is the angle above the ground plane.
* The labels ``light_azimuth_deg`` and ``light_elevation_deg`` give the
  position of the lamp ``Lamp_Key`` on a sphere with radius 5 around the
  world origin. They do not give the light direction at the object.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

# Camera.matrix_world translation in base_scene.blend (identical to CLEVR).
BASE_CAMERA = np.array([7.481132, -6.50764, 5.343665])
LENS_MM = 35.0
SENSOR_MM = 32.0
IMAGE_SIZE = 512
LIGHT_RADIUS = 5.0
CAMERA_JITTER = 0.5

# The base scene has a fourth lamp "Area" (type SUN, strength 0.45) on the
# rendered layer. render_images.py does not change it, so it lights every image.
# This is the direction of travel of its light (world frame).
SUN_TRAVEL_DIRECTION = np.array([-0.4145, 0.1946, -0.889])


def lamp_position(az_deg, el_deg, radius: float = LIGHT_RADIUS) -> np.ndarray:
    """Return the Lamp_Key position for the label angles (shape ``(..., 3)``)."""
    az = np.radians(np.asarray(az_deg, dtype=float))
    el = np.radians(np.asarray(el_deg, dtype=float))
    return np.stack(
        [radius * np.cos(el) * np.cos(az), radius * np.cos(el) * np.sin(az), radius * np.sin(el)],
        axis=-1,
    )


def direction_to_az_el(vec) -> tuple[np.ndarray, np.ndarray]:
    """Return azimuth in [0, 360) and elevation in degrees of 3D vectors."""
    v = np.asarray(vec, dtype=float)
    az = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    el = np.degrees(np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1])))
    return az, el


def local_light_direction(obj_xyz, az_deg, el_deg) -> tuple[np.ndarray, np.ndarray]:
    """Return the azimuth and elevation of the lamp as seen from the object centre."""
    return direction_to_az_el(lamp_position(az_deg, el_deg) - np.asarray(obj_xyz, dtype=float))


def sun_position_az_el() -> tuple[float, float]:
    """Return the azimuth and elevation of the constant sun lamp (towards the sun)."""
    az, el = direction_to_az_el(-SUN_TRAVEL_DIRECTION)
    return float(az), float(el)


def camera_axes(cam_pos) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the right, up and forward unit vectors of the camera.

    The camera has a TRACK_TO constraint (track -Z, up Y) on the empty at the
    world origin, so the jitter changes the view direction too.
    """
    cam = np.asarray(cam_pos, dtype=float)
    fwd = -cam / np.linalg.norm(cam)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def project(points, cam_pos=BASE_CAMERA, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Project world points to pixels like ``world_to_camera_view`` of Blender.

    Return ``(..., 3)`` with pixel x, pixel y (origin top left, y down) and the
    depth along the view axis. CLEVR rounds pixel x and y to integers in the
    JSON files. This function does not round.
    """
    right, up, fwd = camera_axes(cam_pos)
    d = np.asarray(points, dtype=float) - np.asarray(cam_pos, dtype=float)
    x_cam, y_cam, depth = d @ right, d @ up, d @ fwd
    scale = LENS_MM / SENSOR_MM
    px = image_size * (0.5 + scale * x_cam / depth)
    py = image_size - image_size * (0.5 + scale * y_cam / depth)
    return np.stack([px, py, depth], axis=-1)


def fit_camera_jitter(obj_xyz, pixel_coords, bound: float = CAMERA_JITTER) -> tuple[np.ndarray, float]:
    """Find the camera jitter that gives the JSON ``pixel_coords`` of the object.

    The jitter is not in the JSON files. The fit uses pixel x, pixel y and the
    depth, with the bounds of the uniform jitter. Objects near the origin give
    a weak fit, because the camera always looks at the origin.
    Return the jitter and the largest absolute pixel residual.
    """
    p = np.asarray(obj_xyz, dtype=float)
    target = np.asarray(pixel_coords, dtype=float)

    def residual(j):
        pr = project(p, BASE_CAMERA + j)
        # CLEVR does not round the depth, so it gets a large weight.
        return np.r_[pr[:2] - target[:2], 100.0 * (pr[2] - target[2])]

    sol = least_squares(residual, np.zeros(3), bounds=(-bound, bound))
    return sol.x, float(np.abs(sol.fun[:2]).max())


def shadow_point(point, lamp) -> np.ndarray:
    """Return the ground point (z = 0) of the shadow of ``point`` for a point lamp."""
    p = np.asarray(point, dtype=float)
    lmp = np.asarray(lamp, dtype=float)
    t = lmp[..., 2] / (lmp[..., 2] - p[..., 2])
    return lmp + t[..., None] * (p - lmp)


def shadow_segment_px(obj_xyz, lamp, cam_pos=BASE_CAMERA) -> tuple[np.ndarray, np.ndarray]:
    """Return the image segment from the object base to the shadow of its centre.

    The segment shows the direction and the length of the cast shadow of the
    key lamp. Both ends use the full perspective projection.
    """
    p = np.asarray(obj_xyz, dtype=float)
    base = p * np.array([1.0, 1.0, 0.0])
    tip = shadow_point(p, lamp)
    return project(base, cam_pos)[..., :2], project(tip, cam_pos)[..., :2]


def image_angle_deg(start_px, end_px) -> np.ndarray:
    """Return the angle of an image vector in degrees, counter-clockwise from +x (y up)."""
    d = np.asarray(end_px, dtype=float) - np.asarray(start_px, dtype=float)
    return np.degrees(np.arctan2(-d[..., 1], d[..., 0])) % 360.0


def legacy_arrow_angle_deg(obj_xyz, lamp) -> np.ndarray:
    """Return the arrow angle of the ZZSN script ``visualize_predictions.py``.

    That script used the default Blender camera position (not the CLEVR camera),
    no jitter and an orthographic projection of the ground shadow vector.
    """
    cam = np.array([7.358891, -6.925791, 4.959234])
    fwd = -cam / np.linalg.norm(cam)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    p = np.asarray(obj_xyz, dtype=float)
    ground = shadow_point(p, lamp) - p * np.array([1.0, 1.0, 0.0])
    return np.degrees(np.arctan2(ground @ up, ground @ right)) % 360.0


def angle_diff_deg(a, b) -> np.ndarray:
    """Return the absolute difference of two angles in degrees, in [0, 180]."""
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.minimum(d, 360.0 - d)


def azimuth_to_image_angle(az_deg, cam_pos=BASE_CAMERA) -> np.ndarray:
    """Return the image angle of a ground direction with azimuth ``az`` at the origin."""
    az = np.radians(np.asarray(az_deg, dtype=float))
    tip = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=-1)
    origin = np.zeros(3)
    return image_angle_deg(project(origin, cam_pos)[:2], project(tip, cam_pos)[..., :2])


def real_rotation_deg(json_rotation: float) -> float:
    """Return the real z rotation of the object in degrees.

    CLEVR writes ``theta`` in degrees to ``rotation_euler[2]``, which uses
    radians. So the real rotation is ``theta`` radians.
    """
    return math.degrees(json_rotation) % 360.0
