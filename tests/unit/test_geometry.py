import json
import math

import numpy as np
import pytest

from dd3d.geometry import (
    BASE_CAMERA,
    angle_diff_deg,
    azimuth_to_image_angle,
    direction_to_az_el,
    fit_camera_jitter,
    lamp_position,
    local_light_direction,
    object_in_frame,
    project,
    real_rotation_deg,
    shadow_point,
    sun_position_az_el,
)


def scene_files(repo_root):
    scenes = repo_root / "data" / "final_example" / "scenes"
    # CLEVR_scenes.json is a merged file and has another layout.
    return sorted(p for p in scenes.glob("*/*.json") if p.name != "CLEVR_scenes.json")


def test_lamp_position_axes():
    np.testing.assert_allclose(lamp_position(0, 0), [5, 0, 0], atol=1e-9)
    np.testing.assert_allclose(lamp_position(90, 90), [0, 0, 5], atol=1e-9)


def test_lamp_position_vectorized():
    pos = lamp_position(np.linspace(0, 300, 7), np.linspace(-60, 60, 7))
    assert pos.shape == (7, 3)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=-1), 5.0)


def test_direction_roundtrip():
    rng = np.random.default_rng(0)
    az = rng.uniform(0, 360, 200)
    el = rng.uniform(-80, 80, 200)
    az_back, el_back = direction_to_az_el(lamp_position(az, el))
    np.testing.assert_allclose(az_back, az, atol=1e-7)
    np.testing.assert_allclose(el_back, el, atol=1e-7)


def test_local_light_direction_at_origin():
    az, el = local_light_direction((0, 0, 0), 123.0, 45.0)
    assert az == pytest.approx(123.0)
    assert el == pytest.approx(45.0)


def test_project_origin_is_image_centre():
    px = project(np.zeros(3))
    np.testing.assert_allclose(px[:2], [256, 256], atol=1e-9)


def test_camera_jitter_fit_on_all_scenes(repo_root):
    files = scene_files(repo_root)
    assert len(files) == 30
    for path in files:
        obj = json.loads(path.read_text())["objects"][0]
        jitter, residual = fit_camera_jitter(obj["3d_coords"], obj["pixel_coords"])
        assert np.all(np.abs(jitter) <= 0.5 + 1e-9), path.name
        assert residual <= 0.6, path.name
        px = project(obj["3d_coords"], BASE_CAMERA + jitter)
        np.testing.assert_allclose(px[:2], obj["pixel_coords"][:2], atol=0.6, err_msg=path.name)
        assert px[2] == pytest.approx(obj["pixel_coords"][2], abs=0.002), path.name


def test_azimuth_to_image_angle_values():
    angles = azimuth_to_image_angle([0, 90, 180, 270])
    np.testing.assert_allclose(angles, [331.4, 22.4, 151.4, 202.4], atol=0.5)


def test_image_angle_turns_counter_clockwise():
    angles = azimuth_to_image_angle(np.array([0.0, 90.0, 180.0, 270.0]))
    # Each step of 90 degrees adds about 90 degrees modulo 360.
    steps = np.diff(np.r_[angles, angles[0]]) % 360.0
    assert np.all((steps > 45) & (steps < 135))


def test_shadow_point_on_ground_and_on_line():
    lamp = lamp_position(40.0, 55.0)
    p = np.array([1.0, -2.0, 1.4])
    s = shadow_point(p, lamp)
    assert s[2] == pytest.approx(0.0, abs=1e-12)
    cross = np.cross(p - lamp, s - lamp)
    np.testing.assert_allclose(cross, 0.0, atol=1e-9)
    # The shadow is behind the point as seen from the lamp.
    assert np.dot(p - lamp, s - lamp) > 0


def test_angle_diff():
    assert angle_diff_deg(1, 359) == pytest.approx(2)
    assert angle_diff_deg(0, 180) == pytest.approx(180)
    assert angle_diff_deg(10, 10) == pytest.approx(0)


def test_real_rotation_deg():
    assert real_rotation_deg(math.pi) == pytest.approx(180.0, abs=1e-9)


def test_sun_position():
    az, el = sun_position_az_el()
    assert az == pytest.approx(334.85, abs=0.05)
    assert el == pytest.approx(62.75, abs=0.05)


def test_object_in_frame():
    # The centre is inside. The renders at (2, 2) and (-2, -2) cut a large cube at the image edge.
    assert object_in_frame(0.0, 0.0)
    assert not object_in_frame(2.0, 2.0)
    assert not object_in_frame(-2.0, -2.0)
    # A small object fits where a large one does not.
    assert object_in_frame(2.0, 2.0, size=0.3)
