import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dd3d.cues import crisp_shadow_mask, crop_features, object_mask, shadow_azimuth, shadow_kept
from dd3d.geometry import project
from dd3d.metrics import circular_abs_error

PURPLE = (150, 60, 170)


def scene(light_az: float, size: int = 512) -> np.ndarray:
    """Return a grey floor, a purple disc at the scene centre and its dark cast shadow.

    The shadow lies on the ground, on the side away from the light, as the camera sees it.
    """
    img = np.full((size, size, 3), 170, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    away = np.radians(light_az + 180.0)
    for r in np.linspace(0.3, 1.6, 30):
        px, py, _ = project(np.array([r * np.cos(away), r * np.sin(away), 0.0]))
        img[(xx - px) ** 2 + (yy - py) ** 2 < 14**2] = 60
    cx, cy, _ = project(np.zeros(3))
    img[(xx - cx) ** 2 + (yy - cy) ** 2 < 30**2] = PURPLE
    return img


@pytest.mark.parametrize("az", [20.0, 110.0, 200.0, 290.0])
def test_shadow_azimuth_reads_the_light_of_a_synthetic_shadow(az):
    assert circular_abs_error(shadow_azimuth(scene(az)), az) < 10.0


def test_shadow_azimuth_is_nan_without_a_shadow():
    img = np.full((512, 512, 3), 170, dtype=np.uint8)
    assert np.isnan(shadow_azimuth(img))


def test_shadow_kept_is_one_for_the_same_image_and_small_after_a_turn():
    a, b = scene(30.0), scene(120.0)
    assert shadow_kept(a, a) == 1.0
    assert shadow_kept(a, b) < 0.3


def test_crisp_shadow_mask_does_not_include_the_object():
    img = scene(45.0)
    assert not (crisp_shadow_mask(img) & object_mask(img)).any()


def test_crop_features_have_a_grey_floor(tmp_path):
    (tmp_path / "Sphere").mkdir()
    Image.fromarray(scene(45.0)).save(tmp_path / "Sphere" / "a.png")
    x = crop_features(pd.DataFrame({"image_id": ["a"], "shape": ["Sphere"]}), tmp_path)
    assert x.shape == (1, 48 * 48 * 3)
    corner = x.reshape(48, 48, 3)[0, 0]
    assert np.allclose(corner, 128 / 255, atol=0.02)
