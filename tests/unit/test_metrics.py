import numpy as np
import pytest

from dd3d.metrics import (
    bootstrap_ci,
    circular_abs_error,
    decode_azimuth,
    encode_targets,
    errors_from_targets,
    paired_bootstrap_diff,
    summary,
)


def test_circular_abs_error_wraps():
    assert circular_abs_error(1, 359) == pytest.approx(2)
    assert circular_abs_error(0, 180) == pytest.approx(180)


def test_encode_decode_azimuth():
    az = np.array([0.0, 10.0, 90.0, 180.0, 270.0, 359.0])
    targets = encode_targets(az, np.zeros_like(az))
    np.testing.assert_allclose(decode_azimuth(targets[:, 0], targets[:, 1]), az, atol=1e-9)


def test_errors_from_perfect_predictions_are_zero():
    rng = np.random.default_rng(1)
    az = rng.uniform(0, 360, 50)
    el = rng.uniform(-80, 80, 50)
    az_err, el_err = errors_from_targets(encode_targets(az, el), az, el)
    np.testing.assert_allclose(az_err, 0.0, atol=1e-6)
    np.testing.assert_allclose(el_err, 0.0, atol=1e-12)


def test_bootstrap_ci_contains_mean():
    values = np.random.default_rng(2).normal(3.0, 2.0, 500)
    lo, hi = bootstrap_ci(values)
    assert lo < values.mean() < hi
    assert hi - lo < 1.0


def test_paired_bootstrap_of_identical_arrays():
    x = np.random.default_rng(3).normal(size=100)
    assert paired_bootstrap_diff(x, x) == (0.0, 0.0, 0.0)


def test_summary_keys_and_count():
    result = summary(np.arange(10.0), np.arange(10.0))
    assert result["n"] == 10
    assert result["az_mae"] == pytest.approx(4.5)
    assert result["el_median"] == pytest.approx(4.5)
