"""Metrics for the light direction probes."""

from __future__ import annotations

import numpy as np


def encode_targets(az_deg, el_deg) -> np.ndarray:
    """Return the regression targets ``[sin az, cos az, el]``.

    The sine and the cosine remove the jump of the azimuth at 0 and 360 degrees.
    """
    az = np.radians(np.asarray(az_deg, dtype=float))
    return np.stack([np.sin(az), np.cos(az), np.asarray(el_deg, dtype=float)], axis=-1)


def decode_azimuth(sin_pred, cos_pred) -> np.ndarray:
    """Return the azimuth in [0, 360) for predicted sine and cosine values."""
    return np.degrees(np.arctan2(sin_pred, cos_pred)) % 360.0


def circular_abs_error(pred_deg, true_deg) -> np.ndarray:
    """Return the absolute angle difference in degrees, in [0, 180]."""
    d = np.abs(np.asarray(pred_deg, dtype=float) - np.asarray(true_deg, dtype=float)) % 360.0
    return np.minimum(d, 360.0 - d)


def errors_from_targets(pred: np.ndarray, az_true, el_true) -> tuple[np.ndarray, np.ndarray]:
    """Return the azimuth and elevation errors for predictions ``[sin, cos, el]``."""
    pred = np.asarray(pred, dtype=float)
    az_err = circular_abs_error(decode_azimuth(pred[:, 0], pred[:, 1]), az_true)
    el_err = np.abs(pred[:, 2] - np.asarray(el_true, dtype=float))
    return az_err, el_err


def bootstrap_ci(values, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Return a percentile bootstrap interval for the mean of ``values``."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(err_a, err_b, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple:
    """Return the mean of ``err_a - err_b`` and its bootstrap interval (same samples)."""
    d = np.asarray(err_a, dtype=float) - np.asarray(err_b, dtype=float)
    lo, hi = bootstrap_ci(d, n_boot=n_boot, level=level, seed=seed)
    return float(d.mean()), lo, hi


def random_baseline_azimuth_mae() -> float:
    """Return the expected azimuth MAE of a uniform random guess (90 degrees)."""
    return 90.0


def summary(az_err, el_err) -> dict:
    """Return the mean, median and 90th percentile of both errors."""
    az_err, el_err = np.asarray(az_err), np.asarray(el_err)
    return {
        "az_mae": float(az_err.mean()),
        "az_median": float(np.median(az_err)),
        "az_p90": float(np.quantile(az_err, 0.9)),
        "el_mae": float(el_err.mean()),
        "el_median": float(np.median(el_err)),
        "el_p90": float(np.quantile(el_err, 0.9)),
        "n": int(len(az_err)),
    }
