"""Linear probes for the light direction: PCA and Ridge.

Changes against the ZZSN probe:

- The alpha grid follows the PCA eigenvalues. The ZZSN grid (0.001 to 1000) was
  far below the eigenvalues, so the alpha had no effect.
- The alpha comes from the val split only.
- An option gives the PCA scores unit variance before Ridge (``whiten``).
- The result has bootstrap intervals and a paired comparison against a control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dd3d.metrics import bootstrap_ci, encode_targets, errors_from_targets, summary


@dataclass
class ProbeConfig:
    n_components: int = 256
    whiten: bool = False
    # Without whiten, the grid is relative to n_train * PCA variance (see alpha_grid).
    relative_alphas: list[float] = field(default_factory=lambda: [float(a) for a in np.logspace(-5, 1, 19)])
    seed: int = 0


def alpha_grid(z_train: np.ndarray, cfg: ProbeConfig) -> list[float]:
    """Return absolute alpha values for the Ridge on the PCA scores.

    Ridge shrinks component i by the factor lambda_i / (lambda_i + alpha), where
    lambda_i is the sum of squares of the scores. So the useful alpha values are
    near these sums. With whitened scores, lambda_i equals the number of samples.
    """
    scale = float(np.median((z_train**2).sum(axis=0)))
    return [scale * a for a in cfg.relative_alphas]


@dataclass
class ProbeResult:
    alpha: float
    val_curve: list[dict]
    splits: dict[str, dict]
    errors: dict[str, tuple[np.ndarray, np.ndarray]]
    model: Pipeline


def fit_probe(
    x_train: np.ndarray,
    az_train,
    el_train,
    x_val: np.ndarray,
    az_val,
    el_val,
    cfg: ProbeConfig | None = None,
) -> tuple[Pipeline, float, list[dict]]:
    """Fit the probe, select alpha on the val split and refit on the train split.

    The criterion is the sum of the azimuth MAE and the elevation MAE in degrees.
    The PCA is fitted once; only the Ridge part changes with alpha.
    """
    cfg = cfg or ProbeConfig()
    n_comp = min(cfg.n_components, x_train.shape[0] - 1, x_train.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=cfg.seed).fit(x_train)
    scaler = StandardScaler(with_std=cfg.whiten).fit(pca.transform(x_train))
    z_train = scaler.transform(pca.transform(x_train))
    z_val = scaler.transform(pca.transform(x_val))
    y_train = encode_targets(az_train, el_train)
    curve = []
    for alpha in alpha_grid(z_train, cfg):
        ridge = Ridge(alpha=alpha).fit(z_train, y_train)
        az_err, el_err = errors_from_targets(ridge.predict(z_val), az_val, el_val)
        curve.append({"alpha": alpha, "val_az_mae": float(az_err.mean()), "val_el_mae": float(el_err.mean())})
    best = min(curve, key=lambda r: r["val_az_mae"] + r["val_el_mae"])["alpha"]
    model = Pipeline([("pca", pca), ("scale", scaler), ("ridge", Ridge(alpha=best).fit(z_train, y_train))])
    return model, best, curve


def evaluate(model: Pipeline, x: np.ndarray, az, el) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Return the summary with 95 percent intervals, the errors and the predictions."""
    pred = model.predict(x)
    az_err, el_err = errors_from_targets(pred, az, el)
    out = summary(az_err, el_err)
    out["az_mae_ci"] = bootstrap_ci(az_err)
    out["el_mae_ci"] = bootstrap_ci(el_err)
    return out, az_err, el_err, pred


def run(
    features: dict[str, np.ndarray], labels: dict[str, tuple], cfg: ProbeConfig | None = None
) -> ProbeResult:
    """Fit on ``train``, select on ``val`` and evaluate on every other split.

    ``features`` and ``labels`` map a split name to the feature matrix and to the
    tuple (azimuth, elevation) in degrees.
    """
    model, alpha, curve = fit_probe(
        features["train"], *labels["train"], features["val"], *labels["val"], cfg=cfg
    )
    splits, errors = {}, {}
    for name, x in features.items():
        res, az_err, el_err, _ = evaluate(model, x, *labels[name])
        splits[name], errors[name] = res, (az_err, el_err)
    return ProbeResult(alpha=alpha, val_curve=curve, splits=splits, errors=errors, model=model)


def probe_direction(model: Pipeline) -> np.ndarray:
    """Return the probe weights in the input feature space, shape (3, n_features).

    The probe is linear: prediction = W x + b with W = R S^-1 P. Here P is the PCA
    basis, S is the score scale and R holds the Ridge weights. The rows of W are
    concept vectors for sin(az), cos(az) and el.
    """
    pca, scaler, ridge = model.named_steps["pca"], model.named_steps["scale"], model.named_steps["ridge"]
    scale = scaler.scale_ if scaler.scale_ is not None else 1.0
    return (ridge.coef_ / scale) @ pca.components_


def spatial_contributions(model: Pipeline, x: np.ndarray, grid: int) -> np.ndarray:
    """Split the linear prediction of each sample into grid cells.

    ``x`` has rows of ``grid * grid * C`` features in (row, column, channel) order.
    Return shape (n, 3, grid, grid). The sum over the cells plus the offset gives
    the prediction exactly. Thus the map is an exact attribution of the linear probe.
    """
    w = probe_direction(model)
    mean = model.named_steps["pca"].mean_
    centered = (np.asarray(x, dtype=np.float64) - mean).reshape(len(x), grid, grid, -1)
    return np.einsum("nhwc,khwc->nkhw", centered, w.reshape(3, grid, grid, -1))
