"""Pixel-space cue tests: which part of the image carries the light direction.

All functions work on the RGB renders of the dataset. The object is purple, so
a colour threshold gives its mask. The floor is grey.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
)

_SQUARE = np.ones((3, 3), dtype=bool)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a mask by ``radius`` pixels (square neighbourhood)."""
    if radius <= 0:
        return mask.copy()
    return binary_dilation(mask, structure=_SQUARE, iterations=radius)


def object_mask(rgb: np.ndarray, margin: int = 18) -> np.ndarray:
    """Return True for the object pixels (red and blue above green), with holes filled.

    The hole fill adds the white specular highlight, which shows the light direction.
    """
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    mask = ((r - g) > margin) & ((b - g) > margin)
    mask = binary_opening(mask, iterations=1)
    mask = binary_fill_holes(binary_closing(mask, iterations=3))
    # A highlight on the object rim is not inside the mask, so the hole fill misses it.
    shine = (np.minimum(np.minimum(r, g), b) > 170) & ((r - g) > 6) & ((b - g) > 6)
    return mask | (shine & dilate(mask, 8))


def floor_only(rgb: np.ndarray, mask: np.ndarray | None = None, radius: int = 6) -> np.ndarray:
    """Replace the object (and a thin ring) with the median colour of the ring around it.

    The cast shadow and the floor highlight stay. The light and dark sides of the object go away.
    """
    mask = object_mask(rgb) if mask is None else mask
    inner = dilate(mask, radius)
    ring = dilate(inner, 10) & ~inner
    out = rgb.copy()
    fill = np.median(rgb[ring], axis=0) if ring.any() else np.median(rgb.reshape(-1, 3), axis=0)
    out[inner] = fill.astype(rgb.dtype)
    return out


def object_crop(rgb: np.ndarray, mask: np.ndarray | None = None, size: int = 48, pad: int = 4) -> np.ndarray:
    """Return a square crop around the object, with the floor set to grey.

    The crop keeps the light and dark sides of the object, but not its cast shadow or the floor.
    """
    mask = object_mask(rgb) if mask is None else mask
    if not mask.any():
        return np.full((size, size, 3), 128, dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgb.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgb.shape[1])
    crop = rgb[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = 128
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


def highlight_position(
    rgb: np.ndarray, mask: np.ndarray | None = None, top_fraction: float = 0.02
) -> tuple[float, float]:
    """Return the image position (x, y in [0, 1]) of the brightest floor region.

    The function blurs the luminance, ignores the object and takes the centroid of
    the brightest pixels. Two numbers per image are enough to test the shortcut.
    """
    mask = object_mask(rgb) if mask is None else mask
    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum[dilate(mask, 6)] = 0.0
    lum = gaussian_filter(lum, sigma=8)
    thr = np.quantile(lum, 1.0 - top_fraction)
    ys, xs = np.nonzero(lum >= thr)
    return float(xs.mean() / rgb.shape[1]), float(ys.mean() / rgb.shape[0])


def load_rgb(images_dir: Path, shape: str, image_id: str) -> np.ndarray:
    return np.asarray(Image.open(Path(images_dir) / shape / f"{image_id}.png").convert("RGB"))


def cue_features(df: pd.DataFrame, images_dir: Path, resolution: int = 64) -> dict[str, np.ndarray]:
    """Return the feature matrices of all cue variants for the rows of ``df``."""
    full, floor, obj, hl = [], [], [], []
    small = (resolution, resolution)
    for image_id, shape in zip(df["image_id"], df["shape"], strict=True):
        rgb = load_rgb(images_dir, shape, image_id)
        mask = object_mask(rgb)
        full.append(np.asarray(Image.fromarray(rgb).resize(small, Image.BOX), dtype=np.float32).ravel())
        floor_img = Image.fromarray(floor_only(rgb, mask)).resize(small, Image.BOX)
        floor.append(np.asarray(floor_img, dtype=np.float32).ravel())
        obj.append(object_crop(rgb, mask).astype(np.float32).ravel())
        x, y = highlight_position(rgb, mask)
        hl.append([x, y, x * x, y * y, x * y])
    return {
        "full image": np.stack(full) / 255.0,
        "floor only (object removed)": np.stack(floor) / 255.0,
        "object only (crop)": np.stack(obj) / 255.0,
        "floor highlight position (2 numbers)": np.asarray(hl, dtype=np.float32),
    }
