"""Load the splits, the scene JSON files and the feature arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dd3d.geometry import local_light_direction

SPLITS = ("train", "val", "test", "test_cubes")


def load_splits(splits_dir: Path) -> dict[str, pd.DataFrame]:
    """Return the four split tables of data/splits."""
    return {name: pd.read_csv(Path(splits_dir) / f"{name}.csv") for name in SPLITS}


def load_scene(scenes_dir: Path, shape: str, image_id: str) -> dict:
    return json.loads((Path(scenes_dir) / shape / f"{image_id}.json").read_text())


def add_scene_columns(df: pd.DataFrame, scenes_dir: Path) -> pd.DataFrame:
    """Add the object position and the light direction at the object (local label)."""
    rows = []
    for image_id, shape in zip(df["image_id"], df["shape"], strict=True):
        scene = load_scene(scenes_dir, shape, image_id)
        x, y, z = scene["objects"][0]["3d_coords"]
        loc_az, loc_el = local_light_direction(
            (x, y, z), scene["light_azimuth_deg"], scene["light_elevation_deg"]
        )
        rows.append(
            {"obj_x": x, "obj_y": y, "obj_z": z, "local_az": float(loc_az), "local_el": float(loc_el)}
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def target_columns(target: str) -> tuple[str, str]:
    """Return the azimuth and elevation columns of a target name."""
    if target == "global":
        return "light_azimuth_deg", "light_elevation_deg"
    if target == "local":
        return "local_az", "local_el"
    raise ValueError(f"unknown target {target!r}; use 'global' or 'local'")


def load_feature_dir(feature_dir: Path, image_ids) -> np.ndarray:
    """Return the rows of ``features.npy`` in the order of ``image_ids`` (float32)."""
    feature_dir = Path(feature_dir)
    ids = (feature_dir / "ids.txt").read_text().split()
    index = {k: i for i, k in enumerate(ids)}
    missing = [k for k in image_ids if k not in index]
    if missing:
        raise KeyError(f"{len(missing)} images have no features in {feature_dir}, first: {missing[0]}")
    arr = np.load(feature_dir / "features.npy", mmap_mode="r")
    return np.asarray(arr[[index[k] for k in image_ids]], dtype=np.float32)


def pool_grid(x: np.ndarray, grid: int, to: int) -> np.ndarray:
    """Average a (n, grid * grid * C) feature matrix to a smaller grid."""
    if to == grid:
        return x
    n, k = len(x), grid // to
    c = x.shape[1] // (grid * grid)
    return x.reshape(n, to, k, to, k, c).mean(axis=(2, 4)).reshape(n, -1)


def pixel_features(df: pd.DataFrame, images_dir: Path, resolution: int = 64) -> np.ndarray:
    """Return downscaled RGB pixels in [0, 1] as one row for each image."""
    rows = []
    for image_id, shape in zip(df["image_id"], df["shape"], strict=True):
        img = Image.open(Path(images_dir) / shape / f"{image_id}.png").convert("RGB")
        img = img.resize((resolution, resolution), Image.BOX)
        rows.append(np.asarray(img, dtype=np.float32).ravel() / 255.0)
    return np.stack(rows)


def legacy_activation_features(
    activations_root: Path, df: pd.DataFrame, layer: str, timestep: str
) -> np.ndarray:
    """Load ZZSN activation files (one .npy for each image) with the 4x4 average pool."""
    layer_num, t_num = layer.split("_")[1], timestep.split("_")[1]
    rows = []
    for image_id, shape in zip(df["image_id"], df["shape"], strict=True):
        path = Path(activations_root) / layer / timestep / shape / f"{image_id}_l{layer_num}_t{t_num}.npy"
        arr = np.load(path).squeeze(0).reshape(32, 32, -1)
        rows.append(arr.reshape(8, 4, 8, 4, arr.shape[-1]).mean(axis=(1, 3)).ravel())
    return np.stack(rows).astype(np.float32)
