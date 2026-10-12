import numpy as np
import pandas as pd
import pytest

SPLITS = ("train", "val", "test", "test_cubes")
ROWS = {"train": 1600, "val": 200, "test": 200, "test_cubes": 1000}


@pytest.fixture(scope="module")
def splits(repo_root):
    return {name: pd.read_csv(repo_root / "data" / "splits" / f"{name}.csv") for name in SPLITS}


@pytest.fixture(scope="module")
def master(repo_root):
    return pd.read_csv(repo_root / "data" / "master_labels.csv")


def test_split_sizes(splits):
    for name, rows in ROWS.items():
        assert len(splits[name]) == rows, name


def test_splits_do_not_overlap(splits):
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            common = set(splits[a]["image_id"]) & set(splits[b]["image_id"])
            assert not common, (a, b)


def test_image_ids_are_unique_in_each_split(splits):
    for name, df in splits.items():
        assert df["image_id"].is_unique, name


def test_cube_shape_only_in_cube_split(splits):
    assert set(splits["test_cubes"]["shape"]) == {"SmoothCube_v2"}
    for name in ("train", "val", "test"):
        assert "SmoothCube_v2" not in set(splits[name]["shape"]), name


def test_split_labels_match_master(splits, master):
    master = master.set_index("image_id")
    for name, df in splits.items():
        ref = master.loc[df["image_id"]]
        for col in ("light_azimuth_deg", "light_elevation_deg"):
            np.testing.assert_array_equal(df[col].to_numpy(), ref[col].to_numpy(), err_msg=name)


def test_sin_cos_match_azimuth(splits, master):
    for name, df in {**splits, "master": master}.items():
        az = np.radians(df["light_azimuth_deg"].to_numpy())
        np.testing.assert_allclose(df["sin_azimuth"], np.sin(az), atol=1e-6, err_msg=name)
        np.testing.assert_allclose(df["cos_azimuth"], np.cos(az), atol=1e-6, err_msg=name)
