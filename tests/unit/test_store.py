"""Tests of dd3d.store: chunk folders, merged folders and tar files give the same rows."""

import json
import tarfile

import numpy as np
import pytest

from dd3d.store import FeatureStore


def write_task(root, task, ids, rng):
    run = root / f"task{task}" / "sana"
    d = run / "block_10" / "t_400"
    d.mkdir(parents=True)
    arr = rng.normal(size=(len(ids), 6)).astype(np.float16)
    np.save(d / f"chunk_{task:04d}.npy", arr)
    (d / f"chunk_{task:04d}_ids.txt").write_text("\n".join(ids) + "\n")
    (run / "meta.json").write_text(json.dumps({"blocks": [10], "timesteps": [400]}))
    (run / "stats").mkdir()
    (run / "stats" / f"chunk_{task:04d}.csv").write_text("image_id,block,t,mean_token_norm\na,10,400,2.0\n")
    return dict(zip(ids, arr.astype(np.float32), strict=True))


@pytest.fixture
def chunks(tmp_path):
    rng = np.random.default_rng(0)
    rows = write_task(tmp_path / "src", 0, ["a", "b", "c"], rng)
    rows |= write_task(tmp_path / "src", 1, ["d", "e"], rng)
    return tmp_path, rows


def test_chunk_folders(chunks):
    tmp, rows = chunks
    store = FeatureStore(tmp / "src")
    assert store.keys() == ["sana/block_10/t_400"]
    assert store.runs() == ["sana"]
    got = store.load("sana/block_10/t_400", ["e", "a", "c"])
    np.testing.assert_array_equal(got, np.stack([rows["e"], rows["a"], rows["c"]]))
    assert store.covers("sana/block_10/t_400", ["a", "e"])
    assert not store.covers("sana/block_10/t_400", ["a", "z"])
    assert len(store.stats("sana")) == 2


def test_tar_files_give_the_same_rows(chunks):
    tmp, rows = chunks
    (tmp / "tars").mkdir()
    for task in (0, 1):
        with tarfile.open(tmp / "tars" / f"x_task{task:02d}.tar", "w") as tar:
            tar.add(tmp / "src" / f"task{task}", arcname=".")
    store = FeatureStore(tmp / "tars")
    got = store.load("sana/block_10/t_400", ["b", "d"])
    np.testing.assert_array_equal(got, np.stack([rows["b"], rows["d"]]))
    assert store.meta("sana")["blocks"] == [10]


def test_merged_folder_and_missing_ids(tmp_path):
    d = tmp_path / "sana" / "block_04" / "t_200"
    d.mkdir(parents=True)
    np.save(d / "features.npy", np.arange(6, dtype=np.float16).reshape(3, 2))
    (d / "ids.txt").write_text("x\ny\nz\n")
    store = FeatureStore(tmp_path)
    np.testing.assert_array_equal(store.load("sana/block_04/t_200", ["z"]), [[4.0, 5.0]])
    with pytest.raises(KeyError):
        store.load("sana/block_04/t_200", ["x", "q"])
