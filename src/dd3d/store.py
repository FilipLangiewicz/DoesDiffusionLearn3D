"""Read pooled features from merged folders, chunk folders or tar files.

The extraction writes one chunk file for each task. On Kaggle the chunks of
one data set are in the outputs of six jobs (tar files) or in six data sets
(folders). The store finds all chunks under one root folder and joins them
when a script asks for one run, block and timestep. Thus a script needs no
merge step and no unpack step, and the disk keeps no second copy.

A key names one feature folder relative to the run folders, for example
``sana/block_10/t_400`` or ``sana/vae_latent``.
"""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_BLOCK = re.compile(r"block_\d+$")
_TIME = re.compile(r"t_\d+$")


@dataclass(frozen=True)
class _Entry:
    path: Path  # The file on disk, or the tar file.
    offset: int = -1  # The data offset in the tar file, or -1 for a file on disk.
    size: int = 0

    def read_bytes(self) -> bytes:
        with open(self.path, "rb") as f:
            if self.offset < 0:
                return f.read()
            f.seek(self.offset)
            return f.read(self.size)

    def read_array(self) -> np.ndarray:
        if self.offset < 0:
            return np.load(self.path)
        # A tar file keeps the .npy file unchanged at this offset.
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            return np.lib.format.read_array(f)


def _key_of(parts: list[str]) -> str | None:
    """Return the key of a feature file from its path parts, or None."""
    folder = parts[:-1]
    if len(folder) >= 3 and _TIME.match(folder[-1]) and _BLOCK.match(folder[-2]):
        return "/".join(folder[-3:])
    if len(folder) >= 2 and folder[-1] == "vae_latent":
        return "/".join(folder[-2:])
    return None


class FeatureStore:
    """All features under one root folder (folders, symbolic links and tar files)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        # Each key has a list of pairs (array entry, ids entry).
        self._chunks: dict[str, list[tuple[_Entry, _Entry]]] = {}
        self._meta: dict[str, _Entry] = {}
        self._stats: dict[str, list[_Entry]] = {}
        files: dict[str, _Entry] = {}
        for dirpath, _dirs, names in os.walk(self.root, followlinks=True):
            for name in names:
                p = Path(dirpath) / name
                if name.endswith(".tar"):
                    files.update(self._index_tar(p))
                else:
                    files[str(p)] = _Entry(p)
        self._add(files)

    @staticmethod
    def _index_tar(path: Path) -> dict[str, _Entry]:
        out = {}
        with tarfile.open(path) as tar:
            for m in tar:
                if m.isfile():
                    out[f"{path}::{m.name}"] = _Entry(path, m.offset_data, m.size)
        return out

    def _add(self, files: dict[str, _Entry]) -> None:
        for name, entry in files.items():
            parts = re.split(r"[/:]+", name)
            base = parts[-1]
            if base == "meta.json" and len(parts) >= 2:
                self._meta.setdefault(parts[-2], entry)
                continue
            if len(parts) >= 3 and parts[-2] == "stats" and base.endswith(".csv"):
                self._stats.setdefault(parts[-3], []).append(entry)
                continue
            key = _key_of(parts)
            if key is None:
                continue
            if base == "features.npy":
                ids = files.get(name[: -len("features.npy")] + "ids.txt")
            elif base.startswith("chunk_") and base.endswith(".npy"):
                ids = files.get(name[: -len(".npy")] + "_ids.txt")
            else:
                continue
            if ids is not None:
                self._chunks.setdefault(key, []).append((entry, ids))

    def keys(self) -> list[str]:
        return sorted(self._chunks)

    def runs(self) -> list[str]:
        return sorted(self._meta)

    def has(self, key: str) -> bool:
        return key in self._chunks

    def meta(self, run: str) -> dict:
        return json.loads(self._meta[run].read_bytes())

    def stats(self, run: str) -> pd.DataFrame:
        """Return the token norm statistics of all chunks of a run."""
        return pd.concat(pd.read_csv(io.BytesIO(e.read_bytes())) for e in self._stats[run])

    def ids(self, key: str) -> list[str]:
        out = []
        for _arr, ids in self._chunks[key]:
            out += ids.read_bytes().decode().split()
        return out

    def load(self, key: str, image_ids) -> np.ndarray:
        """Return the rows of ``image_ids`` in this order (float32).

        If an image is in two chunks (the same data attached twice), the first
        chunk counts.
        """
        wanted = {k: i for i, k in enumerate(image_ids)}
        out = None
        found = np.zeros(len(wanted), dtype=bool)
        for arr_entry, ids_entry in self._chunks[key]:
            ids = ids_entry.read_bytes().decode().split()
            rows = [(r, wanted[k]) for r, k in enumerate(ids) if k in wanted and not found[wanted[k]]]
            if not rows:
                continue
            arr = arr_entry.read_array()
            if out is None:
                out = np.empty((len(wanted), arr.shape[1]), dtype=np.float32)
            src, dst = (np.array(v) for v in zip(*rows, strict=True))
            out[dst] = arr[src]
            found[dst] = True
            del arr
        if out is None or not found.all():
            missing = [k for k, i in wanted.items() if not found[i]]
            raise KeyError(f"{len(missing)} images have no features for {key}, first: {missing[:1]}")
        return out

    def covers(self, key: str, image_ids) -> bool:
        return self.has(key) and set(image_ids) <= set(self.ids(key))
