"""Duplicate finder (F29): near-identical photos by perceptual hash, keeping the sharpest.

Each photo gets a 64-bit difference hash ("dHash"): its preview shrunk to 9x8 grey pixels, one bit
per horizontal neighbour pair (left brighter than right). Re-saved, resized, lightly edited or
re-compressed copies -- and frames of a burst -- land a few bits apart; different photos about 32.
Photos within `max_distance` bits of each other are grouped (transitively). In every group the
keeper is chosen among the copies at (nearly) the largest resolution -- a downsized copy scores
"sharper" per pixel but is the one to lose -- as the one with the best culling focus score
(app.culling); ties go to the larger file. The hashes are stored in the catalog (`PhotoCatalog.set_hash`) and only recomputed for
changed files, like the culling scores.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

HASH_SIZE = 8
DEFAULT_MAX_DISTANCE = 6  # of 64 bits
STRICTNESS = {"Exact copies": 2, "Near duplicates": 6, "Similar shots": 12}


def dhash(image: Image.Image) -> int:
    grey = np.asarray(image.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (grey[:, :-1] > grey[:, 1:]).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class DuplicateGroup:
    paths: list[str]
    keep: str
    distances: dict[str, int] = field(default_factory=dict)  # bits from the keeper

    @property
    def others(self) -> list[str]:
        return [p for p in self.paths if p != self.keep]


def _popcount64(values: np.ndarray) -> np.ndarray:
    values = values.copy()
    count = np.zeros(values.shape, dtype=np.uint8)
    for _ in range(64):
        count += (values & np.uint64(1)).astype(np.uint8)
        values >>= np.uint64(1)
    return count


def _pixels(entry: dict) -> int:
    if "pixels" not in entry:
        try:
            with Image.open(entry["path"]) as image:
                entry["pixels"] = image.width * image.height
        except (OSError, ValueError):
            entry["pixels"] = 0
    return entry["pixels"]


def find_duplicate_groups(entries: Iterable[dict], max_distance: int = DEFAULT_MAX_DISTANCE) -> list[DuplicateGroup]:
    """`entries`: {"path", "hash" (int), "focus" (float or None)}. Groups of 2+, largest first."""
    entries = [e for e in entries if e.get("hash") is not None]
    if len(entries) < 2:
        return []
    hashes = np.array([e["hash"] for e in entries], dtype=np.uint64)
    parent = list(range(len(entries)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(entries) - 1):
        close = np.nonzero(_popcount64(hashes[i + 1:] ^ hashes[i]) <= max_distance)[0]
        for j in close + i + 1:
            parent[root(int(j))] = root(i)
    members: dict[int, list[int]] = {}
    for i in range(len(entries)):
        members.setdefault(root(i), []).append(i)
    groups = []
    for indices in members.values():
        if len(indices) < 2:
            continue

        largest = max(_pixels(entries[i]) for i in indices)

        def rank(i: int):
            focus = entries[i].get("focus")
            try:
                size = os.path.getsize(entries[i]["path"])
            except OSError:
                size = 0
            return (_pixels(entries[i]) >= 0.9 * largest, focus if focus is not None else -1.0, size)

        keeper = max(indices, key=rank)
        keep_hash = entries[keeper]["hash"]
        paths = [entries[i]["path"] for i in sorted(indices, key=rank, reverse=True)]
        groups.append(DuplicateGroup(paths, entries[keeper]["path"],
                                     {entries[i]["path"]: hamming(entries[i]["hash"], keep_hash) for i in indices}))
    return sorted(groups, key=lambda g: (-len(g.paths), g.keep))


def catalog_hashes(catalog, paths: Iterable[str | Path], preview: Callable[[Path], Image.Image] | None = None,
                   progress=None) -> list[dict]:
    """The hash of each photo, from the catalog when the file hasn't changed, else computed (and
    stored). Unreadable files are left out."""
    if preview is None:
        from app.culling import preview_for_scoring as preview
    paths = [Path(p) for p in paths]
    out = []
    for index, path in enumerate(paths, 1):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        stored, stored_mtime = catalog.get_hash(path)
        if stored is None or stored_mtime != mtime:
            try:
                stored = dhash(preview(path))
            except Exception:  # noqa: BLE001 -- unreadable preview: skipped, like an unscored photo
                stored = None
            if stored is not None:
                catalog.set_hash(path, stored, mtime)
        if stored is not None:
            out.append({"path": str(path), "hash": stored, "focus": catalog.get(path)["focus"]})
        if progress:
            progress(index, len(paths))
    return out
