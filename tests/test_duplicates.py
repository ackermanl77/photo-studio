"""Duplicate finder (F29): dHash, grouping, the sharpest kept, catalog-cached hashes."""
import numpy as np
from PIL import Image, ImageFilter

from app.duplicates import DEFAULT_MAX_DISTANCE, catalog_hashes, dhash, find_duplicate_groups, hamming
from app.library import index_folder
from app.major_features import PhotoCatalog
from tests.synthetic_assets import photo_like


def _other_scene(seed: int) -> Image.Image:
    """A genuinely different picture: random coloured blocks (photo_like is one scene with noise)."""
    blocks = np.random.default_rng(seed).integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(blocks).resize((640, 480), Image.Resampling.NEAREST)


def _library(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    original = photo_like((640, 480), seed=3)
    original.save(folder / "a_original.jpg", quality=95)
    original.resize((320, 240)).save(folder / "b_small_copy.jpg", quality=70)
    original.filter(ImageFilter.GaussianBlur(3)).save(folder / "c_blurred.jpg", quality=92)
    _other_scene(11).save(folder / "d_other.jpg", quality=92)
    photo_like((640, 480), seed=29).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(folder / "e_another.png")
    return folder


def test_hash_distances_separate_copies_from_other_photos():
    original = photo_like((640, 480), seed=3)
    copy = original.resize((300, 225))
    other = _other_scene(11)
    assert hamming(dhash(original), dhash(copy)) <= 4
    assert hamming(dhash(original), dhash(other)) > 16


def test_groups_keep_the_sharpest_and_cache_the_hashes(tmp_path):
    folder = _library(tmp_path)
    database = tmp_path / "catalog.sqlite"
    index_folder(folder, database)
    catalog = PhotoCatalog(database)
    paths = sorted(str(p) for p in folder.iterdir())
    entries = catalog_hashes(catalog, paths)
    assert len(entries) == 5
    groups = find_duplicate_groups(entries, DEFAULT_MAX_DISTANCE)
    assert len(groups) == 1
    group = groups[0]
    names = sorted(p.split("\\")[-1].split("/")[-1] for p in group.paths)
    assert names == ["a_original.jpg", "b_small_copy.jpg", "c_blurred.jpg"]
    assert group.keep.endswith("a_original.jpg")  # the blurred copy scores lower on focus
    assert group.distances[group.keep] == 0
    # Cached: a second pass reads the catalog instead of decoding again.
    calls = []
    catalog_hashes(catalog, paths, preview=lambda path: calls.append(path))
    assert calls == []
    assert catalog.get_hash(paths[0])[0] == entries[0]["hash"]


def test_nothing_to_group():
    assert find_duplicate_groups([]) == []
    assert find_duplicate_groups([{"path": "x", "hash": 0, "focus": 1.0}, {"path": "y", "hash": 2 ** 64 - 1, "focus": 1.0}]) == []
