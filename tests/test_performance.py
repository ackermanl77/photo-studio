from PIL import Image

from app.performance import PreviewCache, recipe_hash


def test_recipe_hash_is_stable_for_dict_order():
    assert recipe_hash({"contrast": 1.2, "exposure": 1.1}) == recipe_hash({"exposure": 1.1, "contrast": 1.2})


def test_preview_cache_round_trip_and_trim(tmp_path):
    cache = PreviewCache(tmp_path / "cache", max_items=1)
    image = Image.new("RGB", (100, 60), (90, 100, 110))
    first = cache.key(b"one", {"exposure": 1.1})
    second = cache.key(b"two", {"exposure": 1.1})

    cache.put(first, image)
    assert cache.get(first).size == (100, 60)
    cache.put(second, image)
    assert cache.get(second) is not None
    assert cache.get(first) is None

    cache.clear()
    assert cache.get(second) is None
