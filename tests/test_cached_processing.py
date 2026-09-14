from PIL import Image

from app.cached_processing import cached_preview


def test_cached_preview_reuses_same_recipe(tmp_path):
    calls = []

    def processor():
        calls.append("processed")
        return Image.new("RGB", (40, 20), (100, 110, 120))

    first, first_report = cached_preview(b"source", "photo.jpg", {"exposure": 1.1}, processor, tmp_path)
    second, second_report = cached_preview(b"source", "photo.jpg", {"exposure": 1.1}, processor, tmp_path)

    assert first.size == second.size
    assert calls == ["processed"]
    assert first_report.cache_hit is False
    assert second_report.cache_hit is True


def test_cached_preview_invalidates_when_recipe_changes(tmp_path):
    calls = []

    def processor():
        calls.append("processed")
        return Image.new("RGB", (40, 20), (100, 110, 120))

    cached_preview(b"source", "photo.jpg", {"exposure": 1.0}, processor, tmp_path)
    cached_preview(b"source", "photo.jpg", {"exposure": 1.2}, processor, tmp_path)

    assert calls == ["processed", "processed"]
