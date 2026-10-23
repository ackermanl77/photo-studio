"""Recent files (app.recent_files) and the settings file (app.settings)."""
from __future__ import annotations

from PIL import Image

from app import recent_files, settings


def _image(color=(200, 60, 40)):
    return Image.new("RGB", (64, 48), color)


def test_an_uploaded_photo_is_kept_as_a_copy_with_a_thumbnail():
    entry_id = recent_files.remember("beach.jpg", b"photo-bytes", "photo", _image())
    [entry] = recent_files.load()
    assert entry["id"] == entry_id and entry["name"] == "beach.jpg" and entry["kind"] == "photo"
    assert recent_files.read_bytes(entry) == b"photo-bytes"
    assert recent_files.thumbnail_path(entry).is_file()


def test_reopening_the_same_file_moves_it_to_the_top_instead_of_listing_it_twice():
    first = recent_files.remember("a.jpg", b"aaa", "photo", _image())
    recent_files.remember("b.jpg", b"bbb", "photo", _image())
    again = recent_files.remember("a.jpg", b"aaa", "photo", _image())
    assert again == first
    assert [entry["name"] for entry in recent_files.load()] == ["a.jpg", "b.jpg"]


def test_a_file_opened_from_a_path_is_remembered_by_it_not_copied(tmp_path):
    photo = tmp_path / "library.jpg"
    photo.write_bytes(b"on-disk")
    recent_files.remember(photo.name, b"on-disk", "photo", _image(), path=str(photo))
    [entry] = recent_files.load()
    assert entry["copy"] is None and entry["path"] == str(photo)
    recent_files.forget(entry["id"])
    assert photo.is_file()  # forgetting it never deletes the user's own file
    recent_files.remember(photo.name, b"on-disk", "photo", None, path=str(photo))
    photo.unlink()
    assert recent_files.load() == []  # a moved or deleted file drops out of the list


def test_the_list_is_capped_and_dropped_entries_take_their_files_with_them():
    for number in range(recent_files.MAX_ENTRIES + 2):
        recent_files.remember(f"{number}.jpg", f"bytes {number}".encode(), "photo", _image())
    entries = recent_files.load()
    assert len(entries) == recent_files.MAX_ENTRIES and entries[0]["name"] == f"{recent_files.MAX_ENTRIES + 1}.jpg"
    kept = {name for entry in entries for name in (entry["copy"], entry["thumb"])}
    assert {path.name for path in recent_files.recent_dir().iterdir()} - {"index.json"} == kept


def test_too_large_uploads_are_not_copied_and_clear_empties_everything(monkeypatch):
    monkeypatch.setattr(recent_files, "MAX_COPY_BYTES", 4)
    assert recent_files.remember("huge.tif", b"0123456789", "photo", _image()) is None
    monkeypatch.undo()
    recent_files.remember("project.photoproject", b"zip", "project")
    recent_files.set_thumbnail(recent_files.load()[0]["id"], _image())
    assert recent_files.thumbnail_path(recent_files.load()[0]) is not None
    recent_files.clear()
    assert recent_files.load() == [] and [p.name for p in recent_files.recent_dir().iterdir()] == ["index.json"]


def test_settings_keep_each_other():
    settings.save_setting("language", "es")
    settings.save_setting("layout", {"tool-tab": "◉ Local"})
    assert settings.read_settings() == {"language": "es", "layout": {"tool-tab": "◉ Local"}}
    settings.settings_path().write_text("not json", encoding="utf-8")
    assert settings.read_settings() == {}
