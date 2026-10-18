"""Culling scores (F20) and colour labels (F21) in the Library."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from app import library
from app.culling import BLURRY_FOCUS, is_likely_blurry, score_image
from app.library import index_folder, save_rating
from app.major_features import COLOR_LABELS, PhotoCatalog
from app.xmp_sidecar import find_sidecar, read_sidecar, write_sidecar

ASSET = Path(__file__).parent / "assets" / "portrait_public_domain.jpg"


def _photo() -> Image.Image:
    return Image.open(ASSET).convert("RGB")


def test_focus_falls_with_blur_and_ignores_noise():
    sharp = _photo()
    scores = [score_image(sharp if r == 0 else sharp.filter(ImageFilter.GaussianBlur(r)))["focus"] for r in (0, 1, 2, 4)]
    assert scores == sorted(scores, reverse=True) and scores[0] - scores[-1] > 40
    assert not is_likely_blurry(scores[0]) and is_likely_blurry(scores[-1])
    blurred = np.asarray(sharp.filter(ImageFilter.GaussianBlur(2)), np.float32)
    noisy = Image.fromarray(np.clip(blurred + np.random.default_rng(0).normal(0, 5, blurred.shape), 0, 255).astype(np.uint8))
    assert score_image(noisy)["focus"] < scores[0] - 15  # grain doesn't make a blurred photo "sharp"
    assert score_image(Image.new("RGB", (300, 200), (120, 120, 120)))["focus"] == 0


def test_clipping_shares():
    image = Image.new("RGB", (100, 100), (128, 128, 128))
    image.paste((255, 255, 255), (0, 0, 50, 100))
    image.paste((0, 0, 0), (50, 0, 60, 100))
    scores = score_image(image)
    assert scores["highlights"] == pytest.approx(50.0) and scores["shadows"] == pytest.approx(10.0)


def _folder(tmp_path) -> Path:
    folder = tmp_path / "shoot"
    folder.mkdir()
    sharp = _photo()
    sharp.save(folder / "sharp.jpg", quality=95)
    sharp.filter(ImageFilter.GaussianBlur(4)).save(folder / "shaky.jpg", quality=95)
    sharp.filter(ImageFilter.GaussianBlur(1)).save(folder / "soft.jpg", quality=95)
    return folder


def test_scanning_scores_photos_once_and_rescores_changed_files(tmp_path, monkeypatch):
    folder, database = _folder(tmp_path), tmp_path / "catalog.sqlite"
    calls = []
    real = library.score_file
    monkeypatch.setattr(library, "score_file", lambda path: calls.append(Path(path).name) or real(path))
    index_folder(folder, database)
    assert sorted(calls) == ["shaky.jpg", "sharp.jpg", "soft.jpg"]
    catalog = PhotoCatalog(database)
    assert is_likely_blurry(catalog.get(folder / "shaky.jpg")["focus"])
    assert catalog.get(folder / "sharp.jpg")["focus"] > BLURRY_FOCUS
    calls.clear()
    index_folder(folder, database)
    assert calls == []  # unchanged files keep their scores
    later = (folder / "soft.jpg").stat().st_mtime + 10
    os.utime(folder / "soft.jpg", (later, later))
    index_folder(folder, database)
    assert calls == ["soft.jpg"]


def test_search_sorts_by_focus_and_filters_by_label(tmp_path):
    folder, database = _folder(tmp_path), tmp_path / "catalog.sqlite"
    index_folder(folder, database)
    catalog = PhotoCatalog(database)
    names = lambda entries: [Path(e["path"]).name for e in entries]  # noqa: E731
    assert names(catalog.search(sort="blurriest")) == ["shaky.jpg", "soft.jpg", "sharp.jpg"]
    assert names(catalog.search(sort="sharpest")) == ["sharp.jpg", "soft.jpg", "shaky.jpg"]
    catalog.upsert(folder / "unscored.jpg")  # never scored: last either way
    assert names(catalog.search(sort="blurriest"))[-1] == "unscored.jpg"
    catalog.set_label(folder / "sharp.jpg", "Green")
    assert names(catalog.search(label="Green")) == ["sharp.jpg"]
    assert "sharp.jpg" not in names(catalog.search(label=""))
    with pytest.raises(ValueError):
        catalog.set_label(folder / "sharp.jpg", "Orange")


def test_an_old_catalog_is_migrated_in_place(tmp_path):
    database = tmp_path / "old.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE photos (path TEXT PRIMARY KEY, rating INTEGER DEFAULT 0, tags TEXT DEFAULT '')")
        connection.execute("INSERT INTO photos VALUES ('a.jpg', 3, 'keeper')")
    entry = PhotoCatalog(database).get("a.jpg")
    assert (entry["rating"], entry["tags"], entry["label"], entry["focus"]) == (3, ["keeper"], "", None)


def test_colour_labels_round_trip_through_xmp_like_lightroom(tmp_path):
    photo = tmp_path / "IMG_1.jpg"
    _photo().save(photo)
    write_sidecar(photo, 3, ["beach"], label="Red")
    assert read_sidecar(photo) == {"rating": 3, "rejected": False, "tags": ["beach"], "label": "Red"}
    # A rating change that doesn't touch the label keeps it in the sidecar.
    database = tmp_path / "catalog.sqlite"
    catalog = PhotoCatalog(database)
    catalog.upsert(photo, 3, ["beach"])
    catalog.set_label(photo, "Red")
    save_rating(catalog, photo, 5, ["beach"], write_xmp=True)
    assert read_sidecar(photo)["label"] == "Red" and read_sidecar(photo)["rating"] == 5
    save_rating(catalog, photo, 5, ["beach"], write_xmp=True, label="")
    assert read_sidecar(photo)["label"] == "" and catalog.get(photo)["label"] == ""


def test_a_label_set_in_lightroom_is_imported_on_scan(tmp_path):
    folder = tmp_path / "lr"
    folder.mkdir()
    _photo().save(folder / "DSC_7.jpg")
    (folder / "DSC_7.xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="2" xmp:Label="Blue"/>'
        "</rdf:RDF></x:xmpmeta>", encoding="utf-8")
    database = tmp_path / "catalog.sqlite"
    index_folder(folder, database, score=False)
    entry = PhotoCatalog(database).get(folder / "DSC_7.jpg")
    assert (entry["rating"], entry["label"]) == (2, "Blue") and find_sidecar(folder / "DSC_7.jpg")
    assert set(COLOR_LABELS) == {"Red", "Yellow", "Green", "Blue", "Purple"}


@pytest.mark.slow
def test_the_library_label_buttons_and_score_line(monkeypatch, tmp_path):
    from tests.test_frontend_unified_app import _fake_process_via_backend, run_with_uploaded_image

    from app import backend_client
    from app.config import user_data_file

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    folder = _folder(tmp_path)
    index_folder(folder, user_data_file("catalog.sqlite"))
    at = run_with_uploaded_image()
    at.radio(key="more-mode").set_value("Library Catalog")
    at.session_state["records"] = [{"path": str(folder / "shaky.jpg"), "name": "shaky.jpg"}]
    at.run(timeout=30)
    assert any("likely blurry" in caption.value for caption in at.caption)
    at.button(key="lib-label-Yellow").click()
    at.run(timeout=30)
    assert not at.exception
    assert PhotoCatalog(user_data_file("catalog.sqlite")).get(folder / "shaky.jpg")["label"] == "Yellow"
    at.selectbox(key="lib-sort").set_value("blurriest")
    at.selectbox(key="lib-label-filter").set_value("Yellow")
    at.run(timeout=30)
    assert not at.exception
