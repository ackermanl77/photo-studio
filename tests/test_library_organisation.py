"""Library organisation (F21): collections, saved searches, virtual copies and the two-up compare."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.library import index_folder
from app.major_features import PhotoCatalog


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "shoot"
    folder.mkdir()
    for index, colour in enumerate(((200, 40, 40), (40, 200, 40), (40, 40, 200))):
        Image.new("RGB", (64, 48), colour).save(folder / f"IMG_{index}.jpg")
    return folder


def test_collections_filter_the_search(tmp_path):
    folder = _folder(tmp_path)
    catalog = PhotoCatalog(tmp_path / "c.sqlite")
    index_folder(folder, catalog.database, score=False)
    catalog.add_to_collection("Portfolio", folder / "IMG_0.jpg")
    catalog.add_to_collection("Portfolio", folder / "IMG_2.jpg")
    catalog.add_to_collection("Portfolio", folder / "IMG_2.jpg")  # twice is once
    catalog.add_to_collection("Family", folder / "IMG_2.jpg")
    assert catalog.collections() == {"Family": 1, "Portfolio": 2}
    assert catalog.collections_of(folder / "IMG_2.jpg") == ["Family", "Portfolio"]
    names = sorted(Path(e["path"]).name for e in catalog.search(collection="Portfolio"))
    assert names == ["IMG_0.jpg", "IMG_2.jpg"]
    assert len(catalog.search()) == 3  # no collection = everything
    catalog.remove_from_collection("Portfolio", folder / "IMG_0.jpg")
    catalog.delete_collection("Family")
    assert catalog.collections() == {"Portfolio": 1}
    with pytest.raises(ValueError):
        catalog.add_to_collection("  ", folder / "IMG_0.jpg")


def test_saved_searches_and_virtual_copies(tmp_path):
    catalog = PhotoCatalog(tmp_path / "c.sqlite")
    catalog.save_search("Best blue", {"text": "beach", "minimum_rating": 4, "label": "Blue", "sort": "sharpest",
                                      "collection": "Portfolio", "bogus": 1})
    catalog.save_search("Best blue", {"text": "sea", "minimum_rating": 3})  # same name replaces
    assert catalog.saved_searches() == {"Best blue": {"text": "sea", "minimum_rating": 3}}
    catalog.delete_search("Best blue")
    assert catalog.saved_searches() == {}

    photo = tmp_path / "IMG_1.jpg"
    first = catalog.add_virtual_copy(photo, "Black & white", {"saturation_amount": -100})
    second = catalog.add_virtual_copy(photo, "", {"exposure_ev": 0.5})
    copies = catalog.virtual_copies(photo)
    assert [(c["name"], c["recipe"]) for c in copies] == [("Black & white", {"saturation_amount": -100}),
                                                          ("Virtual copy", {"exposure_ev": 0.5})]
    catalog.update_virtual_copy(second, {"exposure_ev": 1.0}, name="Bright")
    catalog.delete_virtual_copy(first)
    assert [(c["id"], c["name"], c["recipe"]) for c in catalog.virtual_copies()] == [(second, "Bright", {"exposure_ev": 1.0})]


def test_an_old_catalog_gains_the_new_tables(tmp_path):
    import sqlite3

    database = tmp_path / "old.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE photos (path TEXT PRIMARY KEY, rating INTEGER DEFAULT 0, tags TEXT DEFAULT '')")
        connection.execute("INSERT INTO photos(path, rating, tags) VALUES ('a.jpg', 3, '')")
    catalog = PhotoCatalog(database)
    catalog.add_to_collection("Keep", "a.jpg")
    assert [e["path"] for e in catalog.search(collection="Keep")] == ["a.jpg"]


@pytest.mark.slow
def test_the_library_tab_flow(monkeypatch, tmp_path):
    from tests.test_frontend_unified_app import _fake_process_via_backend, run_with_uploaded_image

    from app import backend_client
    from app.config import user_data_file

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    folder = _folder(tmp_path)
    database = user_data_file("catalog.sqlite")
    index_folder(folder, database, score=False)
    at = run_with_uploaded_image()
    at.radio(key="more-mode").set_value("Library Catalog")
    at.session_state["records"] = [{"path": str(folder / f"IMG_{i}.jpg"), "name": f"IMG_{i}.jpg"} for i in range(3)]
    at.run(timeout=30)
    assert not at.exception

    # a collection for the photo under the cursor, then the filter
    at.text_input(key="lib-coll-new").set_value("Portfolio")
    at.button(key="lib-coll-add").click()
    at.run(timeout=30)
    catalog = PhotoCatalog(database)
    assert catalog.collections_of(folder / "IMG_0.jpg") == ["Portfolio"]
    at.selectbox(key="lib-collection-filter").set_value("Portfolio")
    at.slider(key="lib-r-in").set_value(0)
    at.run(timeout=30)
    assert not at.exception

    # save that search, change the filters, apply it again
    at.text_input(key="lib-search-name").set_value("Portfolio picks")
    at.button(key="lib-search-save").click()
    at.run(timeout=30)
    assert catalog.saved_searches()["Portfolio picks"]["collection"] == "Portfolio"
    at.selectbox(key="lib-collection-filter").set_value("Any")
    at.text_input(key="lib-q-in").set_value("nothing-like-this")
    at.run(timeout=30)
    at.button(key="lib-search-apply").click()
    at.run(timeout=30)
    at.run(timeout=30)
    assert at.selectbox(key="lib-collection-filter").value == "Portfolio"
    assert at.text_input(key="lib-q-in").value == ""

    # a virtual copy needs the photo open in the editor first
    assert at.button(key="lib-vc-save").disabled
    at.button(key="lib-edit-current-btn").click()
    at.run(timeout=30)
    at.run(timeout=30)
    at.text_input(key="lib-vc-name").set_value("Moody")
    at.button(key="lib-vc-save").click()
    at.run(timeout=30)
    copies = catalog.virtual_copies(folder / "IMG_0.jpg")
    assert [c["name"] for c in copies] == ["Moody"] and isinstance(copies[0]["recipe"], dict)
    at.button(key=f"lib-vc-open-{copies[0]['id']}").click()
    at.run(timeout=30)
    assert not at.exception

    # two-up compare
    at.toggle(key="lib-compare").set_value(True)
    at.run(timeout=30)
    assert not at.exception
    at.button(key="lib-cmp-reject-1").click()
    at.run(timeout=30)
    assert "rejected" in catalog.get(folder / "IMG_1.jpg")["tags"]
    at.button(key="lib-compare-swap").click()
    at.run(timeout=30)
    assert at.session_state["lib-cursor"] == 1
