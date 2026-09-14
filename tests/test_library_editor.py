from pathlib import Path

from PIL import Image

from app.library import index_folder
from app.major_features import PhotoCatalog


def test_library_editor_source_selection_contract(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    image_path = folder / "selected.jpg"
    Image.new("RGB", (20, 12), (100, 110, 120)).save(image_path)
    database = tmp_path / "catalog.sqlite"

    records = index_folder(folder, database)
    results = PhotoCatalog(database).search("selected")

    assert records[0]["path"] == str(image_path)
    assert results[0]["path"] == str(image_path)
    assert Path(results[0]["path"]).exists()
