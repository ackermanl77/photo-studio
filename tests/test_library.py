from PIL import Image

from app.library import image_info, index_folder, scan_folder, thumbnail
from app.major_features import PhotoCatalog


def test_library_scans_indexes_and_creates_thumbnails(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    source = folder / "portrait.jpg"
    Image.new("RGB", (640, 480), (90, 100, 110)).save(source)
    (folder / "notes.txt").write_text("ignore", encoding="utf-8")
    database = tmp_path / "catalog.sqlite"

    assert scan_folder(folder) == [source]
    assert image_info(source)["width"] == 640
    records = index_folder(folder, database)
    preview = thumbnail(source, tmp_path / "thumbs")

    assert len(records) == 1
    assert preview.exists()
    assert PhotoCatalog(database).search("portrait")
