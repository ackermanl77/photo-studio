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


def test_index_folder_calls_progress_callback_for_each_file(tmp_path):
    """The frontend's "Scan Folder" button wires this into an st.progress() bar, the same pattern
    app.batch_release.process_files()'s own `progress` callback already established for Run
    Batch -- a real gap found live (index_folder() had no progress feedback at all, unlike batch
    processing, even though scanning a large folder can take a while)."""
    folder = tmp_path / "photos"
    folder.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(folder / "one.jpg")
    Image.new("RGB", (8, 8), (40, 50, 60)).save(folder / "two.jpg")
    calls = []

    def progress(index, total, info):
        calls.append((index, total, "error" in info))

    index_folder(folder, tmp_path / "catalog.sqlite", progress=progress)

    assert calls == [(1, 2, False), (2, 2, False)]


def test_index_folder_progress_callback_reports_per_file_errors_too(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(folder / "good.jpg")
    (folder / "bad.jpg").write_bytes(b"not actually a jpeg")
    calls = []

    records = index_folder(folder, tmp_path / "catalog.sqlite", progress=lambda i, t, info: calls.append("error" in info))

    # scan_folder() sorts by path, so "bad.jpg" is scanned before "good.jpg".
    assert calls == [True, False]
    assert len(records) == 1
