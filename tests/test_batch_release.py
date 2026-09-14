from pathlib import Path
import zipfile
import io

from PIL import Image

from app.batch_release import make_zip, process_files, render_name


def test_batch_release_reports_success_and_errors(tmp_path):
    source = tmp_path / "portrait.jpg"
    Image.new("RGB", (20, 12), (100, 110, 120)).save(source)

    outputs, report = process_files([source, tmp_path / "missing.jpg"], profile="Social")

    assert len(outputs) == 1
    assert report[0].status == "ok"
    assert report[1].status == "error"
    assert render_name(source, "{stem}-{name}{suffix}") == "portrait-portrait.jpg.jpg"


def test_batch_zip_contains_rendered_files(tmp_path):
    source = tmp_path / "one.jpg"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(source)
    outputs, _ = process_files([source], profile="Web")

    archive = zipfile.ZipFile(io.BytesIO(make_zip(outputs)))

    assert archive.namelist() == ["one-edited.jpg"]
