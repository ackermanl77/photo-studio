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


def test_a_bad_name_template_fails_only_that_file_not_the_whole_batch(tmp_path):
    """Regression test: render_name()'s str.format() raises KeyError for an unknown field (or
    IndexError for a stray positional placeholder) in a caller-supplied name_template. Before this
    was caught alongside every other per-file failure mode, one bad template propagated straight
    out of process_files(), aborting the whole batch and discarding the report for every other
    file instead of recording just this one as an error."""
    good = tmp_path / "one.jpg"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(good)
    other = tmp_path / "two.jpg"
    Image.new("RGB", (8, 8), (50, 60, 70)).save(other)

    outputs, report = process_files([good, other], profile="Web", name_template="{bad_field}-edited{suffix}")

    assert outputs == []
    assert len(report) == 2
    assert all(item.status == "error" for item in report)
    assert "bad_field" in report[0].error
