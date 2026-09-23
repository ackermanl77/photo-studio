from pathlib import Path
import zipfile
import io

from PIL import Image

import pytest

from app.batch_release import FileLike, export_multiple, make_zip, process_files, render_name


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


def test_export_multiple_bundles_every_selected_profile_into_one_zip():
    image = Image.new("RGB", (3000, 2000), (100, 120, 140))

    zip_bytes = export_multiple(image, ["Web", "Social"], "poster")

    archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    assert set(archive.namelist()) == {"poster-web.jpg", "poster-social.jpg"}
    # Each entry actually decodes and respects its own profile's max_size.
    with Image.open(io.BytesIO(archive.read("poster-web.jpg"))) as web_output:
        assert web_output.width <= 2048
    with Image.open(io.BytesIO(archive.read("poster-social.jpg"))) as social_output:
        assert social_output.width <= 1080


def test_export_multiple_requires_at_least_one_profile():
    with pytest.raises(ValueError):
        export_multiple(Image.new("RGB", (10, 10)), [], "poster")


def test_process_files_applies_a_saved_recipe_across_every_photo(tmp_path):
    """The `recipe` parameter (a saved custom preset applied to a batch -- "sync settings" across
    a shoot) takes priority over the simpler named `preset`, and actually changes the output --
    routed through app.backend.process_image() rather than the plain apply_preset() path."""
    first = tmp_path / "one.jpg"
    second = tmp_path / "two.jpg"
    Image.new("RGB", (20, 20), (100, 100, 100)).save(first)
    Image.new("RGB", (20, 20), (100, 100, 100)).save(second)
    recipe = {"exposure": 1.4, "vibrance": 60}

    baseline, _ = process_files([first], profile="Web")
    with_recipe, report = process_files([first, second], profile="Web", recipe=recipe)

    assert len(with_recipe) == 2
    assert all(item.status == "ok" for item in report)
    assert baseline[0][1] != with_recipe[0][1]


def test_process_files_recipe_takes_priority_over_named_preset(tmp_path):
    source = tmp_path / "one.jpg"
    Image.new("RGB", (20, 20), (100, 100, 100)).save(source)

    with_recipe, report = process_files([source], preset="travel-clean", profile="Web", recipe={"exposure": 1.3})

    assert report[0].status == "ok"
    assert len(with_recipe) == 1


def test_process_files_calls_progress_callback_for_each_file(tmp_path):
    """The frontend's Batch Processor wires this into an st.progress() bar -- a real gap found
    live (the callback parameter already existed and was tested for its own error-reporting
    behavior, but nothing exercised the callback itself, and the UI never passed one in)."""
    first = tmp_path / "one.jpg"
    second = tmp_path / "two.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(first)
    Image.new("RGB", (8, 8), (40, 50, 60)).save(second)
    calls = []

    def progress(index, total, last_result):
        calls.append((index, total, last_result.status if last_result else None))

    process_files([first, second], profile="Web", progress=progress)

    assert calls == [(1, 2, "ok"), (2, 2, "ok")]


def test_process_files_progress_callback_reports_per_file_errors_too(tmp_path):
    good = tmp_path / "one.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(good)
    calls = []

    process_files([good, tmp_path / "missing.jpg"], profile="Web", progress=lambda i, t, r: calls.append(r.status))

    assert calls == ["ok", "error"]


def test_file_like_matches_the_flask_file_storage_interface_process_image_needs():
    file_like = FileLike("photo.png", b"fake-bytes")
    assert file_like.filename == "photo.png"
    assert file_like.read() == b"fake-bytes"


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
