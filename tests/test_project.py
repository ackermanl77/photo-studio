import io
import json
import zipfile

import pytest

from app.project import PhotoProject


def test_photo_project_load_rejects_an_archive_with_too_many_entries():
    """Regression test: PhotoProject.load() used to open the ZIP directly and never called
    app.reliability.validate_project_archive(), so it had none of the entry-count/size/path/
    manifest-reference checks ProjectBundle.load() already had -- this exercises the entry-count
    guard specifically, the cheapest one to trigger."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("project.json", json.dumps({"version": 1, "files": {"source": "source.png"}}))
        bundle.writestr("source.png", b"data")
        for i in range(205):  # over the 200-entry cap
            bundle.writestr(f"junk{i}.txt", "x")

    with pytest.raises(ValueError, match="too many files"):
        PhotoProject.load(archive.getvalue())


def test_photo_project_load_rejects_an_unsafe_path():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("project.json", json.dumps({"version": 1, "files": {"source": "source.png"}}))
        bundle.writestr("source.png", b"data")
        bundle.writestr("../outside.txt", "bad")

    with pytest.raises(ValueError, match="unsafe path"):
        PhotoProject.load(archive.getvalue())


def test_photo_project_load_rejects_a_manifest_referencing_a_missing_removal_mask():
    """Without this check, a corrupt/hand-edited manifest would surface as a bare KeyError from
    archive.read(files["removal_mask"]) instead of a clear error -- same class of bug already
    fixed for ProjectBundle.load()."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "project.json",
            json.dumps({"version": 1, "files": {"source": "source.png", "removal_mask": "masks/removal.png"}}),
        )
        bundle.writestr("source.png", b"data")
        # masks/removal.png is deliberately never written

    with pytest.raises(ValueError, match="missing removal_mask file"):
        PhotoProject.load(archive.getvalue())


def test_photo_project_round_trip_still_works_through_the_new_validation(tmp_path):
    """The validation added to load() must not break the normal save/load path."""
    project_path = tmp_path / "session.photoproject"
    original = PhotoProject("input.png", b"source-bytes", {"exposure": 1.2}, b"remove", b"local")

    original.save(project_path)
    restored = PhotoProject.load(project_path)

    assert restored.source_name == "input.png"
    assert restored.source_bytes == original.source_bytes
    assert restored.removal_mask == b"remove"
    assert restored.local_mask == b"local"
    assert restored.recipe == {"exposure": 1.2}
