import io
import json
import zipfile

import pytest
from PIL import Image

from app.reliability import image_digest, new_request_id, validate_image_bytes, validate_project_archive


def test_image_validation_and_digest():
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), (10, 20, 30)).save(stream, format="PNG")
    content = stream.getvalue()

    width, height, format_name = validate_image_bytes(content, "photo.png")

    assert (width, height, format_name) == (8, 6, "PNG")
    assert len(image_digest(content)) == 64
    assert new_request_id() != new_request_id()


def test_project_archive_validation_rejects_unsafe_paths():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("project.json", json.dumps({"version": 1, "files": {}}))
        bundle.writestr("../outside.txt", "bad")

    with pytest.raises(ValueError, match="unsafe path"):
        validate_project_archive(archive.getvalue())


def test_project_archive_validation_rejects_a_windows_drive_letter_path():
    """Regression test: a drive-rooted entry name like "C:evil.png" isn't caught by
    PurePosixPath.is_absolute() (it only recognizes a leading "/") or by the ".." check, so it
    sailed straight through the old path-safety check. Unlike a backslash-separated name --
    zipfile itself already normalizes "\\" to "/" on Windows for both entries it writes and ones
    it reads back, so that sub-case can't actually reach this check un-normalized on this app's
    only supported runtime (confirmed: even a direct zipfile.ZipInfo(filename=...) call converts
    it) -- a drive letter gets no such normalization on any platform, so this is a real, reachable
    gap regardless of host OS."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("project.json", json.dumps({"version": 1, "files": {}}))
        bundle.writestr("C:evil.png", "bad")

    with pytest.raises(ValueError, match="unsafe path"):
        validate_project_archive(archive.getvalue())


def test_project_archive_validation_accepts_safe_manifest():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("project.json", json.dumps({"version": 1, "files": {"source": "source.jpg"}}))
        bundle.writestr("source.jpg", b"data")

    assert validate_project_archive(archive.getvalue())["version"] == 1


def test_project_archive_validation_enforces_real_decompressed_size_not_declared_size():
    """A ZIP entry's declared file_size is attacker-controlled metadata, not a guarantee of the
    real decompressed byte count -- a zip bomb relies on exactly that gap. The size check has to
    bound actual bytes read while decompressing, not just sum what the archive claims about
    itself (which is what the previous, bypassable implementation did)."""
    payload = b"Q" * (2 * 1024 * 1024)  # highly compressible; real size is genuinely 2MB
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("project.json", json.dumps({"version": 1, "files": {"source": "source.bin"}}))
        bundle.writestr("source.bin", payload)

    with pytest.raises(ValueError, match="size limit"):
        validate_project_archive(archive.getvalue(), max_uncompressed_bytes=1024 * 1024)


def test_project_archive_validation_rejects_a_manifest_pointing_at_a_missing_file():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("project.json", json.dumps({"version": 1, "files": {"source": "missing.jpg"}}))

    with pytest.raises(ValueError, match="missing source image"):
        validate_project_archive(archive.getvalue())


def test_project_archive_validation_rejects_invalid_json_manifest():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("project.json", b"{not valid json")

    with pytest.raises(ValueError, match="not valid JSON"):
        validate_project_archive(archive.getvalue())
