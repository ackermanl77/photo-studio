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


def test_project_archive_validation_accepts_safe_manifest():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("project.json", json.dumps({"version": 1, "files": {"source": "source.jpg"}}))
        bundle.writestr("source.jpg", b"data")

    assert validate_project_archive(archive.getvalue())["version"] == 1
