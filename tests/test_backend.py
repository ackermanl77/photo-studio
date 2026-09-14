import io
import json

from PIL import Image

from app.backend import app
from app.config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from app.project import PhotoProject
from app.raw_import import is_raw_path


def image_bytes(color=(100, 110, 120), size=(12, 12), format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", size, color=color).save(stream, format=format)
    return stream.getvalue()


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_backend_security_defaults():
    assert app.config["MAX_CONTENT_LENGTH"] == MAX_UPLOAD_BYTES
    assert MAX_IMAGE_PIXELS > 0
    assert app.config["MAX_CONTENT_LENGTH"] > 0


def test_raw_extensions_are_recognized_without_changing_normal_images():
    assert is_raw_path("camera.CR3")
    assert is_raw_path("camera.nef")
    assert not is_raw_path("photo.jpg")


def test_process_endpoint_returns_requested_jpeg():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes()), "input.png"),
            "options": json.dumps({"output_format": "JPEG", "quality": 85, "exposure": 1.1}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert Image.open(io.BytesIO(response.data)).format == "JPEG"


def test_process_endpoint_preserves_icc_profile():
    image = Image.new("RGB", (12, 12), color=(100, 110, 120))
    source = io.BytesIO()
    profile = b"test-icc-profile"
    image.save(source, format="JPEG", icc_profile=profile)
    source.seek(0)
    client = app.test_client()

    response = client.post(
        "/process",
        data={
            "image": (source, "profiled.jpg"),
            "options": json.dumps({"output_format": "JPEG"}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.data)).info.get("icc_profile") == profile


def test_process_endpoint_rejects_unknown_preset():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes()), "input.png"),
            "options": json.dumps({"preset": "missing"}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Unknown preset" in response.get_json()["error"]


def test_photo_project_round_trip(tmp_path):
    project_path = tmp_path / "session.photoproject"
    original = PhotoProject("input.png", image_bytes(), {"exposure": 1.2}, b"remove", b"local")

    original.save(project_path)
    restored = PhotoProject.load(project_path)

    assert restored.source_name == "input.png"
    assert restored.source_bytes == original.source_bytes
    assert restored.recipe == {"exposure": 1.2}
    assert restored.removal_mask == b"remove"
    assert restored.local_mask == b"local"
