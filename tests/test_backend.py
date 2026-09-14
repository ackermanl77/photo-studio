import io
import json

import numpy as np
from PIL import Image

from app.backend import app
from app.config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from app.image_processor import PhotoProcessor
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


def test_process_endpoint_composites_layer():
    client = app.test_client()
    base = image_bytes((40, 40, 40))
    overlay = image_bytes((220, 30, 30))

    baseline = client.post(
        "/process",
        data={"image": (io.BytesIO(base), "input.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    with_layer = client.post(
        "/process",
        data={
            "image": (io.BytesIO(base), "input.png"),
            "layer": (io.BytesIO(overlay), "layer.png"),
            "options": json.dumps({"layer_opacity": 1.0, "layer_blend": "normal"}),
        },
        content_type="multipart/form-data",
    )

    assert baseline.status_code == 200
    assert with_layer.status_code == 200
    assert baseline.data != with_layer.data


def test_process_endpoint_masks_apply_after_layer_composite():
    """The combined mask must see the image AFTER the layer is composited (matching the
    unified UI's Layers -> Masks tab order) -- otherwise a masked adjustment could miss pixels
    the layer just changed. This is the ordering bug class documented under "Recently Fixed"
    in docs/PROJECT_STATUS.md, reproduced here for the backend's own pipeline."""
    base = image_bytes((40, 40, 40), size=(20, 20))
    overlay = image_bytes((250, 250, 250), size=(20, 20))
    mask_buffer = io.BytesIO()
    Image.new("L", (20, 20), 255).save(mask_buffer, format="PNG")

    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(base), "input.png"),
            "layer": (io.BytesIO(overlay), "layer.png"),
            "local_mask": (io.BytesIO(mask_buffer.getvalue()), "mask.png"),
            "options": json.dumps({"layer_opacity": 1.0, "layer_blend": "normal", "mask_exposure": 0.5}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)))
    # Halving exposure on the post-layer (white, 250) image lands at ~125; if masking had seen
    # the pre-layer (dark, 40) image instead, it would land at ~20.
    assert abs(result.mean() - 125.0) < 2.0


def test_process_endpoint_applies_portrait_smoothing():
    rng = np.random.default_rng(0)
    noisy = np.clip(np.array([180, 120, 90], dtype=np.float32) + rng.uniform(-25, 25, size=(24, 24, 3)), 0, 255).astype("uint8")
    stream = io.BytesIO()
    Image.fromarray(noisy, "RGB").save(stream, format="PNG")
    payload = stream.getvalue()

    client = app.test_client()
    unsmoothed = client.post(
        "/process",
        data={"image": (io.BytesIO(payload), "face.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    smoothed = client.post(
        "/process",
        data={"image": (io.BytesIO(payload), "face.png"), "options": json.dumps({"portrait_smoothing": 0.8})},
        content_type="multipart/form-data",
    )

    assert unsmoothed.status_code == 200
    assert smoothed.status_code == 200
    unsmoothed_arr = np.array(Image.open(io.BytesIO(unsmoothed.data))).astype(int)
    smoothed_arr = np.array(Image.open(io.BytesIO(smoothed.data))).astype(int)
    assert np.abs(unsmoothed_arr - smoothed_arr).mean() > 1.0


def test_process_endpoint_skips_pil_validation_for_raw_uploads(monkeypatch):
    """validate_image_bytes() uses PIL, which can't decode camera RAW -- for a RAW filename,
    process_image() must skip straight to PhotoProcessor.from_bytes() (stubbed here) instead of
    rejecting the raw camera bytes as "Invalid image content"."""

    def fake_from_bytes(cls, content, filename):
        processor = cls.__new__(cls)
        processor.input_path = None
        processor.image = Image.new("RGB", (8, 8), (50, 60, 70))
        return processor

    monkeypatch.setattr(PhotoProcessor, "from_bytes", classmethod(fake_from_bytes))

    client = app.test_client()
    response = client.post(
        "/process",
        data={"image": (io.BytesIO(b"not a real raw file"), "photo.cr2"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200


def test_process_endpoint_still_validates_non_raw_uploads():
    client = app.test_client()
    response = client.post(
        "/process",
        data={"image": (io.BytesIO(b"not an image"), "photo.jpg"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Invalid image content" in response.get_json()["error"]


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
