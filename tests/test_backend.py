import io
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

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


def test_process_endpoint_applies_the_hsl_color_mixer():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(color=(220, 40, 40), size=(12, 12))), "input.png"),
            "options": json.dumps({"hsl_bands": {"Red": {"hue": 0, "sat": -100, "lum": 0}}}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)), dtype=np.int32)
    # A full desaturation of the Red band on a solid red image should land close to gray (G ~= B).
    assert abs(int(result[0, 0, 1]) - int(result[0, 0, 2])) < 5


def test_process_endpoint_applies_lens_corrections():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(color=(200, 200, 200), size=(40, 40))), "input.png"),
            "options": json.dumps({"lens_vignette_amount": -80, "lens_vignette_midpoint": 30}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)), dtype=np.int32)
    assert result[:4, :4].mean() < result[18:22, 18:22].mean() - 10  # corners darkened


def test_process_endpoint_preserves_icc_profile():
    """Malformed/placeholder bytes under "icc_profile" (this fake marker isn't a real ICC
    profile) -- normalize_to_srgb() can't open it, so process_image() falls back to the old
    preserve-as-is behavior instead of failing the whole edit over an unparseable color tag."""
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


def test_process_endpoint_normalizes_a_real_embedded_profile_to_srgb():
    """A genuinely valid embedded ICC profile (unlike the fake marker bytes above) should be
    converted through normalize_to_srgb() and the output re-tagged with a real sRGB profile
    rather than the original bytes -- confirms the actual color-management code path runs
    end-to-end through /process, not just that invalid profiles degrade gracefully."""
    from app.professional_color import srgb_icc_bytes

    profile = srgb_icc_bytes()
    image = Image.new("RGB", (12, 12), color=(180, 90, 40))
    source = io.BytesIO()
    image.save(source, format="PNG", icc_profile=profile)
    source.seek(0)
    client = app.test_client()

    response = client.post(
        "/process",
        data={"image": (source, "profiled.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = Image.open(io.BytesIO(response.data))
    from PIL import ImageCms

    output_profile = result.info.get("icc_profile")
    assert output_profile  # re-tagged with a real profile (not silently dropped)
    assert ImageCms.getProfileName(ImageCms.ImageCmsProfile(io.BytesIO(output_profile))).strip()  # a well-formed ICC profile, not garbage
    pixel = np.array(result, dtype=np.int16)[0, 0]
    assert (np.abs(pixel - np.array([180, 90, 40])) <= 2).all()  # near-identity for an already-sRGB source


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


def test_process_endpoint_removes_the_selected_region():
    """PhotoProcessor.remove_objects() (cv2.inpaint) via the removal_mask form field -- this
    endpoint plumbing existed before the unified UI ever sent it (see docs/PROJECT_STATUS.md's
    Repair tab entry); this is the first test exercising it through /process."""
    size = (40, 40)
    object_box = (15, 15, 25, 25)
    base = Image.new("RGB", size, (30, 30, 30))
    ImageDraw.Draw(base).rectangle(object_box, fill=(220, 20, 20))
    base_buffer = io.BytesIO()
    base.save(base_buffer, format="PNG")

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(object_box, fill=255)
    mask_buffer = io.BytesIO()
    mask.save(mask_buffer, format="PNG")

    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(base_buffer.getvalue()), "input.png"),
            "removal_mask": (io.BytesIO(mask_buffer.getvalue()), "mask.png"),
            "options": json.dumps({}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)))
    # Inpainting blends the removed region in from the surrounding dark background -- the red
    # object (220, 20, 20) should be gone, replaced by something close to the background (30).
    removed_region_mean = result[15:25, 15:25].mean()
    assert abs(removed_region_mean - 30) < 10


def test_process_endpoint_applies_a_free_form_crop_box():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(size=(40, 30))), "input.png"),
            "options": json.dumps({"crop_box": [10, 5, 30, 25]}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = Image.open(io.BytesIO(response.data))
    assert result.size == (20, 20)


def test_process_endpoint_crop_box_takes_priority_over_aspect():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(size=(40, 30))), "input.png"),
            "options": json.dumps({"crop_box": [0, 0, 12, 8], "aspect": "1:1"}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = Image.open(io.BytesIO(response.data))
    # A 1:1 aspect fit on a 40x30 source would land at 30x30 -- confirms crop_box won, not aspect.
    assert result.size == (12, 8)


def test_process_endpoint_returns_clean_json_error_for_a_cv2_failure(monkeypatch):
    """apply_denoise() now calls cv2.fastNlMeansDenoisingColored (a real cv2.error source, e.g.
    for a degenerate input) instead of the old flat median filter -- regression test for a real
    gap found in review: cv2.error wasn't in the /process route's except tuple, so it used to
    fall through to Flask's generic HTML 500 instead of the same clean JSON error every other
    failure mode gets. Raises cv2.error directly (rather than trying to construct an image that
    actually triggers OpenCV's internal assertion) to test the route's error handling in
    isolation from OpenCV's own input-validation behavior."""

    def fake_apply_denoise(self, radius=3):
        raise cv2.error("synthetic failure for this test")

    monkeypatch.setattr(PhotoProcessor, "apply_denoise", fake_apply_denoise)

    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes()), "input.png"),
            "options": json.dumps({"denoise": 3}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.is_json
    assert "synthetic failure for this test" in response.get_json()["error"]


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
