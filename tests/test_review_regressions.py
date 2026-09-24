"""Regression tests for a batch of bugs found in a code review (see "Recently Fixed" in
docs/PROJECT_STATUS.md): RAW decoding on Windows, the upload byte cap rejecting the frontend's own
internal Stage 2 hop, runtime files written into the install folder, EXIF orientation, the pixel
limit admitting 2x its value, and export_profile() ignoring strip_metadata."""

import io
import json
import sys
import types

import pytest
from PIL import Image

from app import backend, backend_client, raw_import, reliability
from app.backend import app
from app.config import INTERNAL_STAGE_HEADER, INTERNAL_STAGE_RESUME, user_data_dir
from app.professional_tools import export_profile


def _png(size=(12, 8), color=(90, 120, 160)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeLibRawError(Exception):
    pass


def _install_fake_rawpy(monkeypatch, imread):
    fake = types.ModuleType("rawpy")
    fake.LibRawError = _FakeLibRawError
    fake.imread = imread
    monkeypatch.setitem(sys.modules, "rawpy", fake)


def test_raw_bytes_are_decoded_from_memory_not_a_reopened_temp_file(monkeypatch):
    """rawpy.imread() used to get the name of a still-open NamedTemporaryFile, which Windows
    refuses to reopen (LibRawIOError on every RAW upload) -- it must get a file-like object."""
    seen = {}

    class _Raw:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def postprocess(self, **kwargs):
            import numpy as np

            return np.zeros((4, 6, 3), dtype=np.uint8)

    def fake_imread(source):
        seen["source"] = source
        return _Raw()

    _install_fake_rawpy(monkeypatch, fake_imread)

    image = raw_import.open_bytes(b"raw-bytes", "photo.NEF")

    assert image.size == (6, 4)
    assert hasattr(seen["source"], "read") and seen["source"].read() == b"raw-bytes"


def test_a_corrupt_raw_raises_runtime_error_and_a_clean_400(monkeypatch):
    def fake_imread(source):
        raise _FakeLibRawError("Unsupported file format or not RAW file")

    _install_fake_rawpy(monkeypatch, fake_imread)

    with pytest.raises(RuntimeError, match="camera RAW"):
        raw_import.open_bytes(b"junk", "photo.cr2")

    response = app.test_client().post(
        "/process",
        data={"image": (io.BytesIO(b"junk"), "photo.cr2"), "options": "{}"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "camera RAW" in response.get_json()["error"]


def test_internal_stage_two_upload_is_not_held_to_the_user_upload_byte_cap(monkeypatch):
    """A large photo's Stage 1 PNG is bigger than MAX_UPLOAD_BYTES even when the original JPEG
    wasn't -- the resume_from_detail hop (marked with INTERNAL_STAGE_HEADER) must still go
    through, while an ordinary upload of the same size is still rejected."""
    monkeypatch.setitem(app.config, "MAX_CONTENT_LENGTH", 200)
    payload = _png((30, 30), (1, 2, 3))
    client = app.test_client()

    def post(headers):
        return client.post(
            "/process",
            data={"image": (io.BytesIO(payload), "stage_a.png"), "options": json.dumps({"resume_from_detail": True})},
            content_type="multipart/form-data",
            headers=headers,
        )

    rejected = post({})
    assert rejected.status_code == 413
    # A clean JSON error, not an HTML 500 from the audit hook re-raising the 413.
    assert "exceeds" in rejected.get_json()["error"]
    assert post({INTERNAL_STAGE_HEADER: INTERNAL_STAGE_RESUME}).status_code == 200


def test_process_via_backend_marks_only_stage_two_requests_as_internal(monkeypatch):
    seen = []

    class _Response:
        ok = True
        content = _png()

    def fake_post(url, data=None, files=None, headers=None, timeout=None):
        seen.append(headers or {})
        return _Response()

    monkeypatch.setattr(backend_client.requests, "post", fake_post)

    backend_client.process_via_backend(_png(), "a.png", {"stop_after": "detail"})
    backend_client.process_via_backend(_png(), "stage_a.png", {"resume_from_detail": True})

    assert INTERNAL_STAGE_HEADER not in seen[0]
    assert seen[1][INTERNAL_STAGE_HEADER] == INTERNAL_STAGE_RESUME


def test_pixel_limit_is_enforced_exactly_not_at_twice_its_value(monkeypatch):
    """PIL alone only warns between 1x and 2x Image.MAX_IMAGE_PIXELS -- an image 1.44x the
    configured limit used to be accepted."""
    with pytest.raises(Image.DecompressionBombError):
        reliability.validate_image_bytes(_png((12, 12)), "big.png", max_pixels=100)
    assert reliability.validate_image_bytes(_png((10, 10)), "ok.png", max_pixels=100)[:2] == (10, 10)

    monkeypatch.setattr(reliability, "MAX_IMAGE_PIXELS", 100)
    for options in ({}, {"resume_from_detail": True}):
        response = app.test_client().post(
            "/process",
            data={"image": (io.BytesIO(_png((12, 12))), "big.png"), "options": json.dumps(options)},
            content_type="multipart/form-data",
        )
        assert response.status_code == 413


def _jpeg_with_orientation(size, orientation, extra_tags=None) -> bytes:
    image = Image.new("RGB", size, (200, 30, 30))
    exif = Image.Exif()
    exif[0x0112] = orientation
    for tag, value in (extra_tags or {}).items():
        exif[tag] = value
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    return buffer.getvalue()


def test_exif_orientation_is_applied_at_decode(tmp_path):
    """A phone photo stored landscape with "Orientation: rotate 90" used to open sideways."""
    content = _jpeg_with_orientation((40, 20), 6)

    decoded = raw_import.open_bytes(content, "phone.jpg")
    assert decoded.size == (20, 40)
    assert decoded.getexif().get(0x0112) in (None, 1)  # not re-applied by a viewer on export

    path = tmp_path / "phone.jpg"
    path.write_bytes(content)
    assert raw_import.open_image(path).size == (20, 40)


def test_export_profile_honors_strip_metadata():
    image = Image.open(io.BytesIO(_jpeg_with_orientation((16, 16), 1, {0x010F: "CameraMaker"})))
    image.load()

    archived, _, _ = export_profile(image, "Archive")
    web, _, _ = export_profile(image, "Web")

    assert Image.open(io.BytesIO(archived)).getexif().get(0x010F) == "CameraMaker"
    assert 0x010F not in Image.open(io.BytesIO(web)).getexif()


def test_runtime_files_go_to_the_per_user_data_folder_not_the_install_folder(monkeypatch, tmp_path):
    """Logs and AI Upscale weights used to be written next to PhotoStudio.exe / into the bundled
    realesrgan package -- both under Program Files in an installed build, so unwritable."""
    from app import ai_upscale, desktop

    monkeypatch.setenv("PHOTO_STUDIO_DATA_DIR", str(tmp_path / "data"))
    assert user_data_dir() == tmp_path / "data"
    assert desktop._log_path("backend-child.log").parent == tmp_path / "data" / "logs"
    assert ai_upscale.weights_dir() == tmp_path / "data" / "weights"

    monkeypatch.setattr(desktop.sys, "frozen", True, raising=False)
    assert desktop._child_cwd() == tmp_path / "data"


def test_user_data_dir_uses_localappdata_on_windows(monkeypatch, tmp_path):
    from app import config

    monkeypatch.delenv("PHOTO_STUDIO_DATA_DIR", raising=False)
    monkeypatch.setattr(config.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert config.user_data_dir() == tmp_path / "PhotoStudio"


def test_ai_upscale_ui_gate_does_not_import_torch(monkeypatch):
    from app import ai_upscale

    ai_upscale.ai_upscale_installed.cache_clear()
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    result = ai_upscale.ai_upscale_installed()
    assert isinstance(result, bool)
    assert "torch" not in sys.modules
    ai_upscale.ai_upscale_installed.cache_clear()
