"""HEIC/HEIF and AVIF import, AVIF export (F19)."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from app.color_profiles import display_p3_icc_bytes
from app.export import EXPORT_FORMATS, LOSSY_FORMATS, encode_export, format_for_path
from app.raw_import import SUPPORTED_EXTENSIONS, uploader_types

pillow_heif = pytest.importorskip("pillow_heif")
needs_avif = pytest.mark.skipif("AVIF" not in EXPORT_FORMATS, reason="Pillow built without AVIF")


def _heic(image: Image.Image, **info) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "HEIF", quality=95, **info)
    return buffer.getvalue()


def _exif(orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    exif[0x010F] = "Apple"
    return exif.tobytes()


def _process(content: bytes, name: str, options: dict) -> Image.Image:
    from app.backend import app

    response = app.test_client().post(
        "/process", data={"image": (io.BytesIO(content), name), "options": json.dumps(options)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.data[:300]
    return Image.open(io.BytesIO(response.data))


def test_heic_and_avif_are_offered_everywhere_photos_are_accepted():
    assert {".heic", ".heif"} <= SUPPORTED_EXTENSIONS
    assert {"heic", "heif"} <= set(uploader_types())


def test_an_iphone_style_heic_opens_upright_with_its_profile_and_exports():
    image = Image.new("RGB", (40, 20), (200, 30, 30))
    image.paste((30, 30, 200), (0, 0, 20, 20))  # blue on the left
    content = _heic(image, exif=_exif(6), icc_profile=display_p3_icc_bytes())  # 6 = rotate 90 CW to view
    out = _process(content, "IMG_0001.HEIC", {"output_format": "PNG"})
    assert out.size == (20, 40)  # EXIF orientation applied
    top = np.asarray(out.convert("RGB"))[2, 10].astype(int)
    assert top[2] > top[0]  # the blue half is on top after the rotation
    # Its Display P3 profile reaches a P3 export untouched (B14's working space).
    p3 = _process(content, "IMG_0001.HEIC", {"export": {"format": "PNG", "color_space": "Display P3"}})
    assert p3.info["icc_profile"] == display_p3_icc_bytes()


@needs_avif
def test_avif_export_keeps_alpha_metadata_and_profile_and_respects_quality():
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    image.info["exif"] = _exif(1)
    small, ext, mime = encode_export(image, {"format": "AVIF", "quality": 40})
    large, _ext, _mime = encode_export(image, {"format": "AVIF", "quality": 95})
    assert (ext, mime) == (".avif", "image/avif") and len(small) < len(large)
    decoded = Image.open(io.BytesIO(large))
    assert decoded.format == "AVIF" and decoded.size == (64, 64)
    assert decoded.getexif().get(0x010F) == "Apple"
    p3 = Image.open(io.BytesIO(encode_export(image, {"format": "AVIF", "color_space": "Display P3"})[0]))
    assert p3.info["icc_profile"] == display_p3_icc_bytes()
    rgba = Image.new("RGBA", (16, 16), (10, 200, 10, 90))
    assert Image.open(io.BytesIO(encode_export(rgba, {"format": "AVIF"})[0])).mode == "RGBA"
    # An AVIF opens as a source too.
    assert ".avif" in SUPPORTED_EXTENSIONS
    assert _process(large, "photo.avif", {"output_format": "PNG"}).size == (64, 64)


@needs_avif
def test_the_cli_and_export_panel_know_avif():
    assert format_for_path("out.avif") == "AVIF"
    assert "AVIF" in LOSSY_FORMATS
    with pytest.raises(ValueError, match=r"\.avif"):
        format_for_path("out.bmp")
