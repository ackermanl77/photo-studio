"""Display P3 export (app/color_profiles.py): the generated ICC profile, the sRGB -> P3 conversion
(checked against littlecms using that profile), and the export paths that use it."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageCms

from app import cli
from app.color_profiles import (
    COLOR_SPACES,
    SRGB_TO_P3,
    convert_srgb_to_display_p3,
    convert_srgb_to_display_p3_uint8,
    display_p3_icc_bytes,
)
from app.export import ExportPresetStore, encode_export, validate_export_settings
from app.high_bit import attach


def _p3_profile():
    return ImageCms.ImageCmsProfile(io.BytesIO(display_p3_icc_bytes()))


def test_the_generated_profile_is_a_valid_display_p3_profile():
    icc = display_p3_icc_bytes()
    assert icc[36:40] == b"acsp" and int.from_bytes(icc[:4], "big") == len(icc)
    profile = _p3_profile()
    assert ImageCms.getProfileDescription(profile).strip() == "Display P3"
    assert profile.profile.xcolor_space.strip() == "RGB" and profile.profile.connection_space.strip() == "XYZ"
    # The matrix maps sRGB's white to P3's white (same D65) and stays invertible.
    assert np.allclose(SRGB_TO_P3 @ np.ones(3), np.ones(3), atol=1e-9)


def test_the_conversion_matches_littlecms_with_the_generated_profile():
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    reference = np.asarray(ImageCms.profileToProfile(
        Image.fromarray(pixels), ImageCms.createProfile("sRGB"), _p3_profile(),
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB",
    ))
    ours = convert_srgb_to_display_p3_uint8(pixels)
    assert np.abs(reference.astype(int) - ours.astype(int)).max() <= 1
    # Well-known values: sRGB primaries inside P3, neutrals unchanged.
    named = convert_srgb_to_display_p3_uint8(np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]], np.uint8))
    assert named[0].tolist() == [[234, 51, 35], [117, 251, 76], [0, 0, 245], [128, 128, 128]]
    floats = convert_srgb_to_display_p3(np.array([[0.0, 0.5, 1.0]], np.float32))
    assert floats.dtype == np.float32 and floats.min() >= 0.0 and floats.max() <= 1.0


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "TIFF"])
def test_a_display_p3_export_embeds_the_profile_and_converts_the_pixels(fmt):
    image = Image.new("RGB", (16, 16), (255, 0, 0))
    srgb_bytes, _ext, _mime = encode_export(image, {"format": fmt, "quality": 100})
    p3_bytes, _ext, _mime = encode_export(image, {"format": fmt, "quality": 100, "color_space": "Display P3"})
    srgb_file, p3_file = Image.open(io.BytesIO(srgb_bytes)), Image.open(io.BytesIO(p3_bytes))
    assert srgb_file.info.get("icc_profile") is None  # an untagged source stays untagged in sRGB
    assert p3_file.info["icc_profile"] == display_p3_icc_bytes()
    red = np.asarray(p3_file.convert("RGB"))[8, 8].astype(int)
    assert np.abs(red - [234, 51, 35]).max() <= (3 if fmt in ("JPEG", "WEBP") else 0)


def test_alpha_and_grayscale_survive_a_p3_export():
    rgba = Image.new("RGBA", (4, 4), (0, 0, 255, 100))
    out = Image.open(io.BytesIO(encode_export(rgba, {"format": "PNG", "color_space": "Display P3"})[0]))
    assert out.mode == "RGBA" and np.asarray(out)[0, 0].tolist() == [0, 0, 245, 100]
    gray = Image.new("L", (4, 4), 90)
    out = Image.open(io.BytesIO(encode_export(gray, {"format": "PNG", "color_space": "Display P3"})[0]))
    assert np.asarray(out.convert("L"))[0, 0] == 90


def test_a_16_bit_p3_export_converts_the_float_pixels():
    image = Image.new("RGB", (8, 8), (255, 0, 0))
    attach(image, np.tile(np.array([1.0, 0.0, 0.0], np.float32), (8, 8, 1)))
    content, _ext, _mime = encode_export(image, {"format": "PNG", "bit_depth": 16, "color_space": "Display P3"})
    import cv2

    pixels = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)[..., ::-1]
    expected = convert_srgb_to_display_p3(np.array([1.0, 0.0, 0.0], np.float32))
    assert pixels.dtype == np.uint16
    assert np.allclose(pixels[0, 0] / 65535.0, expected, atol=1 / 65535 * 2)
    assert b"iCCP" in content


def test_settings_presets_and_cli_carry_the_color_space(tmp_path):
    assert validate_export_settings({})["color_space"] == "sRGB"
    assert validate_export_settings({"color_space": "Display P3"})["color_space"] == "Display P3"
    with pytest.raises(ValueError, match="color_space"):
        validate_export_settings({"color_space": "ProPhoto"})
    store = ExportPresetStore(tmp_path / "presets.json")
    store.save("Phone", {"format": "JPEG", "color_space": "Display P3"})
    assert store.load()["Phone"]["color_space"] == "Display P3"
    args = cli.build_parser().parse_args(["--input", "a.jpg", "--output", "b.jpg", "--color-space", "Display P3"])
    assert args.color_space == "Display P3" and set(COLOR_SPACES) == {"sRGB", "Display P3"}


@pytest.mark.slow
def test_the_export_panel_offers_display_p3(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import backend_client
    from tests.test_frontend_unified_app import APP_PATH, _fake_process_via_backend

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buffer, format="PNG")
    at.file_uploader[0].set_value(("red.png", buffer.getvalue(), "image/png"))
    at.run(timeout=60)
    picker = at.selectbox(key="exp-color-space")
    assert picker.value == "sRGB" and list(picker.options) == ["sRGB", "Display P3"]
    picker.set_value("Display P3")
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["exp-color-space"] == "Display P3"
