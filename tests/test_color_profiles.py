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


# --- Display P3 as the working space of a P3 export (B14) ------------------------------------------
# Before B14 the pipeline converted every source into sRGB first (clipping), so a "Display P3"
# export could never hold a colour outside sRGB: the most saturated P3 green it could write was
# sRGB's green converted, about (117, 251, 76).

def _p3_variant_icc() -> bytes:
    """Our Display P3 profile with a different creator field: the same colour space, but not the
    exact bytes app.color_profiles recognises -- so it goes through the littlecms conversion."""
    icc = bytearray(display_p3_icc_bytes())
    icc[80:84] = b"TEST"
    return bytes(icc)


def _process_bytes(content: bytes, name: str, export: dict, **options) -> bytes:
    import json

    from app.backend import app

    response = app.test_client().post(
        "/process",
        data={"image": (io.BytesIO(content), name), "options": json.dumps({**options, "export": export})},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.data[:300]
    return response.data


def _process(content: bytes, name: str, export: dict, **options) -> Image.Image:
    return Image.open(io.BytesIO(_process_bytes(content, name, export, **options)))


def _png(image: Image.Image, icc: bytes | None = None) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG", **({"icc_profile": icc} if icc else {}))
    return buffer.getvalue()


@pytest.mark.parametrize("icc", [display_p3_icc_bytes(), _p3_variant_icc()], ids=["own-profile", "other-p3-profile"])
def test_a_p3_export_keeps_colours_outside_srgb(icc):
    source = _png(Image.new("RGB", (16, 16), (0, 255, 0)), icc)  # P3's own green: outside sRGB
    out = _process(source, "p3.png", {"format": "PNG", "color_space": "Display P3"})
    assert out.info["icc_profile"] == display_p3_icc_bytes()
    green = np.asarray(out.convert("RGB"))[8, 8].astype(int)
    assert np.abs(green - [0, 255, 0]).max() <= 1  # not clipped to sRGB's (117, 251, 76)
    # The sRGB export of the same file still clips it, as it must.
    srgb = np.asarray(_process(source, "p3.png", {"format": "PNG"}).convert("RGB"))[8, 8].astype(int)
    assert srgb[0] == 0 and srgb[2] == 0 and srgb[1] >= 250


def test_an_untagged_source_exports_to_p3_exactly_as_before():
    out = _process(_png(Image.new("RGB", (16, 16), (255, 0, 0))), "red.png", {"format": "PNG", "color_space": "Display P3"})
    assert out.info["icc_profile"] == display_p3_icc_bytes()
    assert np.asarray(out.convert("RGB"))[8, 8].tolist() == [234, 51, 35]  # sRGB red in P3, converted once


def test_edits_run_in_the_p3_working_space():
    # Exposure on P3 green stays P3 green (the edit sees P3 values, nothing clips to sRGB first).
    source = _png(Image.new("RGB", (16, 16), (0, 200, 0)), display_p3_icc_bytes())
    out = _process(source, "p3.png", {"format": "PNG", "color_space": "Display P3"}, exposure_ev=0.3)
    green = np.asarray(out.convert("RGB"))[8, 8].astype(int)
    assert green[0] <= 2 and green[2] <= 2 and green[1] > 200


def test_a_16_bit_p3_export_keeps_colours_outside_srgb():
    import cv2

    from app.export import _png16

    green = np.zeros((8, 8, 3), np.uint16)
    green[..., 1] = 60000  # a P3 green outside sRGB, at 16 bits
    source = _png16(green, None, display_p3_icc_bytes())
    content = _process_bytes(source, "p3-16.png", {"format": "PNG", "bit_depth": 16, "color_space": "Display P3"})
    pixels = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)[..., ::-1]
    assert pixels.dtype == np.uint16
    assert pixels[4, 4, 0] <= 200 and pixels[4, 4, 2] <= 200 and abs(int(pixels[4, 4, 1]) - 60000) <= 300
    assert display_p3_icc_bytes() in _iccp(content)


def _iccp(png: bytes) -> bytes:
    import zlib

    at = png.index(b"iCCP")
    length = int.from_bytes(png[at - 4:at], "big")
    chunk = png[at + 4:at + 4 + length]
    return zlib.decompress(chunk[chunk.index(b"\x00") + 2:])


def test_raw_is_decoded_to_p3_for_a_p3_export():
    pytest.importorskip("rawpy")
    from tests.synthetic_assets import dng_bytes

    from app.pipeline import decode_source
    from app.raw_import import decode_raw_array

    content = dng_bytes()
    image = decode_source(content, "shot.dng", working_space="Display P3")
    assert image.info["icc_profile"] == display_p3_icc_bytes()
    expected = decode_raw_array(io.BytesIO(content), color_space="Display P3")
    assert np.array_equal(np.asarray(image), expected)
    assert not np.array_equal(expected, decode_raw_array(io.BytesIO(content)))  # really a different space
    # Exported without edits, the P3 decode reaches the file unchanged (no second sRGB->P3 conversion).
    out = _process(content, "shot.dng", {"format": "PNG", "color_space": "Display P3"})
    assert np.abs(np.asarray(out.convert("RGB")).astype(int) - expected).max() <= 1
