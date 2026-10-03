"""B4: the float32 high-precision pipeline path and genuine 16-bit PNG/TIFF output.

The 8-bit path must stay byte-identical (tests/test_parity_golden.py guards that); here: the
16-bit path decodes 16-bit/RAW sources at full precision, computes tone/colour/masks/geometry/crop
in float without re-quantizing, matches the 8-bit preview to within rounding, and writes real
16-bit files with their metadata."""
from __future__ import annotations

import io
import json
import sys
import types

import cv2
import numpy as np
import pytest
from PIL import Image

from app import cli, raw_import
from app.backend import app as flask_app
from app.export import encode_export, validate_export_settings, write_export
from app.geometry import apply_geometry
from app.high_bit import _orient, attach, attached, decode_high_bit
from app.image_processor import PhotoProcessor
from app.pipeline import decode_source, run_pipeline
from app.professional_color import srgb_icc_bytes
from tests.synthetic_assets import dng_bytes, photo_like


def _gradient16(width: int = 1024, height: int = 24) -> np.ndarray:
    """A smooth 16-bit RGB ramp (the shape banding shows up in), RGB order."""
    ramp = np.linspace(0.2, 0.45, width, dtype=np.float64)
    rgb = np.stack([ramp, ramp * 0.9, ramp * 0.8], axis=-1)
    return np.rint(np.broadcast_to(rgb, (height, width, 3)) * 65535).astype(np.uint16)


def _png16(array16: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", np.ascontiguousarray(array16[..., ::-1]))
    assert ok
    return encoded.tobytes()


def _decode16(content: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    return decoded[..., ::-1] if decoded.ndim == 3 else decoded


TONE_AND_COLOR = {
    "exposure_ev": 0.7, "contrast": 1.35, "highlights": -40.0, "shadows": 30.0, "temperature": 12.0,
    "tint": -6.0, "saturation": 1.2, "curve": [[0, 10], [128, 150], [255, 245]], "clarity": 20.0,
}


# --- decode ----------------------------------------------------------------------------------------

def test_16_bit_png_and_tiff_decode_at_full_precision_and_8_bit_files_do_not():
    import tifffile

    array16 = _gradient16()
    decoded = decode_high_bit(_png16(array16), "ramp.png")
    assert decoded.dtype == np.float32 and decoded.shape == (24, 1024, 3)
    assert np.abs(decoded - array16 / 65535.0).max() < 1e-6
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array16, photometric="rgb")
    assert np.abs(decode_high_bit(buffer.getvalue(), "ramp.tif") - array16 / 65535.0).max() < 1e-6
    gray = np.rint(np.linspace(0, 65535, 300)).astype(np.uint16)[None, :].repeat(4, axis=0)
    ok, gray_png = cv2.imencode(".png", gray)
    gray_decoded = decode_high_bit(gray_png.tobytes(), "gray.png")
    assert gray_decoded.shape == (4, 300, 3) and len(np.unique(gray_decoded[0, :, 0])) == 300
    eight_bit = io.BytesIO()
    photo_like((32, 24)).save(eight_bit, format="PNG")
    assert decode_high_bit(eight_bit.getvalue(), "photo.png") is None


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_is_applied_to_the_16_bit_decode_like_the_8_bit_one(orientation):
    from PIL import ImageOps

    source = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    image = Image.fromarray(source)
    exif = image.getexif()
    exif[0x0112] = orientation
    image.info["exif"] = exif.tobytes()
    expected = np.asarray(ImageOps.exif_transpose(image))
    assert np.array_equal(_orient(source, orientation), expected)


def test_decode_source_high_bit_returns_the_quantized_image_with_the_float_pixels_attached():
    content = _png16(_gradient16())
    image = decode_source(content, "ramp.png", high_bit=True)
    array = attached(image)
    assert image.size == (1024, 24) and array is not None
    assert np.abs(np.asarray(image, dtype=np.float32) / 255.0 - array).max() <= 0.5 / 255 + 1e-6
    assert attached(decode_source(content, "ramp.png")) is None  # the normal decode is untouched


class _FakeRaw:
    def __init__(self, seen):
        self.seen = seen

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def postprocess(self, **kwargs):
        self.seen.append(kwargs["output_bps"])
        ramp = np.linspace(0, 1, 40)[None, :, None] * np.ones((6, 1, 3))
        return np.rint(ramp * (65535 if kwargs["output_bps"] == 16 else 255)).astype(np.uint16 if kwargs["output_bps"] == 16 else np.uint8)


def test_raw_is_decoded_with_16_bits_per_channel_for_a_16_bit_export(monkeypatch):
    seen: list[int] = []
    fake = types.ModuleType("rawpy")
    fake.LibRawError = type("LibRawError", (Exception,), {})
    fake.imread = lambda source: _FakeRaw(seen)
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    array = decode_high_bit(b"raw", "photo.NEF")
    assert seen == [16] and array.shape == (6, 40, 3) and array.max() == pytest.approx(1.0)
    raw_import.open_bytes(b"raw", "photo.NEF")
    assert seen[-1] == 8  # the interactive decode stays 8-bit


def test_a_real_dng_decodes_at_16_bits_matching_the_8_bit_decode():
    pytest.importorskip("rawpy")
    eight = np.asarray(decode_source(dng_bytes(), "tiny.dng"), dtype=np.float32) / 255.0
    sixteen = decode_high_bit(dng_bytes(), "tiny.dng")
    assert sixteen.shape == eight.shape
    assert np.abs(sixteen - eight).mean() < 0.02


# --- the float pipeline ----------------------------------------------------------------------------

def test_the_default_8_bit_path_attaches_nothing():
    result = run_pipeline(photo_like((64, 48)), dict(TONE_AND_COLOR))
    assert attached(result) is None


def test_the_16_bit_pipeline_matches_the_8_bit_preview_to_within_rounding():
    source = photo_like((160, 120))
    options = {
        **TONE_AND_COLOR, "vibrance": 15.0, "hsl_bands": {"Blue": {"hue": 10, "sat": 20, "lum": -10}},
        "cg_shadows_hue": 200.0, "cg_shadows_intensity": 30.0, "texture": 25.0, "sharpen": True,
        "rotate": 90, "straighten": 3.0, "lens_distortion": -20.0, "lens_vignette_amount": -25.0,
        "lens_perspective_v": 10.0, "lens_chromatic_aberration": 30.0, "constrain_crop": True,
        "mask_exposure_ev": 0.5, "effects_vignette_amount": -30.0, "aspect": "4:5",
    }
    mask = Image.new("L", source.size[::-1], 0)
    mask.paste(255, (0, 0, 60, 80))
    eight = run_pipeline(source, options, local_mask=mask)
    high = run_pipeline(source, {**options, "bit_depth": 16}, local_mask=mask)
    array = attached(high)
    assert array is not None and high.size == eight.size
    assert np.array_equal(np.asarray(high), np.clip(np.rint(array * 255), 0, 255).astype(np.uint8))
    # The 8-bit steps truncate (astype(uint8) floors) after each operation, so a long 8-bit chain
    # drifts ~0.5 level darker per step; the float path doesn't. What remains is that small,
    # near-uniform offset -- not a different look.
    signed = array * 255.0 - np.asarray(eight, dtype=np.float32)
    assert 0.0 <= signed.mean() < 4.0
    assert np.abs(signed - signed.mean()).mean() < 2.5  # ~1 % -- sharpening amplifies rounding
    assert np.percentile(np.abs(signed), 99) < 16


def test_the_16_bit_pipeline_keeps_a_smooth_gradient_free_of_8_bit_banding():
    source = decode_source(_png16(_gradient16()), "ramp.png", high_bit=True)
    options = dict(TONE_AND_COLOR)
    high = attached(run_pipeline(source, {**options, "bit_depth": 16}))
    eight = np.asarray(run_pipeline(decode_source(_png16(_gradient16()), "ramp.png"), options))
    high_levels = len(np.unique(np.rint(high[12, :, 1] * 65535)))
    eight_levels = len(np.unique(eight[12, :, 1]))
    assert eight_levels < 140  # 8 bits across this ramp: a staircase
    assert high_levels > 4 * eight_levels
    # ...and it rises monotonically, with no step bigger than a fraction of one 8-bit level.
    steps = np.diff(high[12, 20:-20, 1])
    assert steps.min() > -1e-4 and steps.max() < 0.5 / 255


def test_geometry_on_float_arrays_matches_the_8_bit_geometry():
    source = photo_like((120, 90))
    options = {"rotate": 270, "flip_horizontal": True, "straighten": -4.0, "lens_distortion": 30.0, "lens_perspective_h": -15.0, "lens_chromatic_aberration": -40.0, "lens_vignette_amount": 20.0, "constrain_crop": True}
    eight = np.asarray(apply_geometry(source, options), dtype=np.float32)
    high = apply_geometry(np.asarray(source, dtype=np.float32) / 255.0, options)
    assert high.shape == eight.shape and high.dtype == np.float32
    assert np.abs(high * 255.0 - eight).mean() < 1.0


def test_steps_without_a_float_version_carry_their_change_onto_the_float_data():
    processor = PhotoProcessor.from_image(photo_like((64, 48)))
    processor.enable_high_precision(np.full((48, 64, 3), 0.5 + 0.3 / 255, dtype=np.float32))
    processor.apply_denoise(3)  # an 8-bit-only step
    assert processor.hdr is not None
    assert np.abs(processor.hdr * 255.0 - np.asarray(processor.image, dtype=np.float32)).max() <= 0.5 + 1e-3
    processor.crop(4, 4, 30, 20)
    assert processor.hdr.shape == (16, 26, 3) and processor.image.size == (26, 16)


# --- 16-bit export ---------------------------------------------------------------------------------

@pytest.mark.parametrize("format_name", ["PNG", "TIFF"])
def test_16_bit_export_writes_genuine_16_bit_files_with_metadata(format_name):
    image = Image.new("RGB", (300, 10))
    attach(image, (np.linspace(0, 1, 300, dtype=np.float32)[None, :, None] * np.ones((10, 1, 3), np.float32)))
    exif = Image.Exif()
    exif[0x010F] = "TestMaker"
    exif[0x8298] = "Me"
    content, extension, _mime = encode_export(image, {"format": format_name, "bit_depth": 16}, exif=exif.tobytes(), icc=srgb_icc_bytes())
    pixels = _decode16(content)
    assert pixels.dtype == np.uint16 and pixels.shape == (10, 300, 3)
    assert len(np.unique(pixels[0, :, 0])) == 300 and pixels.max() == 65535
    reopened = Image.open(io.BytesIO(content))
    assert reopened.info.get("icc_profile") == srgb_icc_bytes()
    assert reopened.getexif().get(0x010F) == "TestMaker" and reopened.getexif().get(0x8298) == "Me"
    assert extension == (".png" if format_name == "PNG" else ".tif")


def test_16_bit_export_resizes_sharpens_and_strips_metadata_like_the_8_bit_one():
    image = photo_like((200, 100))
    content = encode_export(image, {"format": "PNG", "bit_depth": 16, "long_edge": 100, "sharpen": "Screen", "metadata": "none"}, exif=b"Exif\x00\x00" + Image.Exif().tobytes()[6:])[0]
    pixels = _decode16(content)
    assert pixels.shape == (50, 100, 3) and pixels.dtype == np.uint16
    eight = np.asarray(Image.open(io.BytesIO(encode_export(image, {"format": "PNG", "long_edge": 100, "sharpen": "Screen"})[0])), dtype=np.float32)
    assert np.abs(pixels / 257.0 - eight).mean() < 1.5


def test_16_bit_needs_png_or_tiff_and_write_export_falls_back_for_jpeg(tmp_path):
    with pytest.raises(ValueError, match="16-bit"):
        validate_export_settings({"format": "JPEG", "bit_depth": 16})
    with pytest.raises(ValueError):
        validate_export_settings({"format": "PNG", "bit_depth": 12})
    assert validate_export_settings({})["bit_depth"] == 8
    out = write_export(photo_like((20, 10)), tmp_path / "x.jpg", {"bit_depth": 16})
    assert Image.open(out).format == "JPEG"


def test_backend_16_bit_export_returns_a_16_bit_file_from_a_16_bit_source():
    client = flask_app.test_client()
    options = {"exposure_ev": 0.3, "contrast": 1.2, "export": {"format": "TIFF", "bit_depth": 16}}
    response = client.post("/process", data={"options": json.dumps(options), "image": (io.BytesIO(_png16(_gradient16())), "ramp.png")}, content_type="multipart/form-data")
    assert response.status_code == 200, response.get_data(as_text=True)
    pixels = _decode16(response.data)
    assert pixels.dtype == np.uint16 and len(np.unique(pixels[12, :, 1])) > 500
    plain = client.post("/process", data={"options": json.dumps({"bit_depth": 16}), "image": (io.BytesIO(_png16(_gradient16())), "ramp.png")}, content_type="multipart/form-data")
    assert plain.status_code == 200 and _decode16(plain.data).dtype == np.uint16


def test_cli_bit_depth_16_writes_a_16_bit_tiff(tmp_path, monkeypatch):
    source = tmp_path / "ramp.png"
    source.write_bytes(_png16(_gradient16()))
    output = tmp_path / "out.tif"
    monkeypatch.setattr(sys, "argv", ["cli.py", "--input", str(source), "--output", str(output), "--bit-depth", "16", "--exposure-ev", "0.5"])
    cli.main()
    pixels = _decode16(output.read_bytes())
    assert pixels.dtype == np.uint16 and len(np.unique(pixels[12, :, 1])) > 500


@pytest.mark.slow
def test_the_export_panel_offers_16_bit_for_png_and_tiff_and_renders_it_on_the_backend(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from app import backend_client
    from app.backend import process_image
    from tests.test_frontend_unified_app import APP_PATH, _FakeFileStorage, _fake_process_via_backend

    exports = []

    def fake_export(image_bytes_, filename, options, settings, *args, **kwargs):
        exports.append(settings)
        result = process_image(_FakeFileStorage(filename, image_bytes_), {**options, "export": settings})
        return encode_export(result, settings)[0]

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    monkeypatch.setattr(backend_client, "export_via_backend", fake_export)
    captured = {}
    original = st.download_button

    def capture(label, data, *args, **kwargs):
        if kwargs.get("key") == "exp-download-btn":
            captured["data"] = data
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("ramp.png", _png16(_gradient16(256, 16)), "image/png"))
    at.run(timeout=60)
    assert at.selectbox(key="exp-bit-depth").disabled  # JPEG is the default format
    at.selectbox(key="exp-fmt").set_value("PNG")
    at.run(timeout=30)
    at.run(timeout=30)
    assert not at.selectbox(key="exp-bit-depth").disabled
    at.selectbox(key="exp-bit-depth").set_value(16)
    at.run(timeout=30)
    assert not at.exception
    pixels = _decode16(captured["data"]())
    assert exports and exports[-1]["bit_depth"] == 16
    assert pixels.dtype == np.uint16 and len(np.unique(pixels[8, :, 0])) > 200
