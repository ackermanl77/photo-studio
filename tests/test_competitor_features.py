"""The 2026-10-21 competitor rows: preset/Auto Enhance intensity (F30), Background (F31), Relight
(F32), markup arrows and highlighter (F33) and the provenance mark on AI-touched exports (F34)."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import background
from app.background import apply_background, grabcut_subject_mask, subject_mask
from app.edit_schema import OptionsError, parse_options, sanitize_recipe
from app.export import AI_TOOLS, DIGITAL_SOURCE_TYPE_AI_COMPOSITE, encode_export, provenance_xmp, validate_export_settings
from app.image_processor import PhotoProcessor, auto_enhance_recipe, scale_auto_enhance
from app.pipeline import ai_tools_for, run_pipeline
from app.presets import apply_preset
from app.relight import estimate_normals, light_vector, relight
from app.svg_export import layers_to_svg
from app.vector_shapes import HIGHLIGHTER_OPACITY, arrow_head_points, render_path_layer, render_shape_layer
from tests.synthetic_assets import photo_like


def _subject_on_plain_background(size=(200, 150)) -> Image.Image:
    image = Image.new("RGB", size, (40, 120, 200))
    ImageDraw.Draw(image).ellipse([70, 35, 130, 115], fill=(230, 180, 60))
    return image


# --- F30: intensity ---------------------------------------------------------------------------


def test_preset_amount_blends_between_the_photo_and_the_full_look():
    source = photo_like()
    untouched = np.asarray(source, dtype=np.float32)
    full = np.asarray(apply_preset(PhotoProcessor.from_image(source), "travel-clean").image, dtype=np.float32)
    half = np.asarray(apply_preset(PhotoProcessor.from_image(source), "travel-clean", 50).image, dtype=np.float32)
    none = np.asarray(apply_preset(PhotoProcessor.from_image(source), "travel-clean", 0).image, dtype=np.float32)
    assert np.array_equal(none, untouched)
    assert np.abs(half - (untouched + full) / 2).max() <= 1.0
    assert np.abs(full - untouched).mean() > 2 * np.abs(half - untouched).mean() * 0.9


def test_preset_amount_is_a_validated_option_that_reaches_the_pipeline():
    source = photo_like()
    full = run_pipeline(source.copy(), {"preset": "travel-clean"})
    half = run_pipeline(source.copy(), {"preset": "travel-clean", "preset_amount": 50})
    assert not np.array_equal(np.asarray(full), np.asarray(half))
    with pytest.raises(OptionsError):
        parse_options({"preset_amount": 150})


def test_auto_enhance_intensity_scales_every_change_toward_none():
    recipe = auto_enhance_recipe(photo_like())
    recipe.update(exposure_ev=1.0, curve_black=40, curve_white=200, curve_enabled=True, temperature=10, tint=-6, vibrance=20, denoise=3)
    assert scale_auto_enhance(recipe, 100) == recipe
    half = scale_auto_enhance(recipe, 50)
    assert (half["exposure_ev"], half["curve_black"], half["curve_white"], half["temperature"], half["tint"], half["vibrance"]) == (0.5, 20, 228, 5, -3, 10)
    assert half["denoise"] == 3
    zero = scale_auto_enhance(recipe, 0)
    assert (zero["exposure_ev"], zero["curve_enabled"], zero["temperature"], zero["vibrance"], zero["denoise"]) == (0.0, False, 0, 0, 0)


# --- F31: background ----------------------------------------------------------------------------


def test_grabcut_finds_a_centred_subject():
    mask = grabcut_subject_mask(_subject_on_plain_background())
    assert mask.shape == (150, 200)
    assert mask[75, 100] > 0.8  # the subject
    assert mask[10, 10] < 0.2 and mask[140, 190] < 0.2  # the background


def test_subject_mask_is_cached_by_content(monkeypatch):
    calls = []
    monkeypatch.setattr(background, "_model_available", lambda: False)
    monkeypatch.setattr(background, "grabcut_subject_mask", lambda image: calls.append(1) or np.ones((image.height, image.width), np.float32))
    image = _subject_on_plain_background((64, 48))
    subject_mask(image)
    subject_mask(image.copy())
    assert len(calls) == 1
    with pytest.raises(ValueError):
        subject_mask(image, "Sky")


def test_background_color_and_blur_leave_the_subject_alone():
    array = np.asarray(_subject_on_plain_background(), dtype=np.float32) / 255.0
    mask = np.zeros(array.shape[:2], np.float32)
    mask[40:110, 75:125] = 1.0
    filled = apply_background(array, mask, "Color", color="#FF0000")
    assert np.allclose(filled[5, 5], (1.0, 0.0, 0.0))
    assert np.allclose(filled[75, 100], array[75, 100])
    blurred = apply_background(array, mask, "Blur", 80)
    assert np.allclose(blurred[75, 100], array[75, 100])
    assert np.array_equal(apply_background(array, mask, "Remove"), array)


def test_background_remove_gives_a_transparent_result(monkeypatch):
    monkeypatch.setattr(background, "_model_available", lambda: False)
    result = run_pipeline(_subject_on_plain_background(), {"background_mode": "Remove"})
    assert result.mode == "RGBA"
    alpha = np.asarray(result.getchannel("A"))
    assert alpha[75, 100] > 200 and alpha[5, 5] < 50
    png, _extension, _mime = encode_export(result, {"format": "PNG"})
    assert Image.open(io.BytesIO(png)).mode == "RGBA"


def test_background_color_option_is_validated():
    assert parse_options({"background_mode": "Color", "background_color": "#00ff00"})["background_color"] == "#00FF00"
    with pytest.raises(OptionsError):
        parse_options({"background_color": "green"})
    with pytest.raises(OptionsError):
        parse_options({"background_mode": "Sky"})
    assert sanitize_recipe({"background_color": "nope"})["background_color"] == "#FFFFFF"


# --- F32: relight -------------------------------------------------------------------------------


def test_relight_brightens_the_side_facing_the_light():
    # A bright dome in the middle: its left slope faces a light from the left (azimuth 180).
    yy, xx = np.mgrid[0:120, 0:120].astype(np.float32)
    dome = np.clip(1.0 - np.hypot(xx - 60, yy - 60) / 50.0, 0.0, 1.0) * 0.6 + 0.2
    array = np.dstack([dome] * 3).astype(np.float32)
    lit = relight(array, 100, azimuth_deg=180, elevation_deg=30, ambient=20)
    assert lit[60, 35].mean() > array[60, 35].mean()  # left slope: facing the light
    assert lit[60, 85].mean() < array[60, 85].mean()  # right slope: facing away
    assert np.array_equal(relight(array, 0), array)


def test_relight_normals_are_unit_vectors_and_light_points_where_asked():
    normals = estimate_normals(np.asarray(photo_like(), dtype=np.float32) / 255.0)
    assert np.allclose(np.linalg.norm(normals, axis=2), 1.0, atol=1e-4)
    assert np.allclose(light_vector(0, 90), (0, 0, 1), atol=1e-6)
    assert light_vector(180, 0)[0] < -0.99  # from the left


def test_relight_reaches_the_pipeline():
    source = photo_like()
    assert not np.array_equal(np.asarray(run_pipeline(source.copy(), {"relight_amount": 80})), np.asarray(run_pipeline(source.copy(), {})))


# --- F33: markup --------------------------------------------------------------------------------


def test_arrow_head_points_away_from_the_tail():
    tip, left, right = arrow_head_points((100.0, 50.0), (0.0, 50.0), 4)
    assert tip == (100.0, 50.0)
    assert left[0] < 100 and right[0] < 100 and left[1] != right[1]


def test_line_arrow_heads_widen_the_end_of_the_line():
    params = {"shape_type": "Line", "x_pct": 50, "y_pct": 50, "width_pct": 80, "line_thickness": 4, "fill_color": (255, 0, 0)}
    plain = np.asarray(render_shape_layer((200, 100), params).getchannel("A"))
    arrow = np.asarray(render_shape_layer((200, 100), {**params, "arrow_end": True}).getchannel("A"))
    near_end = slice(170, 178)
    assert (arrow[:, near_end] > 0).sum() > (plain[:, near_end] > 0).sum()


def test_highlighter_path_is_translucent():
    points = [{"x_pct": 10, "y_pct": 50}, {"x_pct": 90, "y_pct": 50}]
    params = {"points": points, "closed": False, "stroke_width": 12, "stroke_color": (255, 230, 0)}
    opaque = render_path_layer((200, 100), params)
    marker = render_path_layer((200, 100), {**params, "highlighter": True})
    assert opaque.getpixel((100, 50))[3] == 255
    assert marker.getpixel((100, 50))[3] == round(255 * HIGHLIGHTER_OPACITY)


def test_svg_export_carries_arrow_markers_and_the_highlighter():
    layers = [
        {"kind": "shape", "shape_type": "Line", "arrow_end": True, "fill_color": "#FF0000", "visible": True, "opacity": 1.0},
        {"kind": "path", "points": [{"x_pct": 10, "y_pct": 10}, {"x_pct": 90, "y_pct": 90}], "closed": False,
         "arrow_start": True, "highlighter": True, "visible": True, "opacity": 1.0},
    ]
    svg = layers_to_svg(layers, (200, 100))
    assert "<marker" in svg and 'marker-end="url(#' in svg and 'marker-start="url(#' in svg
    assert f'stroke-opacity="{HIGHLIGHTER_OPACITY:g}"' in svg


# --- F34: provenance ----------------------------------------------------------------------------


def test_no_provenance_without_ai_tools():
    assert provenance_xmp(()) is None
    jpeg, _extension, _mime = encode_export(photo_like(), {"format": "JPEG"})
    assert b"DigitalSourceType" not in jpeg


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "TIFF"])
def test_ai_touched_exports_carry_the_iptc_digital_source_type(fmt):
    content, _extension, _mime = encode_export(photo_like(), {"format": fmt, "metadata": "none", "ai_tools": ["AI Upscale", "AI mask"]})
    assert DIGITAL_SOURCE_TYPE_AI_COMPOSITE.encode() in content
    assert b"AI tools used: AI Upscale, AI mask" in content


@pytest.mark.parametrize("fmt", ["PNG", "TIFF"])
def test_16_bit_exports_carry_the_provenance_too(fmt):
    content, _extension, _mime = encode_export(photo_like(), {"format": fmt, "bit_depth": 16, "ai_tools": ["AI background"]})
    assert DIGITAL_SOURCE_TYPE_AI_COMPOSITE.encode() in content


def test_ai_tools_setting_is_validated_and_derived_from_the_options(monkeypatch):
    assert validate_export_settings({"ai_tools": ["AI mask", "AI Upscale"]})["ai_tools"] == ("AI Upscale", "AI mask")
    with pytest.raises(ValueError):
        validate_export_settings({"ai_tools": ["Magic"]})
    assert set(AI_TOOLS) == {"AI Upscale", "AI mask", "AI background"}
    monkeypatch.setattr(background, "_model_available", lambda: True)
    assert ai_tools_for({"ai_upscale": True, "background_mode": "Blur"}) == ["AI Upscale", "AI background"]
    monkeypatch.setattr(background, "_model_available", lambda: False)
    assert ai_tools_for({"background_mode": "Blur"}) == []
