import io

import numpy as np
import pytest
from PIL import Image

from app.professional_color import (
    HSL_BAND_HUES,
    ICC_TRANSFORM_ERRORS,
    _highlight_mismatches,
    apply_color_grading,
    apply_hsl_mixer,
    apply_vibrance,
    clipping_warning_overlay,
    estimate_white_balance,
    from_float_rgb,
    gamut_warning_overlay,
    hsl_to_rgb,
    normalize_to_srgb,
    rgb_to_hsl,
    srgb_icc_bytes,
    to_float_rgb,
)


def test_float_rgb_round_trip_is_bounded():
    image = Image.new("RGB", (8, 6), (40, 100, 220))
    values = to_float_rgb(image)
    restored = from_float_rgb(values)
    assert values.min() >= 0 and values.max() <= 1
    assert np.array_equal(np.asarray(image), np.asarray(restored))


def test_srgb_icc_bytes_returns_a_profile_littlecms_can_reload():
    from PIL import ImageCms

    data = srgb_icc_bytes()
    assert isinstance(data, bytes) and len(data) > 100
    reloaded = ImageCms.ImageCmsProfile(io.BytesIO(data))
    assert ImageCms.getProfileName(reloaded).strip()


def test_normalize_to_srgb_is_a_near_identity_for_an_already_srgb_source():
    """Converting an sRGB-tagged image through its own profile into sRGB again should reproduce
    the same pixel values (modulo tiny floating-point rounding) -- the common case, since most
    photos are already sRGB."""
    image = Image.new("RGB", (6, 6), (180, 90, 40))
    result = normalize_to_srgb(image, srgb_icc_bytes())
    before = np.asarray(image, dtype=np.int16)
    after = np.asarray(result, dtype=np.int16)
    assert np.abs(before - after).max() <= 2


def test_icc_transform_errors_covers_every_failure_mode_normalize_to_srgb_can_raise():
    """A shared constant, not duplicated per-callsite tuples -- app/backend.py and
    app/frontend_unified_app.py both import this instead of each hand-writing their own copy of
    "every exception a malformed ICC profile can raise" (a real reuse gap found in review: they'd
    drifted out of sync risk if littlecms started raising something new and only one copy got
    updated). ImageCms.open() raises OSError for unparseable bytes; profileToProfile() raises
    ImageCms.PyCMSError when it can open a profile but can't build a transform from it."""
    from PIL import ImageCms

    assert OSError in ICC_TRANSFORM_ERRORS
    assert ImageCms.PyCMSError in ICC_TRANSFORM_ERRORS

    with pytest.raises(ICC_TRANSFORM_ERRORS):
        normalize_to_srgb(Image.new("RGB", (4, 4), (1, 2, 3)), b"not a real icc profile")


def test_gamut_warning_overlay_shows_no_warnings_for_an_already_srgb_source():
    image = Image.new("RGB", (10, 10), (220, 40, 200))
    result = gamut_warning_overlay(image, srgb_icc_bytes())
    assert np.array_equal(np.asarray(result), np.asarray(image))


def test_highlight_mismatches_flags_only_pixels_beyond_the_threshold():
    base = np.zeros((2, 3, 3), dtype=np.uint8)
    comparison = base.copy()
    comparison[0, 0] = [50, 0, 0]  # 50 > threshold -- should be flagged
    comparison[0, 1] = [5, 0, 0]  # 5 <= threshold -- should not be flagged

    result = _highlight_mismatches(base, comparison, warning_color=(255, 0, 255), threshold=24)

    assert list(result[0, 0]) == [255, 0, 255]
    assert list(result[0, 1]) == [0, 0, 0]
    assert list(result[1, 2]) == [0, 0, 0]


def test_rgb_to_hsl_matches_known_colors():
    # Pure red, pure green, and a mid-gray -- easy to verify by hand against the standard HSL definition.
    rgb = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]]], dtype=np.float32)
    hsl = rgb_to_hsl(rgb)
    np.testing.assert_allclose(hsl[0, 0], [0.0, 1.0, 0.5], atol=1e-5)
    np.testing.assert_allclose(hsl[0, 1], [120.0, 1.0, 0.5], atol=1e-5)
    np.testing.assert_allclose(hsl[0, 2], [0.0, 0.0, 0.5], atol=1e-5)


def test_hsl_round_trips_through_rgb_for_random_colors():
    rng = np.random.default_rng(0)
    rgb = rng.random((20, 20, 3)).astype(np.float32)
    restored = hsl_to_rgb(rgb_to_hsl(rgb))
    np.testing.assert_allclose(restored, rgb, atol=1e-4)


def test_apply_hsl_mixer_is_a_no_op_with_no_or_zero_adjustments():
    image = Image.new("RGB", (10, 10), (200, 60, 60))
    assert apply_hsl_mixer(image, {}) is image
    result = apply_hsl_mixer(image, {"Red": {"hue": 0, "sat": 0, "lum": 0}})
    assert np.array_equal(np.asarray(result), np.asarray(image))


def test_apply_hsl_mixer_only_affects_the_targeted_band():
    """A saturation cut on Red should desaturate a red region far more than a blue region on the
    opposite side of the hue wheel, exercising the per-pixel hue-weighted blending."""
    image = Image.new("RGB", (2, 20))
    for y in range(20):
        color = (220, 40, 40) if y < 10 else (40, 40, 220)  # top half red, bottom half blue
        for x in range(2):
            image.putpixel((x, y), color)

    result = apply_hsl_mixer(image, {"Red": {"hue": 0, "sat": -100, "lum": 0}})
    before = np.asarray(image, dtype=np.float32)
    after = np.asarray(result, dtype=np.float32)

    red_region_change = np.abs(after[:10] - before[:10]).mean()
    blue_region_change = np.abs(after[10:] - before[10:]).mean()
    assert red_region_change > 5.0
    assert blue_region_change < 1.0


def test_apply_hsl_mixer_hue_shift_moves_a_color_toward_its_neighbor():
    """Shifting Red's hue toward Orange (positive shift, per HSL_BAND_HUES ordering) on a pure red
    image should measurably increase the green channel relative to blue, matching a red->orange move."""
    image = Image.new("RGB", (4, 4), (220, 30, 30))
    result = apply_hsl_mixer(image, {"Red": {"hue": 100, "sat": 0, "lum": 0}})
    r, g, b = np.asarray(result, dtype=np.float32).reshape(-1, 3).mean(axis=0)
    assert g - b > 10.0


def test_hsl_band_hues_cover_all_eight_named_bands():
    from app.professional_color import HSL_BAND_ORDER

    assert set(HSL_BAND_HUES) == set(HSL_BAND_ORDER)
    assert len(HSL_BAND_ORDER) == 8


def test_color_grading_is_a_no_op_at_zero_intensity():
    image = Image.new("RGB", (10, 10), (120, 130, 140))
    assert apply_color_grading(image) is image
    result = apply_color_grading(image, shadows_hue=200, midtones_hue=10, highlights_hue=300)
    assert np.array_equal(np.asarray(result), np.asarray(image))


def test_color_grading_tints_shadows_without_moving_highlights():
    """A near-black region and a near-white region on the same image -- cranking Shadows
    intensity with a blue hue should visibly shift the dark region toward blue while leaving the
    bright region close to untouched, exercising the per-pixel luminance-weighted blend."""
    image = Image.new("RGB", (4, 20))
    for y in range(20):
        color = (10, 10, 10) if y < 10 else (245, 245, 245)
        for x in range(4):
            image.putpixel((x, y), color)

    result = apply_color_grading(image, shadows_hue=220, shadows_intensity=100)
    before = np.asarray(image, dtype=np.float32)
    after = np.asarray(result, dtype=np.float32)

    shadow_region_change = np.abs(after[:10] - before[:10]).mean()
    highlight_region_change = np.abs(after[10:] - before[10:]).mean()
    assert shadow_region_change > 10.0
    assert highlight_region_change < 1.0
    # Blue hue (~220 degrees): blue channel should have gained relative to red in the shadow region.
    shadow_after = after[:10].reshape(-1, 3).mean(axis=0)
    assert shadow_after[2] > shadow_after[0]


def test_color_grading_tints_highlights_without_moving_shadows():
    image = Image.new("RGB", (4, 20))
    for y in range(20):
        color = (10, 10, 10) if y < 10 else (245, 245, 245)
        for x in range(4):
            image.putpixel((x, y), color)

    result = apply_color_grading(image, highlights_hue=30, highlights_intensity=100)
    before = np.asarray(image, dtype=np.float32)
    after = np.asarray(result, dtype=np.float32)

    shadow_region_change = np.abs(after[:10] - before[:10]).mean()
    highlight_region_change = np.abs(after[10:] - before[10:]).mean()
    assert highlight_region_change > 10.0
    assert shadow_region_change < 1.0


def test_color_grading_balance_shifts_the_shadow_highlight_split():
    """A mid-gray pixel: with balance pushed hard toward highlights (positive), it should fall
    inside the shadows range instead of midtones/highlights -- so a strong shadow tint moves it,
    while the same image at balance=0 is barely touched by that same shadow tint (mid-gray sits
    close to the default 0.5 pivot, near-equally weighted toward midtones)."""
    image = Image.new("RGB", (4, 4), (128, 128, 128))

    neutral = apply_color_grading(image, shadows_hue=220, shadows_intensity=100, balance=0)
    shifted = apply_color_grading(image, shadows_hue=220, shadows_intensity=100, balance=100)

    before = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
    neutral_change = np.abs(np.asarray(neutral, dtype=np.float32).mean(axis=(0, 1)) - before).sum()
    shifted_change = np.abs(np.asarray(shifted, dtype=np.float32).mean(axis=(0, 1)) - before).sum()
    assert shifted_change > neutral_change


def test_clipping_warning_overlay_flags_pure_black_and_white_pixels():
    image = Image.new("RGB", (3, 1), (128, 128, 128))
    image.putpixel((0, 0), (0, 0, 0))
    image.putpixel((2, 0), (255, 255, 255))

    result = np.asarray(clipping_warning_overlay(image, shadow_color=(0, 80, 255), highlight_color=(255, 40, 40)))

    assert list(result[0, 0]) == [0, 80, 255]
    assert list(result[0, 1]) == [128, 128, 128]
    assert list(result[0, 2]) == [255, 40, 40]


def test_clipping_warning_overlay_flags_a_single_clipped_channel():
    """A pixel with only its blue channel blown out still counts as a whole-pixel highlight
    warning -- matching Lightroom's "any channel clips" convention rather than requiring all
    three."""
    image = Image.new("RGB", (1, 1), (100, 100, 255))
    result = np.asarray(clipping_warning_overlay(image, highlight_color=(9, 9, 9)))
    assert list(result[0, 0]) == [9, 9, 9]


def _solid_from_hsl(hue: float, sat: float, light: float, size=(4, 4)) -> Image.Image:
    hsl = np.tile(np.array([hue, sat, light], dtype=np.float32), (size[1], size[0], 1))
    return from_float_rgb(hsl_to_rgb(hsl))


def test_vibrance_is_a_no_op_at_zero():
    image = Image.new("RGB", (5, 5), (120, 140, 90))
    assert apply_vibrance(image, 0) is image


def test_vibrance_boosts_pastel_saturation_more_than_already_saturated():
    """RawTherapee's own description of Vibrance: it "protects skin tones and greys" and treats
    pastel vs. already-saturated colors differently. Same hue (well outside the protected
    skin-tone range) and lightness, two different starting saturations."""
    pastel = _solid_from_hsl(200, 0.15, 0.5)
    saturated = _solid_from_hsl(200, 0.9, 0.5)

    pastel_gain = rgb_to_hsl(to_float_rgb(apply_vibrance(pastel, 80)))[0, 0, 1] - 0.15
    saturated_gain = rgb_to_hsl(to_float_rgb(apply_vibrance(saturated, 80)))[0, 0, 1] - 0.9

    assert pastel_gain > saturated_gain
    assert pastel_gain > 0.05


def test_vibrance_protects_skin_tone_hues():
    skin = _solid_from_hsl(25, 0.4, 0.6)  # the protected skin-tone hue center
    non_skin = _solid_from_hsl(200, 0.4, 0.6)  # same starting saturation/lightness, unrelated hue

    skin_gain = rgb_to_hsl(to_float_rgb(apply_vibrance(skin, 80)))[0, 0, 1] - 0.4
    non_skin_gain = rgb_to_hsl(to_float_rgb(apply_vibrance(non_skin, 80)))[0, 0, 1] - 0.4

    assert skin_gain < non_skin_gain


def test_vibrance_negative_amount_desaturates():
    image = _solid_from_hsl(200, 0.5, 0.5)
    result_sat = rgb_to_hsl(to_float_rgb(apply_vibrance(image, -80)))[0, 0, 1]
    assert result_sat < 0.5


def test_estimate_white_balance_is_zero_for_an_already_neutral_pixel():
    image = Image.new("RGB", (10, 10), (128, 128, 128))
    temp, tint = estimate_white_balance(image, 0.5, 0.5)
    assert abs(temp) < 1e-6
    assert abs(tint) < 1e-6


def test_estimate_white_balance_cools_a_warm_cast_pixel():
    image = Image.new("RGB", (10, 10), (180, 140, 90))  # R > B: a warm cast
    temp, _ = estimate_white_balance(image, 0.5, 0.5)
    assert temp < 0  # negative temperature cools


def test_estimate_white_balance_pushes_toward_magenta_for_a_green_cast_pixel():
    image = Image.new("RGB", (10, 10), (120, 170, 120))  # G high relative to R/B: a green cast
    _, tint = estimate_white_balance(image, 0.5, 0.5)
    assert tint > 0  # positive tint counteracts green with magenta


def test_estimate_white_balance_samples_the_correct_fractional_pixel():
    image = Image.new("RGB", (10, 10), (128, 128, 128))
    image.putpixel((8, 8), (200, 100, 50))  # a warm patch near the bottom-right corner
    temp, _ = estimate_white_balance(image, 0.85, 0.85)
    assert temp < -5  # picked up the warm patch, not the neutral background


def test_estimate_white_balance_handles_pure_black_without_dividing_by_zero():
    image = Image.new("RGB", (5, 5), (0, 0, 0))
    assert estimate_white_balance(image, 0.5, 0.5) == (0.0, 0.0)


def test_estimate_white_balance_correction_actually_neutralizes_the_sampled_pixel():
    """End-to-end sanity check: applying the returned temperature/tint through the real
    PhotoProcessor.apply_temperature_tint() should land close to neutral gray -- confirms the
    closed-form solve is numerically close, not just directionally right. A mild cast (not one
    severe enough to clip temperature/tint at +-100) keeps the small linearization error this
    formula's own docstring describes from being swamped by clipping."""
    from app.image_processor import PhotoProcessor

    image = Image.new("RGB", (10, 10), (150, 140, 125))
    temp, tint = estimate_white_balance(image, 0.5, 0.5)
    corrected = PhotoProcessor.from_image(image).apply_temperature_tint(temp, tint).image
    r, g, b = corrected.getpixel((5, 5))
    assert max(r, g, b) - min(r, g, b) < 3
