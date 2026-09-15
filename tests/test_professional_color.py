import io

import numpy as np
import pytest
from PIL import Image

from app.professional_color import (
    ColorSettings,
    HSL_BAND_HUES,
    WORKING_SPACES,
    apply_gamma_curve,
    apply_hsl_mixer,
    from_float_rgb,
    hsl_to_rgb,
    profile_export,
    rgb_to_hsl,
    save_16bit_tiff,
    to_float_rgb,
)


def test_float_rgb_round_trip_is_bounded():
    image = Image.new("RGB", (8, 6), (40, 100, 220))
    values = to_float_rgb(image)
    restored = from_float_rgb(values)
    assert values.min() >= 0 and values.max() <= 1
    assert np.array_equal(np.asarray(image), np.asarray(restored))


def test_gamma_curve_and_16bit_tiff_export():
    image = Image.new("RGB", (10, 10), (80, 100, 120))
    curved = apply_gamma_curve(image, [(0, 0), (128, 150), (255, 255)])
    output = io.BytesIO()
    save_16bit_tiff(curved, output)
    output.seek(0)
    exported = Image.open(output)
    assert exported.format == "TIFF"
    assert exported.size == image.size


def test_gamma_curve_leaves_an_identity_curve_unchanged_regardless_of_gamma():
    """An identity curve (0,0)-(255,255) round-trips through the linear-light conversion (x**gamma
    then **(1/gamma)) back to itself exactly, for any gamma -- a basic sanity property any
    gamma-aware curve implementation must hold."""
    image = Image.new("RGB", (6, 6), (40, 120, 200))
    for gamma in (1.0, 1.8, 2.2, 3.0):
        result = apply_gamma_curve(image, [(0, 0), (255, 255)], gamma=gamma)
        assert np.array_equal(np.asarray(result), np.asarray(image))


def test_gamma_curve_actually_uses_the_gamma_parameter():
    """Regression test: apply_gamma_curve() used to compute a `linear_input = ... ** gamma` array
    and never use it, silently interpolating the curve over raw 0-255 indices exactly like the
    plain, non-gamma-aware PhotoProcessor.apply_curves() -- so changing `gamma` had no effect on
    where the curve's midtones landed beyond the final, standalone **(1/gamma) step. Now that the
    interpolation itself happens in linear light, a non-identity curve must produce different
    output for different gamma values."""
    image = Image.new("RGB", (10, 10), (60, 130, 200))
    points = [(0, 0), (128, 180), (255, 255)]

    result_linear = apply_gamma_curve(image, points, gamma=1.0)
    result_gamma = apply_gamma_curve(image, points, gamma=2.2)

    assert not np.array_equal(np.asarray(result_linear), np.asarray(result_gamma))


def test_profile_export_supports_declared_profiles():
    image = Image.new("RGB", (5, 5), (10, 20, 30))
    for profile in WORKING_SPACES:
        content, extension, mime = profile_export(image, profile)
        assert content and extension == ".png" and mime == "image/png"
    content, extension, mime = profile_export(image, "sRGB", sixteen_bit=True)
    assert content and extension == ".tif" and mime == "image/tiff"


def test_invalid_color_settings_are_rejected():
    with pytest.raises(ValueError):
        ColorSettings("CMYK")
    with pytest.raises(ValueError):
        ColorSettings("sRGB", gamma=0)


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
