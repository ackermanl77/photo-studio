import io

import numpy as np
import pytest
from PIL import Image

from app.professional_color import ColorSettings, WORKING_SPACES, apply_gamma_curve, from_float_rgb, profile_export, save_16bit_tiff, to_float_rgb


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
