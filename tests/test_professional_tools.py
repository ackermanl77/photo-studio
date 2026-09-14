import io

import numpy as np
from PIL import Image

from app.professional_tools import EXPORT_PROFILES, apply_mask_overlay, export_profile, normalize_mask, validate_project_bytes, zoom_preview


def test_zoom_preview_crops_and_preserves_valid_image():
    image = Image.new("RGB", (400, 200), (100, 120, 140))
    result = zoom_preview(image, zoom=2.0, center=(0.75, 0.5))
    assert result.size[0] <= 1200
    assert result.size[1] <= 900


def test_mask_tools_support_invert_feather_and_overlay():
    image = Image.new("RGB", (20, 10), (100, 100, 100))
    mask = np.zeros((5, 10), dtype=np.uint8)
    mask[2, 5] = 255
    normalized = normalize_mask(mask, image.size, invert=True, feather=2)
    overlay = apply_mask_overlay(image, normalized)
    assert normalized.size == image.size
    assert overlay.size == image.size


def test_export_profiles_produce_expected_formats_and_sizes():
    image = Image.new("RGB", (3000, 2000), (100, 120, 140))
    for name, profile in EXPORT_PROFILES.items():
        content, extension, mime = export_profile(image, name)
        assert content
        assert extension.startswith(".")
        assert mime.startswith("image/")
        with Image.open(io.BytesIO(content)) as output:
            assert output.width <= profile["max_size"][0] if profile["max_size"] else True


def test_project_manifest_validation():
    assert validate_project_bytes(b'{"version": 1, "recipe": {}}')["version"] == 1
