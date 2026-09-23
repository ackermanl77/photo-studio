import io

import numpy as np
from PIL import Image

from app.professional_tools import (
    EXPORT_PROFILES,
    GRID_PRESETS,
    apply_mask_overlay,
    compute_histogram,
    draw_guides_and_grid,
    export_profile,
    grid_line_fractions,
    normalize_mask,
    validate_project_bytes,
    zoom_preview,
)


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


def test_compute_histogram_bins_each_channel_and_counts_every_pixel():
    image = Image.new("RGB", (10, 4), (10, 200, 0))
    histogram = compute_histogram(image, bins=32)

    assert set(histogram) == {"Red", "Green", "Blue"}
    for channel_counts in histogram.values():
        assert channel_counts.shape == (32,)
        assert channel_counts.sum() == 40  # 10x4 pixels, every one falls in some bin
    # A solid (10, 200, 0) image: the Red bin near value 10 and the Green bin near value 200
    # should hold all 40 pixels, and the (near-zero) Blue channel should peak in the first bin.
    assert histogram["Red"].argmax() * (256 / 32) <= 10 + (256 / 32)
    assert histogram["Blue"][0] == 40


def test_grid_line_fractions_returns_the_expected_positions_for_each_named_style():
    assert grid_line_fractions("None") == ()
    assert grid_line_fractions("Rule of Thirds") == (1 / 3, 2 / 3)
    assert grid_line_fractions("Not A Real Style") == ()  # unknown style is a no-op, not an error
    assert set(GRID_PRESETS) == {"None", "Rule of Thirds", "Grid 4x4", "Golden Ratio"}


def test_draw_guides_and_grid_is_a_no_op_with_no_grid_and_no_guides():
    image = Image.new("RGB", (100, 60), (50, 50, 50))

    result = draw_guides_and_grid(image, grid_type="None", guides=[])

    assert np.array_equal(np.asarray(result), np.asarray(image.convert("RGB")))


def test_draw_guides_and_grid_draws_rule_of_thirds_lines_at_the_expected_position():
    image = Image.new("RGB", (300, 300), (50, 50, 50))

    result = draw_guides_and_grid(image, grid_type="Rule of Thirds", guides=[])

    pixels = np.asarray(result)
    assert not np.array_equal(pixels[100], np.full_like(pixels[100], 50))  # a horizontal line crosses y=100 (300/3)
    assert not np.array_equal(pixels[:, 100], np.full_like(pixels[:, 100], 50))  # and a vertical line crosses x=100
    assert tuple(pixels[10, 10]) == (50, 50, 50)  # far from either grid line: untouched


def test_draw_guides_and_grid_draws_a_custom_guide_at_its_own_position():
    image = Image.new("RGB", (200, 100), (20, 20, 20))
    guides = [{"orientation": "vertical", "position_pct": 25.0}]

    result = draw_guides_and_grid(image, grid_type="None", guides=guides)

    pixels = np.asarray(result)
    assert not np.array_equal(pixels[:, 50], np.full_like(pixels[:, 50], 20))  # x=50 is 25% of width=200
    assert np.array_equal(pixels[:, 150], np.full_like(pixels[:, 150], 20))  # elsewhere: untouched
