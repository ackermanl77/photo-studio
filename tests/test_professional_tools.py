import io

import numpy as np
from PIL import Image

from app.professional_tools import (
    EXPORT_PROFILES,
    GRID_PRESETS,
    WATERMARK_POSITIONS,
    apply_mask_overlay,
    apply_watermark,
    compute_histogram,
    compute_waveform,
    draw_guides_and_grid,
    export_profile,
    grid_line_fractions,
    normalize_mask,
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


def test_compute_waveform_returns_an_image_sized_to_source_width_and_requested_height():
    image = Image.new("RGB", (50, 20), (128, 128, 128))
    result = compute_waveform(image, mode="Luma", height=100)
    assert result.size == (50, 100)
    assert result.mode == "RGB"


def test_compute_waveform_unknown_mode_falls_back_to_luma_instead_of_raising():
    image = Image.new("RGB", (10, 10), (128, 128, 128))
    result = compute_waveform(image, mode="Not A Real Mode", height=64)
    assert result.size == (10, 64)


def test_compute_waveform_luma_places_a_solid_colors_full_density_at_its_expected_row():
    bins = 256
    image = Image.new("RGB", (10, 10), (128, 128, 128))  # every pixel has luma 128
    pixels = np.asarray(compute_waveform(image, mode="Luma", height=bins))

    expected_row = round((255 - 128) / 255 * (bins - 1))
    # Every column's entire pixel count lands in this one row -- it's this channel's own peak,
    # so it normalizes to full brightness (235, the fixed Luma trace color) across every column.
    assert np.all(pixels[expected_row] == 235)
    assert np.all(pixels[0] == 0)
    assert np.all(pixels[bins - 1] == 0)


def test_compute_waveform_rgb_mode_combines_equal_channels_additively_to_white():
    bins = 256
    image = Image.new("RGB", (6, 6), (128, 128, 128))  # R == G == B: all three traces coincide
    pixels = np.asarray(compute_waveform(image, mode="RGB", height=bins))

    expected_row = round((255 - 128) / 255 * (bins - 1))
    assert np.all(pixels[expected_row] == (255, 255, 255))
    assert np.all(pixels[0] == 0)


def test_watermark_positions_lists_the_five_named_presets():
    assert set(WATERMARK_POSITIONS) == {"Top Left", "Top Right", "Bottom Left", "Bottom Right", "Center"}


def test_apply_watermark_with_blank_text_is_a_no_op():
    image = Image.new("RGB", (100, 60), (10, 20, 30))
    result = apply_watermark(image, "   ")
    assert np.array_equal(np.asarray(result), np.asarray(image.convert("RGB")))


def test_apply_watermark_draws_near_the_requested_corner_and_leaves_the_opposite_corner_untouched():
    image = Image.new("RGB", (200, 120), (10, 10, 10))
    result = apply_watermark(image, "HELLO", position="Top Left", opacity=100, font_size=24)
    pixels = np.asarray(result)

    top_left_region = pixels[:30, :80]
    bottom_right_region = pixels[-10:, -10:]
    assert not np.array_equal(top_left_region, np.full_like(top_left_region, 10))
    assert np.array_equal(bottom_right_region, np.full_like(bottom_right_region, 10))


def test_apply_watermark_different_positions_produce_different_images():
    image = Image.new("RGB", (200, 120), (10, 10, 10))
    top_left = np.asarray(apply_watermark(image, "X", position="Top Left", opacity=100, font_size=24))
    bottom_right = np.asarray(apply_watermark(image, "X", position="Bottom Right", opacity=100, font_size=24))
    assert not np.array_equal(top_left, bottom_right)


def test_apply_watermark_low_opacity_blends_more_subtly_than_high_opacity():
    image = Image.new("RGB", (200, 120), (10, 10, 10))
    faint = np.asarray(apply_watermark(image, "X", opacity=10, font_size=40, color=(255, 255, 255))).astype(int)
    strong = np.asarray(apply_watermark(image, "X", opacity=100, font_size=40, color=(255, 255, 255))).astype(int)
    background = np.full_like(faint, 10)
    assert 0 < np.abs(faint - background).sum() < np.abs(strong - background).sum()


def test_export_profile_applies_watermark_when_text_given():
    image = Image.new("RGB", (400, 300), (10, 10, 10))
    plain_bytes, _, _ = export_profile(image, "Web")
    watermarked_bytes, _, _ = export_profile(image, "Web", watermark_text="Copyright", watermark_opacity=100)
    assert plain_bytes != watermarked_bytes


def test_export_profile_ignores_blank_watermark_text():
    image = Image.new("RGB", (100, 80), (50, 60, 70))
    plain_bytes, _, _ = export_profile(image, "Archive")
    blank_watermark_bytes, _, _ = export_profile(image, "Archive", watermark_text="   ")
    assert plain_bytes == blank_watermark_bytes
