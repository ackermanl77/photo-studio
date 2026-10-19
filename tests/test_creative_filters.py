"""Creative filters (F26): lens/motion blur, bloom, halftone, glitch -- the math, resolution
independence, the 16-bit path and the pipeline's Effects step."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import high_bit
from app.creative_filters import apply_creative_filters, bloom, glitch, halftone, lens_blur, motion_blur
from app.edit_schema import parse_options
from app.pipeline import run_pipeline


def _points(size=(300, 200)):
    image = Image.new("RGB", size, (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.ellipse((size[0] // 2 - 2, size[1] // 2 - 2, size[0] // 2 + 2, size[1] // 2 + 2), fill=(255, 255, 255))
    return np.asarray(image, np.float32) / 255.0


def test_lens_blur_spreads_a_point_into_a_round_disc_and_respects_focus():
    out = lens_blur(_points(), 100)
    h, w = out.shape[:2]
    row, col = out[h // 2, :, 0], out[:, w // 2, 0]
    spread_x, spread_y = (row > 0.12).sum(), (col > 0.12).sum()
    assert spread_x > 6 and abs(spread_x - spread_y) <= 2  # round, not streaked
    centre_kept = lens_blur(_points(), 100, "Center")
    np.testing.assert_allclose(centre_kept[h // 2, w // 2], _points()[h // 2, w // 2], atol=0.02)  # middle stays sharp


def test_motion_blur_streaks_along_its_angle():
    out = motion_blur(_points(), 100, 0)
    h, w = out.shape[:2]
    assert (out[h // 2, :, 0] > 0.1).sum() > 2 * (out[:, w // 2, 0] > 0.1).sum()
    vertical = motion_blur(_points(), 100, 90)
    assert (vertical[:, w // 2, 0] > 0.1).sum() > 2 * (vertical[h // 2, :, 0] > 0.1).sum()


def test_bloom_brightens_around_highlights_only():
    image = _points()
    out = bloom(image, 100, 70)
    h, w = image.shape[:2]
    assert out[h // 2, w // 2 + 8, 0] > image[h // 2, w // 2 + 8, 0] + 0.02  # glow next to the point
    np.testing.assert_allclose(out[5, 5], image[5, 5], atol=0.01)  # dark corner untouched
    dark = np.full((50, 50, 3), 0.3, np.float32)
    np.testing.assert_allclose(bloom(dark, 100, 70), dark, atol=1e-4)  # nothing above the threshold


def test_halftone_dots_follow_the_tone():
    ramp = np.repeat(np.linspace(0.0, 1.0, 400, dtype=np.float32)[None, :, None], 200, axis=0).repeat(3, axis=2)
    out = halftone(ramp, 60)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[:, :40].mean() < 0.2 and out[:, -40:].mean() > 0.8  # dark side mostly ink, light side paper
    color = halftone(np.dstack([ramp[..., 0], 1 - ramp[..., 1], ramp[..., 2] * 0 + 0.5]), 60, "Color")
    assert not np.allclose(color[..., 0], color[..., 1])  # one screen per channel


def test_glitch_is_deterministic_and_keeps_the_size():
    image = np.random.default_rng(0).random((120, 160, 3)).astype(np.float32)
    first, second = glitch(image, 70), glitch(image, 70)
    np.testing.assert_array_equal(first, second)
    assert first.shape == image.shape and not np.allclose(first, image)
    np.testing.assert_array_equal(glitch(image, 0), image)


@pytest.mark.parametrize("options", [
    {"effects_lens_blur": 60}, {"effects_motion_blur": 60, "effects_motion_angle": 30}, {"effects_bloom": 80},
    {"effects_halftone": 50}, {"effects_glitch": 50},
])
def test_resolution_independent(options):
    small = np.asarray(Image.fromarray((_points((300, 200)) * 255).astype(np.uint8)).resize((300, 200)), np.float32) / 255
    big = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((900, 600), Image.Resampling.NEAREST), np.float32) / 255
    out_small = apply_creative_filters(small, options)
    out_big = np.asarray(Image.fromarray((apply_creative_filters(big, options) * 255).astype(np.uint8)).resize((300, 200), Image.Resampling.BOX),
                         np.float32) / 255
    assert np.abs(out_small - out_big).mean() < 0.06


def test_pipeline_runs_them_after_the_crop_on_both_paths():
    image = Image.fromarray((_points() * 255).astype(np.uint8))
    options = parse_options({"effects_lens_blur": 80, "effects_lens_blur_focus": "Whole photo", "aspect": "1:1"})
    result = run_pipeline(image.copy(), options)
    assert result.size == (200, 200)
    assert np.asarray(result)[100, 100, 0] < 200  # the point is blurred
    float_image = high_bit.attach(image.copy(), high_bit.from_image(image))
    hdr = run_pipeline(float_image, parse_options({"effects_bloom": 80, "bit_depth": 16}))
    assert high_bit.attached(hdr) is not None
    assert parse_options({"effects_halftone_mode": "Color"})["effects_halftone_mode"] == "Color"
