"""F13: the Whites/Blacks end-point sliders and the Black & White mixer."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.cli import build_parser, options_from_args, parse_black_white
from app.image_processor import PhotoProcessor, whites_blacks_curve
from app.pipeline import run_pipeline
from app.professional_color import black_white_float
from tests.synthetic_assets import photo_like

RAMP = np.linspace(0.0, 1.0, 256)


def test_whites_blacks_is_the_identity_at_zero():
    assert np.array_equal(whites_blacks_curve(RAMP, 0, 0), RAMP)


@pytest.mark.parametrize("whites, blacks", [(100, 0), (-100, 0), (0, 100), (0, -100), (100, -100), (-100, 100)])
def test_whites_blacks_never_inverts_tones(whites, blacks):
    assert np.all(np.diff(whites_blacks_curve(RAMP, whites, blacks)) >= 0)


def test_whites_moves_the_top_and_blacks_the_bottom_with_midtones_nearly_unchanged():
    whites = whites_blacks_curve(RAMP, 100, 0)
    blacks = whites_blacks_curve(RAMP, 0, -100)
    assert whites[-1] == 1.0 and whites[192] > RAMP[192] + 0.05
    assert blacks[0] == 0.0 and blacks[64] < RAMP[64] - 0.05
    assert abs(whites[128] - RAMP[128]) < 0.04 and abs(blacks[128] - RAMP[128]) < 0.04


def test_whites_and_blacks_change_the_pipeline_output():
    image = photo_like()
    base = np.asarray(run_pipeline(image, {}), dtype=float)
    assert np.asarray(run_pipeline(image, {"whites": 60}), dtype=float).mean() > base.mean()
    assert np.asarray(run_pipeline(image, {"blacks": -60}), dtype=float).mean() < base.mean()


def test_black_white_mix_brightens_or_darkens_only_its_own_colour():
    blue, red, gray = [0.1, 0.2, 0.9], [0.9, 0.1, 0.1], [0.5, 0.5, 0.5]
    rgb = np.array([[blue, red, gray]], dtype=np.float32)
    plain = black_white_float(rgb)[0, :, 0]
    darker_sky = black_white_float(rgb, {"Blue": -100})[0, :, 0]
    assert darker_sky[0] < plain[0] - 0.1
    assert darker_sky[1] == pytest.approx(plain[1]) and darker_sky[2] == pytest.approx(plain[2])
    assert black_white_float(rgb, {"Red": 100})[0, 1, 0] > plain[1] + 0.1


def test_black_white_output_is_gray_in_8_and_16_bit():
    image = photo_like()
    options = {"black_white": True, "bw_green": -50, "bw_orange": 40}
    eight = np.asarray(run_pipeline(image, options), dtype=np.int16)
    sixteen = np.asarray(run_pipeline(image, dict(options, bit_depth=16)), dtype=np.int16)
    assert np.array_equal(eight[..., 0], eight[..., 1]) and np.array_equal(eight[..., 1], eight[..., 2])
    assert np.abs(eight - sixteen).max() <= 1


def test_color_grading_can_tone_a_black_and_white_conversion():
    out = np.asarray(run_pipeline(photo_like(), {"black_white": True, "cg_highlights_hue": 40, "cg_highlights_intensity": 60}), dtype=np.int16)
    assert np.abs(out[..., 0] - out[..., 2]).max() > 5


def test_invert_adjustment_is_exact_and_off_by_default():
    image = Image.new("RGB", (2, 1))
    image.putdata([(0, 64, 255), (31, 127, 200)])
    assert np.array_equal(np.asarray(run_pipeline(image, {})), np.asarray(image))
    assert np.array_equal(
        np.asarray(run_pipeline(image, {"invert": True})),
        255 - np.asarray(image),
    )


def test_invert_adjustment_preserves_high_precision_values():
    processor = PhotoProcessor.from_image(Image.new("RGB", (1, 1)))
    processor.enable_high_precision(np.array([[[0.125, 0.5, 0.875]]], dtype=np.float32))
    processor.apply_invert()
    assert processor.hdr is not None
    assert processor.hdr == pytest.approx(np.array([[[0.875, 0.5, 0.125]]], dtype=np.float32))


def test_cli_flags():
    args = build_parser().parse_args(["--input", "in.jpg", "--output", "out.jpg", "--whites", "20", "--blacks", "-15", "--black-white", "Blue=-60,red=30"])
    options = options_from_args(args)
    assert options["whites"] == 20 and options["blacks"] == -15
    assert options["black_white"] is True and options["bw_blue"] == -60 and options["bw_red"] == 30
    assert parse_black_white("") == {"black_white": True}
    with pytest.raises(ValueError):
        parse_black_white("Teal=10")
