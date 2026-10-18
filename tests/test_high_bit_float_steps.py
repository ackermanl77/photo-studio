"""B4: ICC conversion, dehaze and skin smoothing in float in the 16-bit path (8-bit path unchanged)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.color_profiles import display_p3_icc_bytes
from app.dehaze import apply_dehaze, dehaze_float
from app.high_bit import attach, attached, from_image
from app.icc_float import parse_matrix_trc, to_srgb_float
from app.pipeline import run_pipeline
from app.portrait_tools import smooth_skin, smooth_skin_float
from app.professional_color import normalize_to_srgb, srgb_icc_bytes
from tests.synthetic_assets import photo_like


def _ramp(width=1024, height=8, low=0.2, high=0.45) -> np.ndarray:
    ramp = np.linspace(low, high, width, dtype=np.float32)
    return np.ascontiguousarray(np.broadcast_to(np.stack([ramp, ramp * 0.9, ramp * 0.8], -1), (height, width, 3)))


def _image_with(pixels: np.ndarray, icc: bytes | None = None) -> Image.Image:
    image = Image.fromarray(np.clip(pixels * 255 + 0.5, 0, 255).astype(np.uint8), mode="RGB")
    if icc:
        image.info["icc_profile"] = icc
    return attach(image, pixels)


def test_matrix_trc_profiles_are_parsed_and_lut_or_broken_ones_are_not():
    assert parse_matrix_trc(display_p3_icc_bytes()) is not None
    assert parse_matrix_trc(srgb_icc_bytes()) is not None
    assert parse_matrix_trc(b"not a profile") is None
    assert parse_matrix_trc(display_p3_icc_bytes()[:140]) is None


def test_float_icc_conversion_matches_littlecms():
    rng = np.random.default_rng(0)
    pixels8 = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
    cms = np.asarray(normalize_to_srgb(Image.fromarray(pixels8), display_p3_icc_bytes()), dtype=np.float32)
    ours = to_srgb_float(pixels8.astype(np.float32) / 255.0, display_p3_icc_bytes()) * 255.0
    assert np.abs(cms - ours).mean() < 0.4 and np.abs(cms - ours).max() < 1.5
    same = to_srgb_float(pixels8.astype(np.float32) / 255.0, srgb_icc_bytes()) * 255.0
    assert np.abs(same - pixels8).max() < 1.0  # sRGB -> sRGB is the identity


def test_a_16_bit_display_p3_source_keeps_its_precision_through_the_icc_step():
    source = _image_with(_ramp(), display_p3_icc_bytes())
    high = attached(run_pipeline(source, {"bit_depth": 16}))
    levels = len(np.unique(np.rint(high[4, :, 1] * 65535)))
    assert levels > 600  # through 8 bits it would be ~65 levels across this ramp
    eight = np.asarray(run_pipeline(_image_with(_ramp(), display_p3_icc_bytes()), {}), dtype=np.float32)
    assert np.abs(high * 255 - eight).mean() < 0.8  # the same conversion


def test_float_dehaze_matches_the_8_bit_one_and_keeps_the_gradient():
    photo = photo_like((96, 64))
    eight = np.asarray(apply_dehaze(photo, 60), dtype=np.float32)
    float_ = np.clip(dehaze_float(from_image(photo), 60), 0, 1) * 255
    assert np.abs(eight - float_).mean() < 0.6
    ramp = dehaze_float(_ramp(), 60)
    assert len(np.unique(np.rint(ramp[4, :, 1] * 65535))) > 600


def test_float_skin_smoothing_matches_the_8_bit_one():
    photo = Image.new("RGB", (64, 48), (205, 150, 120))  # skin tone
    noisy = np.clip(np.asarray(photo, dtype=np.float32) + np.random.default_rng(1).normal(0, 12, (48, 64, 3)), 0, 255)
    noisy_image = Image.fromarray(noisy.astype(np.uint8))
    eight = np.asarray(smooth_skin(noisy_image, 0.8, blur_radius=2.0), dtype=np.float32)
    float_ = smooth_skin_float(np.asarray(noisy_image, dtype=np.float32) / 255, noisy_image, 0.8, blur_radius=2.0) * 255
    assert np.abs(eight - float_).mean() < 1.5
    assert float_.std() < np.asarray(noisy_image, dtype=np.float32).std()  # it did smooth


def test_the_16_bit_pipeline_runs_dehaze_and_portrait_in_float():
    source = _image_with(_ramp())
    high = attached(run_pipeline(source, {"bit_depth": 16, "dehaze": 40.0, "portrait_smoothing": 0.5,
                                          "portrait_faces_only": False}))
    assert len(np.unique(np.rint(high[4, :, 1] * 65535))) > 600
