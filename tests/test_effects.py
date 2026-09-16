import numpy as np
from PIL import Image

from app.effects import apply_film_grain


def test_film_grain_is_a_no_op_at_zero():
    image = Image.new("RGB", (20, 20), (100, 110, 120))
    assert apply_film_grain(image, 0) is image


def test_film_grain_increases_pixel_variance():
    image = Image.new("RGB", (60, 60), (120, 120, 120))
    result = apply_film_grain(image, 80)

    before = np.asarray(image, dtype=np.float32)
    after = np.asarray(result, dtype=np.float32)
    assert after.std() > before.std() + 3.0


def test_film_grain_is_monochromatic_per_pixel():
    """Real film grain is luminance-only -- the same noise value added to all three channels at a
    given pixel -- unlike per-channel color speckle, which would visibly tint instead of texture."""
    image = Image.new("RGB", (40, 40), (128, 128, 128))
    result = np.asarray(apply_film_grain(image, 100), dtype=np.int16)

    per_pixel_channel_spread = np.abs(result[..., 0] - result[..., 1]) + np.abs(result[..., 1] - result[..., 2])
    assert per_pixel_channel_spread.max() == 0


def test_film_grain_does_not_shift_the_mean_much():
    image = Image.new("RGB", (200, 200), (128, 128, 128))
    result = np.asarray(apply_film_grain(image, 80), dtype=np.float32)
    assert abs(result.mean() - 128) < 3.0
