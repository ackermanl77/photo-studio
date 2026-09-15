import numpy as np
from PIL import Image

from app.dehaze import apply_dehaze


def test_dehaze_is_a_no_op_at_zero_strength():
    image = Image.new("RGB", (20, 20), (150, 150, 150))
    assert apply_dehaze(image, 0) is image


def test_dehaze_recovers_contrast_lost_to_simulated_haze():
    """Simulates haze with the same physical model Dark Channel Prior inverts: hazy = clear *
    transmission + atmosphere * (1 - transmission). A high-contrast two-tone image loses most of
    its contrast once haze is blended in; dehazing a meaningful amount should recover a
    substantial fraction of it back, not just nudge it slightly."""
    size = 60
    clear = np.zeros((size, size, 3), dtype=np.float32)
    clear[:, : size // 2] = [20, 20, 20]
    clear[:, size // 2 :] = [230, 230, 230]

    atmosphere = np.array([200.0, 200.0, 210.0], dtype=np.float32)
    transmission = 0.35  # heavy, uniform haze
    hazy = clear * transmission + atmosphere * (1 - transmission)
    hazy_image = Image.fromarray(np.clip(hazy, 0, 255).astype(np.uint8), mode="RGB")

    dehazed = np.asarray(apply_dehaze(hazy_image, 90), dtype=np.float32)
    hazy_arr = np.asarray(hazy_image, dtype=np.float32)

    hazy_contrast = abs(hazy_arr[:, : size // 2].mean() - hazy_arr[:, size // 2 :].mean())
    dehazed_contrast = abs(dehazed[:, : size // 2].mean() - dehazed[:, size // 2 :].mean())
    assert dehazed_contrast > hazy_contrast


def test_dehaze_strength_is_clamped_to_0_100():
    image = Image.new("RGB", (16, 16), (170, 175, 180))
    over = np.asarray(apply_dehaze(image, 500), dtype=np.float32)
    at_max = np.asarray(apply_dehaze(image, 100), dtype=np.float32)
    np.testing.assert_allclose(over, at_max)


def test_dehaze_preserves_image_size_and_mode():
    image = Image.new("RGB", (33, 21), (100, 120, 140))
    result = apply_dehaze(image, 50)
    assert result.size == image.size
    assert result.mode == "RGB"
