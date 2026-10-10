from __future__ import annotations

import numpy as np
from PIL import Image


GRAIN_GRID_LONG_EDGE = 1800


def apply_film_grain(image: Image.Image, amount: float = 0.0) -> Image.Image:
    """Creative film grain -- Lightroom's/RawTherapee's "Effects" panel offers the same idea,
    always applied after crop/export framing (like the Post-Crop Vignette above) since grain is a
    finishing texture on the delivered frame, not a lens/sensor correction. `amount` 0-100 (0 is a
    no-op) scales monochromatic Gaussian noise -- the same value added to all three channels at
    each pixel, matching real film grain's luminance-only look, unlike per-channel color speckle
    (which would tint instead of texture). Deliberately unseeded: a fixed seed would make every
    export of the same photo carry an identical, visibly repeating grain pattern.
    """
    amount = max(0.0, min(100.0, float(amount)))
    if not amount:
        return image
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = film_grain_noise(arr.shape[1], arr.shape[0], amount)
    result = np.clip(arr + noise[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def film_grain_noise(width: int, height: int, amount: float) -> np.ndarray:
    """The grain itself: a (height, width) float32 field in 8-bit levels, added to all three
    channels. The 16-bit export path adds it (/255) to its float pixels."""
    std_dev = (max(0.0, min(100.0, float(amount))) / 100.0) * 20.0
    # Generated on a grid of at most GRAIN_GRID_LONG_EDGE px and resized to the frame, so the grain
    # has the same size relative to the photo at any resolution -- per-pixel noise was invisible on
    # a 24 MP export but three times coarser on the 2048 px interactive preview.
    grid_scale = min(1.0, GRAIN_GRID_LONG_EDGE / max(width, height))
    grid = (max(1, round(width * grid_scale)), max(1, round(height * grid_scale)))
    noise = np.random.default_rng().normal(0.0, std_dev, size=(grid[1], grid[0])).astype(np.float32)
    if grid != (width, height):
        noise = np.asarray(Image.fromarray(noise, mode="F").resize((width, height), Image.Resampling.BICUBIC))
    return noise
