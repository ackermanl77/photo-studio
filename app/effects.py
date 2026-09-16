from __future__ import annotations

import numpy as np
from PIL import Image


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
    std_dev = (amount / 100.0) * 20.0
    noise = np.random.default_rng().normal(0.0, std_dev, size=arr.shape[:2])[..., None]
    result = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")
