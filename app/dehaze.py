from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Classical Dark Channel Prior dehaze (He, Sun & Tang, 2009): in almost any haze-free outdoor
# patch, at least one color channel has some very dark pixels (shadows, colorful/saturated
# surfaces, dark objects); haze raises that minimum toward the atmosphere's own color, so how far
# the per-pixel channel-minimum sits above zero in a local patch is a usable estimate of how much
# haze covers that patch. Deliberately not a learned/ML model -- the same CPU-first, no-GPU,
# no-heavy-new-dependency constraint this app holds to everywhere else (see docs/PROJECT_STATUS.md).
#
# The transmission map is smoothed with a plain Gaussian blur rather than a real edge-aware guided
# filter -- OpenCV's guided filter lives in opencv-contrib, which this app doesn't depend on (only
# opencv-python-headless, see requirements.txt). A real guided filter would preserve edges (e.g. a
# building's silhouette against a hazy sky) more precisely; this softer approximation can show a
# faint halo around strong edges at high strength instead of a crisp line.

_PATCH_SIZE = 15
_OMEGA = 0.95
_MIN_TRANSMISSION = 0.1
_DARK_CHANNEL_TOP_FRACTION = 0.001
_MIN_ATMOSPHERIC_LIGHT = 0.2


def _dark_channel(image: np.ndarray, patch_size: int = _PATCH_SIZE) -> np.ndarray:
    """Per-pixel minimum across R/G/B, then eroded (a local minimum filter) over `patch_size` --
    cv2.erode on a single-channel float image is exactly a min filter over its structuring element,
    much faster than a manual sliding-window minimum in Python/numpy."""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


def _estimate_atmospheric_light(image: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
    """The classic DCP heuristic: take the brightest 0.1% of pixels in the dark channel (the
    patches haze affects most), then use the brightest of *those* pixels in the original image,
    per channel, as the estimated color of the haze/atmosphere itself."""
    height, width = dark_channel.shape
    num_pixels = max(1, int(height * width * _DARK_CHANNEL_TOP_FRACTION))
    flat_dark = dark_channel.reshape(-1)
    flat_image = image.reshape(-1, 3)
    brightest_indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    return flat_image[brightest_indices].max(axis=0).astype(np.float32)


def apply_dehaze(image: Image.Image, strength: float = 0.0, pixel_scale: float = 1.0) -> Image.Image:
    """Reduces atmospheric haze via the Dark Channel Prior. `strength` is 0-100 (0 is a no-op);
    it scales `omega`, the fraction of the estimated haze this actually removes for a given pixel
    -- kept well under the theoretical maximum even at strength=100 so a hazy sky doesn't turn
    into visible halos and crushed blacks, a common DCP failure mode this simplified
    implementation doesn't try to fully solve (no sky segmentation, unlike some commercial tools;
    see the module-level note above about the missing guided filter too)."""
    strength = max(0.0, min(100.0, float(strength)))
    if strength <= 0:
        return image
    normalized = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    # `pixel_scale` < 1 on a reduced-size preview (see app.pipeline): the patch shrinks with it so
    # it covers the same fraction of the scene as at full resolution. Odd, at least 3 px.
    patch_size = _PATCH_SIZE if pixel_scale == 1.0 else max(3, int(round(_PATCH_SIZE * pixel_scale)) | 1)
    dark = _dark_channel(normalized, patch_size)
    atmospheric_light = _estimate_atmospheric_light(normalized, dark)
    atmospheric_light = np.clip(atmospheric_light, _MIN_ATMOSPHERIC_LIGHT, 1.0)

    omega = _OMEGA * (strength / 100.0)
    transmission = 1.0 - omega * _dark_channel(normalized / atmospheric_light[None, None, :], patch_size)
    blur_sigma = max(1.0, normalized.shape[1] / 50.0)
    transmission = cv2.GaussianBlur(transmission, (0, 0), sigmaX=blur_sigma)
    transmission = np.clip(transmission, _MIN_TRANSMISSION, 1.0)

    recovered = (normalized - atmospheric_light[None, None, :]) / transmission[..., None] + atmospheric_light[None, None, :]
    result = np.clip(recovered * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")
