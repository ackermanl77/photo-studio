"""Creative filters (F26): lens (bokeh) blur, motion blur, bloom, halftone and glitch.

Classical image processing only (OpenCV + NumPy), on the finished, cropped frame -- part of the
pipeline's Effects step. Every size is a fraction of the frame, never pixels, so a proxy preview and
the full-resolution export look the same; large blurs are computed on a reduced copy and scaled
back (they're smooth anyway). The filters work on float RGB arrays (0..1), so the 16-bit path
keeps its precision; blurs and bloom run in linear light, which is what makes out-of-focus
highlights bloom into bright discs instead of grey smudges.

`apply_creative_filters(array, options)` runs them in a fixed order: lens blur, motion blur, bloom,
halftone, glitch. An option at 0 is skipped.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

LENS_BLUR_FOCUS = ("Whole photo", "Center", "Tilt-shift band")
HALFTONE_MODES = ("Black & white", "Color")
_WORK_RADIUS = 12.0  # blurs run on a copy reduced so their radius is about this many pixels
OPTION_KEYS = ("effects_lens_blur", "effects_lens_blur_focus", "effects_motion_blur", "effects_motion_angle",
               "effects_bloom", "effects_bloom_threshold", "effects_halftone", "effects_halftone_mode", "effects_glitch")


def srgb_to_linear(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1 / 2.4) - 0.055).astype(np.float32)


def _reduced_filter(array: np.ndarray, radius: float, make_kernel) -> np.ndarray:
    """Convolve `array` with `make_kernel(radius_px)`, on a copy reduced so the kernel stays small."""
    height, width = array.shape[:2]
    scale = min(1.0, _WORK_RADIUS / max(radius, 1e-6))
    small = array if scale >= 1.0 else cv2.resize(array, (max(1, round(width * scale)), max(1, round(height * scale))),
                                                  interpolation=cv2.INTER_AREA)
    kernel = make_kernel(max(0.5, radius * scale))
    blurred = cv2.filter2D(small, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    return blurred if scale >= 1.0 else cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)


def _disc(radius: float) -> np.ndarray:
    r = max(1, int(math.ceil(radius)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    kernel = np.clip(radius + 0.5 - np.sqrt(xx * xx + yy * yy), 0.0, 1.0)  # anti-aliased edge
    return kernel / kernel.sum()


def _line(length: float, angle: float) -> np.ndarray:
    half = max(1, int(math.ceil(length / 2)))
    size = 2 * half + 1
    kernel = np.zeros((size, size), np.float32)
    dx, dy = math.cos(math.radians(angle)), -math.sin(math.radians(angle))
    for step in np.linspace(-length / 2, length / 2, max(2, int(length * 2))):
        x, y = half + step * dx, half + step * dy
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        for xi, wx in ((x0, 1 - (x - x0)), (x0 + 1, x - x0)):
            for yi, wy in ((y0, 1 - (y - y0)), (y0 + 1, y - y0)):
                if 0 <= xi < size and 0 <= yi < size:
                    kernel[yi, xi] += wx * wy
    return kernel / kernel.sum()


def _focus_mask(height: int, width: int, focus: str) -> np.ndarray | None:
    """0 = sharp, 1 = fully blurred. None = blur everything."""
    if focus == "Center":
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        d = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
        return np.clip((d - 0.3) / 0.45, 0.0, 1.0)[..., None]
    if focus == "Tilt-shift band":
        y = np.abs(np.linspace(-1.0, 1.0, height, dtype=np.float32))
        return np.broadcast_to(np.clip((y - 0.15) / 0.4, 0.0, 1.0)[:, None, None], (height, width, 1))
    return None


def lens_blur(array: np.ndarray, amount: float, focus: str = "Whole photo") -> np.ndarray:
    """Out-of-focus blur with a round aperture: radius up to 3 % of the short edge, in linear light."""
    height, width = array.shape[:2]
    radius = max(0.0, min(100.0, amount)) / 100.0 * 0.03 * min(height, width)
    if radius < 0.5:
        return array
    blurred = linear_to_srgb(_reduced_filter(srgb_to_linear(array), radius, _disc))
    mask = _focus_mask(height, width, focus)
    return blurred if mask is None else array * (1 - mask) + blurred * mask


def motion_blur(array: np.ndarray, amount: float, angle: float = 0.0) -> np.ndarray:
    """A straight streak up to 5 % of the long edge, at `angle` degrees (0 = horizontal)."""
    height, width = array.shape[:2]
    length = max(0.0, min(100.0, amount)) / 100.0 * 0.05 * max(height, width)
    if length < 1.0:
        return array
    return linear_to_srgb(_reduced_filter(srgb_to_linear(array), length, lambda r: _line(r, angle)))


def bloom(array: np.ndarray, amount: float, threshold: float = 75.0) -> np.ndarray:
    """A soft glow around the brightest parts (above `threshold` % brightness), screened on top."""
    strength = max(0.0, min(100.0, amount)) / 100.0 * 3.0
    if strength <= 0:
        return array
    t = max(0.0, min(0.99, threshold / 100.0))
    linear = srgb_to_linear(array)
    luma = linear @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    bright = linear * np.clip((luma - t) / (1 - t), 0.0, 1.0)[..., None]
    height, width = array.shape[:2]
    sigma = 0.03 * min(height, width)
    scale = min(1.0, _WORK_RADIUS / max(sigma, 1e-6))
    small = cv2.resize(bright, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(0.5, sigma * scale)), (width, height), interpolation=cv2.INTER_LINEAR)
    screened = 1 - (1 - linear) * (1 - np.clip(glow * strength, 0.0, 1.0))
    return linear_to_srgb(screened)


def halftone(array: np.ndarray, amount: float, mode: str = "Black & white") -> np.ndarray:
    """Printed-dot look: cells up to 6 % of the short edge, each a dot sized by its darkness
    (one grey screen, or one screen per R/G/B channel for "Color")."""
    height, width = array.shape[:2]
    cell = max(0.0, min(100.0, amount)) / 100.0 * 0.06 * min(height, width)
    if cell < 2.0:
        return array
    grid_w, grid_h = max(1, int(width / cell)), max(1, int(height / cell))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = (xx / width * grid_w) % 1.0 - 0.5, (yy / height * grid_h) % 1.0 - 0.5
    distance = np.sqrt(cx * cx + cy * cy) * cell  # pixels from the cell centre
    if mode == "Color":
        channels = [array[..., c] for c in range(3)]
    else:
        channels = [array @ np.array([0.299, 0.587, 0.114], np.float32)]
    out = []
    for channel in channels:
        tone = cv2.resize(cv2.resize(channel, (grid_w, grid_h), interpolation=cv2.INTER_AREA), (width, height),
                          interpolation=cv2.INTER_NEAREST)
        radius = np.sqrt(np.clip(1.0 - tone, 0.0, 1.0)) * cell * 0.7071  # full darkness covers the cell
        out.append(np.clip(distance - radius + 0.5, 0.0, 1.0))  # 1 = paper, 0 = ink, 1 px soft edge
    result = np.dstack(out) if mode == "Color" else np.repeat(out[0][..., None], 3, axis=2)
    return result.astype(np.float32)


def glitch(array: np.ndarray, amount: float) -> np.ndarray:
    """Digital glitch: red/blue channels split sideways and a few horizontal slices shifted.
    Seeded, so the preview and the export show the same glitch."""
    amount = max(0.0, min(100.0, amount))
    if amount <= 0:
        return array
    height, width = array.shape[:2]
    result = array.copy()
    shift = max(1, round(amount / 100.0 * 0.012 * width))
    result[..., 0] = np.roll(array[..., 0], shift, axis=1)
    result[..., 2] = np.roll(array[..., 2], -shift, axis=1)
    rng = np.random.default_rng(20261019)
    for _ in range(int(2 + amount / 12)):
        top = rng.uniform(0.0, 0.95)
        band = rng.uniform(0.005, 0.04) * amount / 100.0 + 0.004
        offset = rng.uniform(-1.0, 1.0) * amount / 100.0 * 0.08
        y0, y1 = int(top * height), min(height, int((top + band) * height) + 1)
        result[y0:y1] = np.roll(result[y0:y1], int(offset * width), axis=1)
    return result


def has_creative_filters(options: dict) -> bool:
    return any(options.get(key) for key in ("effects_lens_blur", "effects_motion_blur", "effects_bloom",
                                            "effects_halftone", "effects_glitch"))


def apply_creative_filters(array: np.ndarray, options: dict) -> np.ndarray:
    """`array`: float32 HxWx3 in 0..1. Returns the same shape."""
    array = np.ascontiguousarray(array, dtype=np.float32)
    if options.get("effects_lens_blur"):
        array = lens_blur(array, options["effects_lens_blur"], options.get("effects_lens_blur_focus") or "Whole photo")
    if options.get("effects_motion_blur"):
        array = motion_blur(array, options["effects_motion_blur"], options.get("effects_motion_angle", 0.0))
    if options.get("effects_bloom"):
        array = bloom(array, options["effects_bloom"], options.get("effects_bloom_threshold", 75.0))
    if options.get("effects_halftone"):
        array = halftone(array, options["effects_halftone"], options.get("effects_halftone_mode") or "Black & white")
    if options.get("effects_glitch"):
        array = glitch(array, options["effects_glitch"])
    return np.clip(array, 0.0, 1.0)
