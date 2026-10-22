from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from app.image_processor import PhotoProcessor

Preset = Callable[[PhotoProcessor], PhotoProcessor]


def _portrait_natural(processor: PhotoProcessor) -> PhotoProcessor:
    return (
        processor.apply_exposure(1.06)
        .apply_contrast(1.08)
        .apply_saturation(1.04)
        .apply_denoise(3)
        .apply_sharpen(1, 100)
    )


def _travel_clean(processor: PhotoProcessor) -> PhotoProcessor:
    return (
        processor.apply_exposure(1.08)
        .apply_contrast(1.18)
        .apply_saturation(1.14)
        .apply_sharpen(1, 120)
    )


def _product_neutral(processor: PhotoProcessor) -> PhotoProcessor:
    return processor.apply_contrast(1.12).apply_saturation(0.96).apply_sharpen(1, 110)


def _cinematic_soft(processor: PhotoProcessor) -> PhotoProcessor:
    return (
        processor.apply_temperature_tint(6, 2)
        .apply_curves([(0, 8), (64, 58), (128, 132), (192, 196), (255, 248)])
        .apply_contrast(1.05)
        .apply_saturation(0.92)
    )


PRESETS: dict[str, Preset] = {
    "portrait-natural": _portrait_natural,
    "travel-clean": _travel_clean,
    "product-neutral": _product_neutral,
    "cinematic-soft": _cinematic_soft,
}


def apply_preset(processor: PhotoProcessor, name: str, amount: float = 100.0) -> PhotoProcessor:
    """The built-in look `name` at `amount` percent (F30): 100 is the preset as designed, lower
    blends its result with the image it was applied to (0 = unchanged)."""
    try:
        preset = PRESETS[name]
    except KeyError as error:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}") from error
    strength = min(1.0, max(0.0, float(amount) / 100.0))
    if strength <= 0.0:
        return processor
    if strength >= 1.0:
        return preset(processor)
    if processor.hdr is not None:
        before = processor.hdr.copy()
        preset(processor)
        return processor.set_high_precision(before + (processor.hdr - before) * strength)
    before_image = processor.image
    preset(processor)
    after_image = processor.image
    if after_image.size != before_image.size or after_image.mode != before_image.mode:
        return processor
    before = np.asarray(before_image, dtype=np.float32)
    after = np.asarray(after_image, dtype=np.float32)
    blended = Image.fromarray(np.clip(np.rint(before + (after - before) * strength), 0, 255).astype(np.uint8), mode=after_image.mode)
    blended.info = dict(after_image.info)
    processor.image = blended
    return processor
