from __future__ import annotations

from collections.abc import Callable

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


def apply_preset(processor: PhotoProcessor, name: str) -> PhotoProcessor:
    try:
        preset = PRESETS[name]
    except KeyError as error:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}") from error
    return preset(processor)
