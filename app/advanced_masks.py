from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageFilter


@dataclass
class MaskLayer:
    name: str
    image: Image.Image
    enabled: bool = True
    invert: bool = False
    opacity: float = 1.0

    def normalized(self, size: tuple[int, int]) -> Image.Image:
        mask = self.image.convert("L").resize(size, Image.Resampling.BILINEAR)
        if self.invert:
            mask = ImageChops.invert(mask)
        if self.opacity < 1.0:
            mask = mask.point(lambda value: int(value * max(0.0, min(1.0, self.opacity))))
        return mask


class MaskStack:
    def __init__(self, size: tuple[int, int]):
        self.size = size
        self.layers: list[MaskLayer] = []

    def add(self, layer: MaskLayer) -> "MaskStack":
        self.layers.append(layer)
        return self

    def remove(self, name: str) -> "MaskStack":
        self.layers = [layer for layer in self.layers if layer.name != name]
        return self

    def set_enabled(self, name: str, enabled: bool) -> "MaskStack":
        for layer in self.layers:
            if layer.name == name:
                layer.enabled = enabled
                return self
        raise KeyError(name)

    def combined(self) -> Image.Image:
        result = np.zeros((self.size[1], self.size[0]), dtype=np.float32)
        for layer in self.layers:
            if layer.enabled:
                result = np.maximum(result, np.asarray(layer.normalized(self.size), dtype=np.float32))
        return Image.fromarray(result.astype(np.uint8), mode="L")


def gradient_mask(size: tuple[int, int], direction: str = "horizontal", reverse: bool = False) -> Image.Image:
    width, height = size
    if direction == "vertical":
        values = np.tile(np.linspace(0, 255, height, dtype=np.uint8)[:, None], (1, width))
    else:
        values = np.tile(np.linspace(0, 255, width, dtype=np.uint8)[None, :], (height, 1))
    if reverse:
        values = 255 - values
    return Image.fromarray(values, mode="L")


def radial_mask(size: tuple[int, int], center: tuple[float, float] = (0.5, 0.5), radius: float = 0.5, feather: float = 0.25) -> Image.Image:
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.sqrt(((xx / max(1, width - 1) - center[0]) ** 2) + ((yy / max(1, height - 1) - center[1]) ** 2))
    edge = max(0.001, radius * max(0.01, feather))
    values = np.clip((radius - distance) / edge, 0.0, 1.0) * 255
    return Image.fromarray(values.astype(np.uint8), mode="L")


def luminance_mask(image: Image.Image, low: int = 0, high: int = 255, softness: int = 10) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=2)
    soft = max(1, int(softness))
    lower = np.clip((pixels - (low - soft)) / soft, 0, 1)
    upper = np.clip(((high + soft) - pixels) / soft, 0, 1)
    return Image.fromarray((lower * upper * 255).astype(np.uint8), mode="L")


def color_mask(image: Image.Image, target: tuple[int, int, int], tolerance: int = 40, softness: int = 10) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    distance = np.sqrt(((pixels - np.asarray(target, dtype=np.float32)) ** 2).sum(axis=2))
    soft = max(1, int(softness))
    values = np.clip((float(tolerance) + soft - distance) / soft, 0, 1) * 255
    return Image.fromarray(values.astype(np.uint8), mode="L")


def soften_mask(mask: Image.Image, radius: int) -> Image.Image:
    return mask.convert("L").filter(ImageFilter.GaussianBlur(max(0, int(radius))))
