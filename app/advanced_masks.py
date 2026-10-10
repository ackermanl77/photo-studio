from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter


COMBINE_MODES = ("add", "subtract", "intersect")


@dataclass
class MaskLayer:
    name: str
    image: Image.Image
    enabled: bool = True
    invert: bool = False
    opacity: float = 1.0
    # How this layer joins the ones above it in MaskStack.combined(): "add" (union), "subtract"
    # (removes its area), or "intersect" (keeps only the overlap). The first enabled layer always adds.
    combine: str = "add"

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
        first = True
        for layer in self.layers:
            if not layer.enabled:
                continue
            mask = np.asarray(layer.normalized(self.size), dtype=np.float32)
            if first or layer.combine not in COMBINE_MODES or layer.combine == "add":
                result = np.maximum(result, mask)
            elif layer.combine == "subtract":
                result = result * (1.0 - mask / 255.0)
            else:  # intersect
                result = np.minimum(result, mask)
            first = False
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="L")


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


WAND_SAMPLE_SIZES = (1, 3, 5, 11)


def magic_wand_mask(image: Image.Image, click_x: float, click_y: float, tolerance: int = 40, contiguous: bool = True,
                    sample_size: int = 1) -> Image.Image:
    """Click-to-select: samples the color at (`click_x`, `click_y`) -- fractions 0..1 of the
    image, resolution-independent since the on-canvas picker shows a downscaled preview -- and
    selects pixels within `tolerance` of it. Non-contiguous mode is exactly color_mask() with the
    target color sampled from the image instead of chosen via a color picker swatch; contiguous
    mode (the traditional "magic wand" behavior, and the default) restricts the result to the
    single connected region reachable from the click point by flood fill, so clicking one red
    apple doesn't also select an unconnected red object elsewhere in the frame.

    `sample_size` (1, 3, 5 or 11 px, Photoshop's "Sample Size"): above 1 the target is the average
    colour of that square around the click, steadier than one possibly noisy/anti-aliased pixel,
    and the contiguous fill compares every pixel with that average (a fixed range). 1 keeps the
    original single-pixel, neighbour-to-neighbour fill exactly.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    px = min(width - 1, max(0, int(click_x * width)))
    py = min(height - 1, max(0, int(click_y * height)))
    half = max(0, int(sample_size) // 2)
    if half:
        window = np.asarray(rgb)[max(0, py - half):py + half + 1, max(0, px - half):px + half + 1]
        target = tuple(int(round(v)) for v in window.reshape(-1, 3).mean(axis=0))
    else:
        target = rgb.getpixel((px, py))
    if not contiguous:
        return color_mask(rgb, target, tolerance)

    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    diff = (tolerance, tolerance, tolerance)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    if half:
        # The seed carries the sampled average and FIXED_RANGE compares every pixel with it.
        bgr[py, px] = target[::-1]
        flags |= cv2.FLOODFILL_FIXED_RANGE
    # FLOODFILL_MASK_ONLY: only `flood_mask` is used as output (the fill value written into it --
    # 255, not the default 1 -- is packed into flags' upper byte).
    cv2.floodFill(bgr, flood_mask, (px, py), (255, 255, 255), loDiff=diff, upDiff=diff, flags=flags)
    region = flood_mask[1:-1, 1:-1]
    return Image.fromarray(region, mode="L")
