"""Generated test images -- no copyrighted or personal photos in the repository (T6 in the
improvement report: tests/assets used to carry anime wallpapers and Facebook downloads).

Everything here is built from code at test time, so the suite needs no binary fixtures except
tests/assets/portrait_public_domain.jpg (a public-domain 1863 photograph, needed because face
detection can't be exercised with a drawn face).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw


def photo_like(size: tuple[int, int] = (320, 240), seed: int = 0) -> Image.Image:
    """A photo-ish RGB image: a sky-to-ground gradient, a few flat shapes, and mild noise, so every
    adjustment (tone, color, detail, masks) has something to act on."""
    width, height = size
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, height)[:, None]
    x = np.linspace(0.0, 1.0, width)[None, :]
    red = 60 + 150 * y + 20 * x
    green = 110 + 80 * (1 - y) + 30 * x
    blue = 200 - 120 * y
    array = np.stack([np.broadcast_to(c, (height, width)) for c in (red, green, blue)], axis=-1)
    array = array + rng.normal(0, 6, array.shape)
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse((width * 0.6, height * 0.1, width * 0.8, height * 0.35), fill=(250, 235, 180))
    draw.rectangle((width * 0.1, height * 0.55, width * 0.4, height * 0.95), fill=(70, 50, 40))
    draw.polygon([(width * 0.45, height * 0.95), (width * 0.65, height * 0.5), (width * 0.85, height * 0.95)], fill=(40, 90, 50))
    return image


def encode(image: Image.Image, fmt: str = "JPEG", **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def gray16_png(size: tuple[int, int] = (256, 16)) -> bytes:
    """A 0..65535 horizontal gradient as a 16-bit grayscale PNG (mode I;16) -- the input that used
    to decode almost entirely white (B7)."""
    gradient = np.tile(np.linspace(0, 65535, size[0]).astype(np.uint16), (size[1], 1))
    return encode(Image.fromarray(gradient), "PNG")


def dng_bytes(size: tuple[int, int] = (64, 48)) -> bytes:
    """A tiny but real CFA (RGGB Bayer) DNG that LibRaw demosaics: red rises left to right, green
    rises top to bottom, blue is flat. Needs tifffile (a requirements.txt dependency)."""
    import tifffile

    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    red = xx / width * 3000 + 500
    green = yy / height * 3000 + 500
    blue = np.full((height, width), 1500.0)
    cfa = np.zeros((height, width), np.uint16)
    cfa[0::2, 0::2] = red[0::2, 0::2]
    cfa[0::2, 1::2] = green[0::2, 1::2]
    cfa[1::2, 0::2] = green[1::2, 0::2]
    cfa[1::2, 1::2] = blue[1::2, 1::2]
    identity = (10000, 10000, 0, 10000, 0, 10000, 0, 10000, 10000, 10000, 0, 10000, 0, 10000, 0, 10000, 10000, 10000)
    extratags = [
        (50706, "B", 4, (1, 4, 0, 0), True),  # DNGVersion
        (50707, "B", 4, (1, 1, 0, 0), True),  # DNGBackwardVersion
        (50708, "s", 0, "Synthetic Test Camera", True),  # UniqueCameraModel
        (33421, "H", 2, (2, 2), True),  # CFARepeatPatternDim
        (33422, "B", 4, (0, 1, 1, 2), True),  # CFAPattern: RGGB
        (50710, "B", 3, (0, 1, 2), True),  # CFAPlaneColor
        (50711, "H", 1, 1, True),  # CFALayout
        (50714, "H", 1, 0, True),  # BlackLevel
        (50717, "H", 1, 4095, True),  # WhiteLevel
        (50721, "2i", 9, identity, True),  # ColorMatrix1
        (50728, "2I", 3, (1, 1, 1, 1, 1, 1), True),  # AsShotNeutral
        (50778, "H", 1, 21, True),  # CalibrationIlluminant1: D65
    ]
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, cfa, photometric=32803, compression=None, extratags=extratags, metadata=None, subfiletype=0)
    return buffer.getvalue()
