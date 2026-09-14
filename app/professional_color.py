from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import tifffile
from PIL import Image

WORKING_SPACES = ("sRGB", "Adobe RGB", "Display P3")


@dataclass(frozen=True)
class ColorSettings:
    working_space: str = "sRGB"
    gamma: float = 2.2
    preserve_profile: bool = True

    def __post_init__(self):
        if self.working_space not in WORKING_SPACES:
            raise ValueError(f"Unsupported working space: {self.working_space}")
        if self.gamma <= 0:
            raise ValueError("Gamma must be positive")


def to_float_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def from_float_rgb(values: np.ndarray) -> Image.Image:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Expected an RGB array")
    return Image.fromarray(np.clip(values * 255.0, 0, 255).round().astype(np.uint8), mode="RGB")


def apply_gamma_curve(image: Image.Image, points: list[tuple[int, int]], gamma: float = 2.2) -> Image.Image:
    if gamma <= 0:
        raise ValueError("Gamma must be positive")
    ordered = sorted((int(x), int(y)) for x, y in points)
    x_values = np.array([point[0] for point in ordered], dtype=np.float32)
    y_values = np.array([point[1] for point in ordered], dtype=np.float32)
    if len(points) < 2 or x_values[0] < 0 or x_values[-1] > 255 or np.any(np.diff(x_values) <= 0):
        raise ValueError("Curve inputs must be unique and within 0-255")
    linear_input = np.linspace(0, 1, 256) ** gamma
    linear_output = np.interp(np.arange(256), x_values, y_values).clip(0, 255) / 255.0
    lookup = np.clip(linear_output ** (1.0 / gamma), 0, 1)
    values = to_float_rgb(image)
    indexes = np.clip((values * 255).round().astype(np.uint8), 0, 255)
    return from_float_rgb(lookup[indexes])


def save_16bit_tiff(image: Image.Image, output: io.BytesIO | str) -> io.BytesIO | str:
    values = np.asarray(image.convert("RGB"), dtype=np.uint16) * 257
    tifffile.imwrite(output, values, photometric="rgb")
    return output


def profile_export(image: Image.Image, profile: str = "sRGB", sixteen_bit: bool = False) -> tuple[bytes, str, str]:
    settings = ColorSettings(profile)
    buffer = io.BytesIO()
    if sixteen_bit:
        save_16bit_tiff(image, buffer)
        return buffer.getvalue(), ".tif", "image/tiff"
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), ".png", "image/png"
