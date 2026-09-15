from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import tifffile
from PIL import Image

WORKING_SPACES = ("sRGB", "Adobe RGB", "Display P3")

HSL_BAND_ORDER = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
HSL_BAND_HUES = {
    "Red": 0.0, "Orange": 30.0, "Yellow": 60.0, "Green": 120.0,
    "Aqua": 180.0, "Blue": 240.0, "Purple": 275.0, "Magenta": 320.0,
}
_HSL_MIXER_FALLOFF_DEGREES = 45.0


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


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB -> HSL, matching colorsys.rgb_to_hls's math (just returned as h/s/l order).
    `rgb`: float array (..., 3) in [0, 1]. Returns (..., 3): hue in [0, 360), saturation and
    lightness in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    lightness = (maxc + minc) / 2.0
    delta = maxc - minc
    has_color = delta > 1e-8

    denom = np.where(lightness <= 0.5, maxc + minc, 2.0 - maxc - minc)
    saturation = np.where(has_color, delta / np.where(denom < 1e-8, 1.0, denom), 0.0)

    safe_delta = np.where(has_color, delta, 1.0)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    hue = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hue = np.where(has_color, (hue / 6.0) % 1.0, 0.0) * 360.0
    return np.stack([hue, saturation, lightness], axis=-1)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_hsl. `hsl`: (..., 3), hue in degrees, saturation/lightness in [0, 1].
    Returns float RGB (..., 3) in [0, 1] (not rounded/clipped to uint8 -- the caller does that)."""
    hue = (hsl[..., 0] / 360.0) % 1.0
    sat = hsl[..., 1]
    light = hsl[..., 2]

    m2 = np.where(light <= 0.5, light * (1.0 + sat), light + sat - light * sat)
    m1 = 2.0 * light - m2

    def _channel(hue_shift: float) -> np.ndarray:
        hue_c = (hue + hue_shift) % 1.0
        return np.select(
            [hue_c < 1 / 6, hue_c < 0.5, hue_c < 2 / 3],
            [m1 + (m2 - m1) * hue_c * 6.0, m2, m1 + (m2 - m1) * (2 / 3 - hue_c) * 6.0],
            default=m1,
        )

    return np.stack([_channel(1 / 3), _channel(0.0), _channel(-1 / 3)], axis=-1)


def apply_hsl_mixer(image: Image.Image, adjustments: dict[str, dict[str, float]]) -> Image.Image:
    """Per-color-band hue/saturation/luminance grading -- a professional editor's "Color Mixer" /
    HSL panel. `adjustments` maps a subset of `HSL_BAND_ORDER` band names to
    `{"hue": -100..100, "sat": -100..100, "lum": -100..100}`; a band that's absent, or present
    with all three at 0, is skipped entirely (cheap no-op, and the common case since most bands
    are left untouched).

    Each pixel's hue determines how much weight it gets from each active band: a smooth falloff
    (1.0 at the band's own hue center, 0 at `_HSL_MIXER_FALLOFF_DEGREES` away) so a color between
    two named bands (e.g. a yellow-green) is graded as a blend of both rather than snapping
    entirely to one. Weights are deliberately not normalized to sum to 1 across bands -- two
    adjacent bands pushed the same direction are meant to reinforce each other, the same way they
    would in a real color mixer panel; the final saturation/lightness are still clamped to a valid
    range regardless.
    """
    active = {name: values for name, values in adjustments.items() if any(values.get(k, 0) for k in ("hue", "sat", "lum"))}
    if not active:
        return image

    rgb = to_float_rgb(image)
    hsl = rgb_to_hsl(rgb)
    hue, sat, light = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    hue_shift = np.zeros_like(hue)
    sat_delta = np.zeros_like(hue)
    lum_delta = np.zeros_like(hue)
    for name, values in active.items():
        center = HSL_BAND_HUES[name]
        angular_distance = np.abs(((hue - center + 180.0) % 360.0) - 180.0)
        weight = np.clip(1.0 - angular_distance / _HSL_MIXER_FALLOFF_DEGREES, 0.0, 1.0)
        hue_shift += weight * (float(values.get("hue", 0.0)) / 100.0) * 30.0
        sat_delta += weight * (float(values.get("sat", 0.0)) / 100.0)
        lum_delta += weight * (float(values.get("lum", 0.0)) / 100.0)

    new_hsl = np.stack(
        [(hue + hue_shift) % 360.0, np.clip(sat * (1.0 + sat_delta), 0.0, 1.0), np.clip(light * (1.0 + lum_delta * 0.5), 0.0, 1.0)],
        axis=-1,
    )
    return from_float_rgb(np.clip(hsl_to_rgb(new_hsl), 0.0, 1.0))


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
