from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import tifffile
from PIL import Image, ImageCms

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
    """Like PhotoProcessor.apply_curves(), but interpolates the curve in LINEAR light instead of
    raw 0-255 display values. `apply_curves()` (app/image_processor.py) treats brightness as if it
    were perceptually linear, which isn't physically accurate for a curve meant to model real
    light; this instead converts both the lookup domain and the control points' x-axis to linear
    light (`value ** gamma`), interpolates there, then converts the result back to a display value
    (`** (1 / gamma)`) -- the same round-trip a properly gamma-aware curve tool applies.
    """
    if gamma <= 0:
        raise ValueError("Gamma must be positive")
    ordered = sorted((int(x), int(y)) for x, y in points)
    x_values = np.array([point[0] for point in ordered], dtype=np.float32)
    y_values = np.array([point[1] for point in ordered], dtype=np.float32)
    if len(points) < 2 or x_values[0] < 0 or x_values[-1] > 255 or np.any(np.diff(x_values) <= 0):
        raise ValueError("Curve inputs must be unique and within 0-255")
    linear_input = np.linspace(0, 1, 256) ** gamma
    linear_x = (x_values / 255.0) ** gamma
    linear_y = (y_values / 255.0) ** gamma
    linear_output = np.interp(linear_input, linear_x, linear_y).clip(0, 1)
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


# --- ICC color management -----------------------------------------------------------------
# Real conversion through littlecms (bundled with Pillow as PIL.ImageCms), not just tagging and
# hoping. Deliberately does NOT simulate a specific external printer/press profile -- that's what
# "soft proofing" usually implies in Lightroom/Photoshop, but doing it properly needs a calibrated
# profile file (e.g. a press house's own .icc) this offline-first app has no database or bundled
# copy of. What's here instead: (1) real conversion of an image's *own* embedded profile into the
# sRGB working space every other step in this app's pipeline assumes, replacing the previous
# "preserve the original bytes, don't actually look at them" behavior, and (2) a soft-proof aimed
# at this app's own sRGB pipeline as the "destination" -- proofing against your own output space is
# itself a standard, meaningful proof target, and needs no external file since both conversions
# below target the same built-in sRGB profile.

def srgb_icc_bytes() -> bytes:
    """A standard sRGB ICC profile's raw bytes, synthesized by littlecms (no file on disk needed).
    For re-tagging an image that's just been normalize_to_srgb()'d -- re-attaching the *original*
    (pre-conversion) profile bytes to already-converted pixel values would make a downstream viewer
    interpret them a second time, effectively double-converting the color."""
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def normalize_to_srgb(image: Image.Image, icc_bytes: bytes) -> Image.Image:
    """Converts `image` from whatever profile `icc_bytes` describes into the sRGB working space,
    using RELATIVE_COLORIMETRIC intent (preserves in-gamut colors exactly, clips the rest to the
    nearest reproducible one -- the standard choice for a working-space conversion, as opposed to
    PERCEPTUAL, which is meant for a final proof/print step instead). A no-op-equivalent (identity
    transform, modulo floating-point rounding) when the embedded profile already describes sRGB;
    genuinely changes pixel values for a wider source gamut (Adobe RGB, ProPhoto RGB, a camera's
    own profile, ...) so colors that were correct under that wider profile still look correct once
    downstream code (every other step in this app's pipeline) treats the pixels as plain sRGB.
    """
    input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
    srgb_profile = ImageCms.createProfile("sRGB")
    return ImageCms.profileToProfile(
        image.convert("RGB"), input_profile, srgb_profile,
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB",
    )


def _highlight_mismatches(base: np.ndarray, comparison: np.ndarray, warning_color: tuple[int, int, int], threshold: int) -> np.ndarray:
    """Pure array helper behind gamut_warning_overlay, factored out so its thresholding logic can
    be unit-tested directly with synthetic arrays -- constructing a real, genuinely-different-gamut
    ICC profile for a test fixture isn't practical without either bundling a real profile file
    (which this feature deliberately avoids, see the module-level note above) or hand-building raw
    ICC binary data, so the actual gamut-clipping-detection behavior is covered by this plus live
    verification instead of an end-to-end automated test."""
    base = np.asarray(base, dtype=np.uint8)
    mismatch = np.abs(base.astype(np.int16) - np.asarray(comparison, dtype=np.int16)).sum(axis=2) > threshold
    result = base.copy()
    result[mismatch] = warning_color
    return result


def gamut_warning_overlay(image: Image.Image, icc_bytes: bytes, warning_color: tuple[int, int, int] = (255, 0, 255), threshold: int = 24) -> Image.Image:
    """A soft-proofing preview: highlights, in `warning_color`, pixels that get clipped when this
    app's pipeline converts `image` from `icc_bytes`'s source gamut down into the sRGB working
    space (see normalize_to_srgb and the module-level note above for why sRGB, specifically, is the
    proof target here).

    Detects clipping by comparing two conversions of the same source: RELATIVE_COLORIMETRIC (clips
    out-of-gamut colors to the nearest reproducible one) against PERCEPTUAL (compresses the whole
    source gamut smoothly instead, so it never clips anything) -- a pixel where the two disagree by
    more than `threshold` was pushed to the sRGB gamut boundary by the colorimetric conversion, i.e.
    it was out of gamut for it. A source profile that's already within (or equal to) sRGB produces
    no warnings at all, since both intents then agree everywhere.
    """
    input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
    srgb_profile = ImageCms.createProfile("sRGB")
    rgb_image = image.convert("RGB")
    colorimetric = ImageCms.profileToProfile(rgb_image, input_profile, srgb_profile, renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB")
    perceptual = ImageCms.profileToProfile(rgb_image, input_profile, srgb_profile, renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB")
    highlighted = _highlight_mismatches(np.asarray(colorimetric), np.asarray(perceptual), warning_color, threshold)
    return Image.fromarray(highlighted, mode="RGB")
