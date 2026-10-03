from __future__ import annotations

import colorsys
import io

import cv2
import numpy as np
from PIL import Image, ImageCms

WORKING_SPACES = ("sRGB", "Adobe RGB", "Display P3")

HSL_BAND_ORDER = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
HSL_BAND_HUES = {
    "Red": 0.0, "Orange": 30.0, "Yellow": 60.0, "Green": 120.0,
    "Aqua": 180.0, "Blue": 240.0, "Purple": 275.0, "Magenta": 320.0,
}
_HSL_MIXER_FALLOFF_DEGREES = 45.0


def to_float_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def from_float_rgb(values: np.ndarray) -> Image.Image:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Expected an RGB array")
    return Image.fromarray(np.clip(values * 255.0, 0, 255).round().astype(np.uint8), mode="RGB")


def _as_cv_image(values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """cv2.cvtColor needs a 2-D, 3-channel, contiguous float32 image -- reshape any (..., 3) array
    into one (N x 1 pixels) and hand back the original shape to restore afterward."""
    array = np.asarray(values, dtype=np.float32)
    return np.ascontiguousarray(array.reshape(-1, 1, 3)), array.shape


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB -> HSL, matching colorsys.rgb_to_hls's math (just returned as h/s/l order).
    `rgb`: float array (..., 3) in [0, 1]. Returns float32 (..., 3): hue in [0, 360), saturation
    and lightness in [0, 1].

    Uses OpenCV's float32 RGB2HLS conversion -- about 75x faster than the hand-written NumPy
    version it replaced (kept below as _rgb_to_hsl_reference for the parity test), which made
    Vibrance and the HSL mixer cost ~5 s each on a 24 MP photo. Same formula; OpenCV returns the
    channels in H, L, S order, so they're reordered here."""
    image, shape = _as_cv_image(rgb)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).reshape(shape)
    hue = np.where(hls[..., 0] >= 360.0, hls[..., 0] - 360.0, hls[..., 0])
    return np.stack([hue, hls[..., 2], hls[..., 1]], axis=-1)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_hsl. `hsl`: (..., 3), hue in degrees, saturation/lightness in [0, 1].
    Returns float32 RGB (..., 3) in [0, 1] (not rounded/clipped to uint8 -- the caller does that).
    OpenCV's HLS2RGB, the counterpart of rgb_to_hsl's conversion (see there)."""
    hsl = np.asarray(hsl, dtype=np.float32)
    hls = np.stack([np.mod(hsl[..., 0], 360.0), hsl[..., 2], hsl[..., 1]], axis=-1)
    image, shape = _as_cv_image(hls)
    return cv2.cvtColor(image, cv2.COLOR_HLS2RGB).reshape(shape)


def _rgb_to_hsl_reference(rgb: np.ndarray) -> np.ndarray:
    """The original pure-NumPy RGB -> HSL (colorsys.rgb_to_hls's math). No longer used by the
    pipeline -- rgb_to_hsl() above is the OpenCV-backed replacement -- kept only so a test can pin
    the two to the same output."""
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


def _hsl_to_rgb_reference(hsl: np.ndarray) -> np.ndarray:
    """The original pure-NumPy inverse of _rgb_to_hsl_reference -- see there."""
    hue =(hsl[..., 0] / 360.0) % 1.0
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


_SKIN_TONE_HUE_CENTER = 25.0
_SKIN_TONE_HUE_RANGE = 25.0
_SKIN_TONE_PROTECTION = 0.6  # skin-tone hues get 60% less vibrance boost than the formula would otherwise give


def apply_vibrance(image: Image.Image, amount: float = 0.0) -> Image.Image:
    """Vibrance: a "smart" saturation boost, distinct from the app's existing flat Saturation
    slider (PhotoProcessor.apply_saturation, PIL's ImageEnhance.Color, which scales every pixel's
    saturation by the same factor regardless of how saturated it already is). RawTherapee's
    Vibrance tool is the model here -- it "boosts colors but also protects skin tones and greys,
    and can separately boost pastels and saturated colors" -- so each pixel's boost is weighted by
    `1 - saturation`: an already-vivid pixel gets little extra push (avoiding clipped, oversaturated
    colors), while a washed-out/pastel pixel gets nearly the full amount. A second weight further
    dampens the effect specifically in the skin-tone hue range (a smooth falloff centered on ~25
    degrees, the same "distance from a center hue" shape apply_hsl_mixer already uses for its
    per-band falloff) so portraits don't turn artificially ruddy the way a global saturation boost
    can. `amount` -100..100 (0 is a no-op; negative desaturates pastels first, the mirror image of
    the positive case).
    """
    amount = max(-100.0, min(100.0, float(amount)))
    if not amount:
        return image
    rgb = to_float_rgb(image)
    hsl = rgb_to_hsl(rgb)
    hue, sat, light = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    pastel_weight = 1.0 - sat
    skin_distance = np.abs(((hue - _SKIN_TONE_HUE_CENTER + 180.0) % 360.0) - 180.0)
    skin_protection = 1.0 - _SKIN_TONE_PROTECTION * np.clip(1.0 - skin_distance / _SKIN_TONE_HUE_RANGE, 0.0, 1.0)
    new_sat = np.clip(sat * (1.0 + pastel_weight * skin_protection * (amount / 100.0)), 0.0, 1.0)

    new_hsl = np.stack([hue, new_sat, light], axis=-1)
    return from_float_rgb(np.clip(hsl_to_rgb(new_hsl), 0.0, 1.0))


def estimate_white_balance(image: Image.Image, x_fraction: float, y_fraction: float) -> tuple[float, float]:
    """The math behind a "click a neutral spot to set white balance" eyedropper, the same
    convenience darktable's color-calibration module offers ("click on the eyedropper... the
    module will automatically set the white balance, and often it works really well" -- an
    approximation by the original tool's own admission, not an exact solve, which is exactly the
    spirit implemented here). Samples the pixel at the given fractional (0..1, 0..1) position and
    returns the (temperature, tint) pair (each -100..100, PhotoProcessor.apply_temperature_tint's
    own scale) that would make that pixel neutral gray under that function's formula.

    Closed-form solved from apply_temperature_tint's own multipliers (R *= 1+0.25t, G *= 1-0.2u,
    B *= 1-0.25t, with the +-0.1u cross-terms on R/B dropped as second-order-small): setting all
    three post-multiplier channels equal and solving the resulting 2x2 linear system in (t, u)
    gives temperature = 600*g*(b-r)/(rb+rg+bg), tint = 500*(rg+bg-2rb)/(rb+rg+bg). Exact for a
    pixel already close to neutral (the intended use -- click something that's *supposed* to be
    gray or white); like the real tool, a good starting point to fine-tune from rather than a
    guaranteed perfect neutralization for a wildly off-color sample.
    """
    width, height = image.size
    x = min(width - 1, max(0, int(x_fraction * width)))
    y = min(height - 1, max(0, int(y_fraction * height)))
    r, g, b = (float(channel) for channel in image.convert("RGB").getpixel((x, y)))
    denom = r * b + r * g + b * g
    if denom <= 0:
        return 0.0, 0.0
    temperature = 600.0 * g * (b - r) / denom
    tint = 500.0 * (r * g + b * g - 2.0 * r * b) / denom
    return max(-100.0, min(100.0, temperature)), max(-100.0, min(100.0, tint))


def _tint_color(hue_degrees: float) -> np.ndarray:
    """A pure, mid-lightness color at the given hue (0-360) -- the "paint" apply_color_grading()
    tints each tonal range toward. Fixed at full saturation/50% lightness (colorsys.hls_to_rgb),
    matching how every split-toning tool's hue slider works: the *amount* of tint comes from the
    per-range intensity slider, not from varying this base color's own saturation or lightness."""
    r, g, b = colorsys.hls_to_rgb((hue_degrees % 360.0) / 360.0, 0.5, 1.0)
    return np.array([r, g, b], dtype=np.float32) * 255.0


def apply_color_grading(
    image: Image.Image,
    shadows_hue: float = 0.0, shadows_intensity: float = 0.0,
    midtones_hue: float = 0.0, midtones_intensity: float = 0.0,
    highlights_hue: float = 0.0, highlights_intensity: float = 0.0,
    balance: float = 0.0,
) -> Image.Image:
    """Split toning / color grading: tints shadows, midtones, and highlights toward independent
    hues, like every professional editor's Color Grading panel (three color wheels, pre-2021
    Lightroom's separate Split Toning panel, darktable's color balance). `*_intensity` is 0-100
    (0 = that range untouched); `balance` (-100..100) shifts the shadow/highlight split point,
    same direction convention as Lightroom's Balance slider (positive favors highlights).

    Each pixel gets a luminance-based weight per range (shadows near 0, highlights near 255,
    midtones peaking at the balance-adjusted midpoint, all three summing to 1) -- the same
    "smooth handoff between tonal ranges" shape apply_highlights_shadows() already uses, just
    normalized into three simultaneous weights instead of two independent ones. The tint color at
    each pixel is then a weighted blend toward whichever range(s) apply, scaled by that range's
    intensity; a pixel purely in one range only sees that range's tint, one near a boundary blends
    both smoothly rather than snapping.
    """
    if not (shadows_intensity or midtones_intensity or highlights_intensity):
        return image
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = arr.mean(axis=2) / 255.0

    pivot = min(0.9, max(0.1, 0.5 + (float(balance) / 100.0) * 0.35))
    shadow_w = np.clip((pivot - luminance) / pivot, 0.0, 1.0)
    highlight_w = np.clip((luminance - pivot) / (1.0 - pivot), 0.0, 1.0)
    midtone_w = np.clip(1.0 - shadow_w - highlight_w, 0.0, 1.0)

    shift = np.zeros_like(arr)
    for hue, intensity, weight in (
        (shadows_hue, shadows_intensity, shadow_w),
        (midtones_hue, midtones_intensity, midtone_w),
        (highlights_hue, highlights_intensity, highlight_w),
    ):
        if intensity:
            tint = _tint_color(hue)
            shift += (tint - arr) * weight[..., None] * (float(intensity) / 100.0) * 0.5
    result = np.clip(arr + shift, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


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

# Every exception littlecms can raise for a malformed/truncated/non-ICC "icc_profile" tag, or for a
# profile it can open but can't build a transform from -- one shared tuple so normalize_to_srgb()'s
# and gamut_warning_overlay()'s callers (app/backend.py, app/frontend_unified_app.py) all fall back
# on exactly the same set of failure modes instead of keeping their own copies in sync by hand.
ICC_TRANSFORM_ERRORS = (OSError, ValueError, ImageCms.PyCMSError)


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


def clipping_warning_overlay(
    image: Image.Image,
    shadow_color: tuple[int, int, int] = (0, 80, 255),
    highlight_color: tuple[int, int, int] = (255, 40, 40),
    shadow_threshold: int = 0,
    highlight_threshold: int = 255,
) -> Image.Image:
    """A histogram clipping warning, the same idea as Lightroom's blown-highlight/blocked-shadow
    triangles, but as a full preview overlay (matching this app's existing overlay-checkbox
    pattern, e.g. gamut_warning_overlay above and the Masks tab's selection preview) rather than a
    tiny toggleable icon on the histogram itself. Any channel at or below `shadow_threshold`
    (default: pure black, 0) marks the whole pixel as a blocked shadow; any channel at or above
    `highlight_threshold` (default: pure white, 255) marks it a blown highlight -- a channel-level
    clip is treated as a whole-pixel warning, the same convention Lightroom's warning uses."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    shadow_clip = (arr <= shadow_threshold).any(axis=2)
    highlight_clip = (arr >= highlight_threshold).any(axis=2)
    result = arr.copy()
    result[shadow_clip] = shadow_color
    result[highlight_clip] = highlight_color
    return Image.fromarray(result, mode="RGB")
