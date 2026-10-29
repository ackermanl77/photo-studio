"""Photoshop's Image > Adjustments that the rest of the pipeline doesn't already cover (PS1):
Exposure's Offset and Gamma Correction, Levels, Channel Mixer, Photo Filter, Hue/Saturation's
Colorize, Gradient Map, Posterize, Threshold and Equalize.

Pure functions on float RGB arrays in 0..1 (H x W x 3), so the same code serves the 8-bit preview
and the 16-bit float export. ``apply_adjustments`` runs the ones `options` turns on, in the order
above -- after the LUT and before Invert (app.pipeline), like a stack of adjustment layers on top
of the photo's own grade.
"""
from __future__ import annotations

import numpy as np

# Photoshop's Photo Filter presets, with its colours.
PHOTO_FILTERS = {
    "Warming Filter (85)": "#EC8A00", "Warming Filter (LBA)": "#FA9600", "Warming Filter (81)": "#EBB113",
    "Cooling Filter (80)": "#006DFF", "Cooling Filter (LBB)": "#005DFF", "Cooling Filter (82)": "#00B5FF",
    "Red": "#EA1A1A", "Orange": "#F38417", "Yellow": "#F9E31C", "Green": "#19C919", "Cyan": "#1DCBEA",
    "Blue": "#1D35EA", "Violet": "#9B1DEA", "Magenta": "#E31CE3", "Sepia": "#AC7A33", "Deep Red": "#FF0000",
    "Deep Blue": "#0022CD", "Deep Emerald": "#008C00", "Deep Yellow": "#FFD500", "Underwater": "#00C1B1",
}
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec. 709 / sRGB weights
_MIXER_CHANNELS = ("r", "g", "b")


def _rgb(hex_color: str) -> np.ndarray:
    value = hex_color.lstrip("#")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def _luma(array: np.ndarray) -> np.ndarray:
    return array @ _LUMA


def exposure_offset_gamma(array: np.ndarray, offset: float, gamma: float) -> np.ndarray:
    """Exposure's Offset (shifts the shadows, -0.5..+0.5) and Gamma Correction (out = in^(1/gamma))."""
    result = np.clip(array + offset, 0.0, 1.0)
    return result ** (1.0 / gamma) if gamma != 1.0 else result


def levels(array: np.ndarray, in_black: float, in_white: float, gamma: float, out_black: float, out_white: float) -> np.ndarray:
    """Levels on all channels: input black/white points (0-255), midtone gamma, output range."""
    low, high = in_black / 255.0, max(in_white, in_black + 1) / 255.0
    result = np.clip((array - low) / (high - low), 0.0, 1.0)
    if gamma != 1.0:
        result = result ** (1.0 / gamma)
    return out_black / 255.0 + result * ((out_white - out_black) / 255.0)


def channel_mixer(array: np.ndarray, matrix: dict) -> np.ndarray:
    """matrix["r"]["g"] is how much (percent) of the source Green goes into the Red output."""
    weights = np.array([[matrix[out][src] / 100.0 for src in _MIXER_CHANNELS] for out in _MIXER_CHANNELS], dtype=np.float32)
    return np.clip(array @ weights.T, 0.0, 1.0)


def photo_filter(array: np.ndarray, color: str, density: float, preserve_luminosity: bool) -> np.ndarray:
    """Photo Filter: the photo seen through a coloured filter -- multiplied by the colour, mixed in at
    `density` percent; Preserve Luminosity keeps each pixel's brightness."""
    tint = _rgb(color)
    filtered = array * (1.0 - density / 100.0) + array * tint * (density / 100.0)
    if preserve_luminosity:
        before, after = _luma(array), _luma(filtered)
        scale = np.divide(before, after, out=np.ones_like(before), where=after > 1e-6)
        filtered = filtered * scale[..., None]
    return np.clip(filtered, 0.0, 1.0)


def colorize(array: np.ndarray, hue: float, saturation: float, lightness: float) -> np.ndarray:
    """Hue/Saturation's Colorize: every pixel takes one hue (0-360) and saturation (0-100), keeping
    its own lightness, shifted by `lightness` (-100 toward black .. +100 toward white)."""
    import cv2

    level = _luma(array)
    if lightness > 0:
        level = level + (1.0 - level) * (lightness / 100.0)
    elif lightness < 0:
        level = level * (1.0 + lightness / 100.0)
    hls = np.dstack([np.full_like(level, hue % 360.0), level, np.full_like(level, saturation / 100.0)]).astype(np.float32)
    return np.clip(cv2.cvtColor(hls, cv2.COLOR_HLS2RGB), 0.0, 1.0)


def gradient_map(array: np.ndarray, shadow: str, highlight: str, amount: float) -> np.ndarray:
    """Gradient Map: brightness mapped onto a two-colour gradient, mixed in at `amount` percent."""
    level = _luma(array)[..., None]
    mapped = _rgb(shadow) * (1.0 - level) + _rgb(highlight) * level
    return array * (1.0 - amount / 100.0) + mapped * (amount / 100.0)


def posterize(array: np.ndarray, levels_count: int) -> np.ndarray:
    steps = max(2, int(levels_count)) - 1
    return np.round(array * steps) / steps


def threshold(array: np.ndarray, level: float) -> np.ndarray:
    """Black where brightness is below `level` (1-255), white from it up."""
    mask = (_luma(array) >= level / 255.0).astype(np.float32)
    return np.repeat(mask[..., None], 3, axis=2)


def equalize(array: np.ndarray) -> np.ndarray:
    """Spread the brightness evenly over the range: one mapping from the overall brightness
    histogram, applied to every channel (so colours don't shift as with per-channel equalizing)."""
    histogram, _edges = np.histogram(_luma(array), bins=256, range=(0.0, 1.0))
    cdf = np.cumsum(histogram).astype(np.float64)
    if cdf[-1] <= 0:
        return array
    first = cdf[np.nonzero(cdf)[0][0]]
    mapping = np.clip((cdf - first) / max(cdf[-1] - first, 1.0), 0.0, 1.0).astype(np.float32)
    positions = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    return np.interp(array, positions, mapping).astype(np.float32)


# The Channel Mixer's options, spelled out: (output, source) -> option name.
MIXER_OPTIONS = {
    ("r", "r"): "mixer_r_r", ("r", "g"): "mixer_r_g", ("r", "b"): "mixer_r_b",
    ("g", "r"): "mixer_g_r", ("g", "g"): "mixer_g_g", ("g", "b"): "mixer_g_b",
    ("b", "r"): "mixer_b_r", ("b", "g"): "mixer_b_g", ("b", "b"): "mixer_b_b",
}


def mixer_matrix(options: dict) -> dict:
    return {out: {src: float(options.get(MIXER_OPTIONS[out, src], 100.0 if out == src else 0.0)) for src in _MIXER_CHANNELS}
            for out in _MIXER_CHANNELS}


def adjustments_active(options: dict) -> bool:
    return any((
        options.get("exposure_offset", 0.0), options.get("exposure_gamma", 1.0) != 1.0, options.get("levels_enabled"),
        options.get("mixer_enabled"), options.get("photo_filter", "None") not in (None, "None"),
        options.get("colorize_enabled"), options.get("gradient_map_amount", 0.0), options.get("posterize_levels", 0),
        options.get("threshold_enabled"), options.get("equalize"),
    ))


def apply_adjustments(array: np.ndarray, options: dict) -> np.ndarray:
    """The adjustments `options` turns on, in Photoshop's menu order, on a float RGB array."""
    array = np.asarray(array, dtype=np.float32)
    offset, gamma = float(options.get("exposure_offset", 0.0)), float(options.get("exposure_gamma", 1.0))
    if offset or gamma != 1.0:
        array = exposure_offset_gamma(array, offset, gamma)
    if options.get("levels_enabled"):
        array = levels(array, options.get("levels_in_black", 0), options.get("levels_in_white", 255),
                       options.get("levels_gamma", 1.0), options.get("levels_out_black", 0), options.get("levels_out_white", 255))
    if options.get("mixer_enabled"):
        array = channel_mixer(array, mixer_matrix(options))
    if options.get("photo_filter", "None") in PHOTO_FILTERS:
        array = photo_filter(array, PHOTO_FILTERS[options["photo_filter"]], options.get("photo_filter_density", 25.0),
                             options.get("photo_filter_luminosity", True))
    if options.get("colorize_enabled"):
        array = colorize(array, options.get("colorize_hue", 0.0), options.get("colorize_saturation", 25.0),
                         options.get("colorize_lightness", 0.0))
    if options.get("gradient_map_amount", 0.0):
        array = gradient_map(array, options.get("gradient_map_shadow", "#000000"), options.get("gradient_map_highlight", "#FFFFFF"),
                             options["gradient_map_amount"])
    if options.get("posterize_levels", 0):
        array = posterize(array, options["posterize_levels"])
    if options.get("threshold_enabled"):
        array = threshold(array, options.get("threshold_level", 128))
    if options.get("equalize"):
        array = equalize(array)
    return np.clip(array, 0.0, 1.0).astype(np.float32)
