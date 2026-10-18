"""Float ICC conversion for matrix/TRC profiles (B4): a 16-bit source with an embedded Adobe RGB,
ProPhoto, Display P3 (or similar) profile converted to sRGB without going through 8 bits.

LittleCMS through Pillow only converts 8-bit images, so the 16-bit path used to convert the
quantized image and carry the change over to the float pixels. For the common matrix/TRC
("matrix-shaper") profiles the conversion is simple enough to do directly in float: each channel's
tone curve (``rTRC``/``gTRC``/``bTRC``: a gamma, a table or a parametric curve) to linear light,
the colorant matrix (``rXYZ``/``gXYZ``/``bXYZ``, already adapted to the D50 connection space) to
XYZ, the inverse of sRGB's own matrix back to linear sRGB, and sRGB's curve -- relative colorimetric,
out-of-gamut colours clipped, the same result LittleCMS gives for these profiles. sRGB's matrix and
curve are read from the very profile the pipeline tags its output with
(``professional_color.srgb_icc_bytes()``), so both sides agree.

Profiles with lookup tables (``A2B0``, camera and printer profiles) aren't handled here
(:func:`parse_matrix_trc` returns None); the caller keeps the 8-bit LittleCMS conversion for them.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np

_CURVE_SAMPLES = 4096


@dataclass(frozen=True)
class MatrixTRC:
    matrix: np.ndarray  # 3x3: linear RGB -> XYZ (D50)
    curves: tuple  # three callables: encoded 0..1 -> linear 0..1 (float arrays)
    inverse_curves: tuple  # linear -> encoded


def _s15(data: bytes, offset: int) -> float:
    return struct.unpack(">i", data[offset:offset + 4])[0] / 65536.0


def _tags(data: bytes) -> dict:
    count = struct.unpack(">I", data[128:132])[0]
    tags = {}
    for index in range(min(count, 200)):
        entry = 132 + 12 * index
        signature, offset, size = struct.unpack(">4sII", data[entry:entry + 12])
        tags[signature.decode("latin-1")] = data[offset:offset + size]
    return tags


def _xyz(tag: bytes) -> np.ndarray:
    if tag[:4] != b"XYZ ":
        raise ValueError("not an XYZ tag")
    return np.array([_s15(tag, 8), _s15(tag, 12), _s15(tag, 16)])


def _curve(tag: bytes) -> Callable[[np.ndarray], np.ndarray]:
    kind = tag[:4]
    if kind == b"curv":
        count = struct.unpack(">I", tag[8:12])[0]
        if count == 0:
            return lambda x: x
        if count == 1:
            gamma = struct.unpack(">H", tag[12:14])[0] / 256.0
            return lambda x, g=gamma: np.power(np.clip(x, 0.0, 1.0), g)
        table = np.frombuffer(tag[12:12 + 2 * count], dtype=">u2").astype(np.float64) / 65535.0
        grid = np.linspace(0.0, 1.0, count)
        return lambda x, g=grid, t=table: np.interp(x, g, t)
    if kind == b"para":
        function = struct.unpack(">H", tag[8:10])[0]
        n = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(function)
        if n is None:
            raise ValueError("unknown parametric curve")
        p = [_s15(tag, 12 + 4 * i) for i in range(n)]
        g = p[0]
        if function == 0:
            return lambda x: np.power(np.clip(x, 0.0, 1.0), g)
        if function == 1:
            a, b = p[1], p[2]
            return lambda x: np.where(x >= -b / a, np.power(np.clip(a * x + b, 0.0, None), g), 0.0)
        if function == 2:
            a, b, c = p[1], p[2], p[3]
            return lambda x: np.where(x >= -b / a, np.power(np.clip(a * x + b, 0.0, None), g) + c, c)
        if function == 3:
            a, b, c, d = p[1], p[2], p[3], p[4]
            return lambda x: np.where(x >= d, np.power(np.clip(a * x + b, 0.0, None), g), c * x)
        a, b, c, d, e, f = p[1], p[2], p[3], p[4], p[5], p[6]
        return lambda x: np.where(x >= d, np.power(np.clip(a * x + b, 0.0, None), g) + e, c * x + f)
    raise ValueError("unknown curve type")


def _inverse(curve: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Numerical inverse of a monotonic curve on 0..1 (sampled densely, interpolated)."""
    grid = np.linspace(0.0, 1.0, _CURVE_SAMPLES)
    values = np.maximum.accumulate(np.asarray(curve(grid), dtype=np.float64))
    return lambda y, v=values, gr=grid: np.interp(y, v, gr)


def parse_matrix_trc(icc_bytes: bytes) -> Optional[MatrixTRC]:
    """The matrix and curves of an RGB matrix/TRC profile, or None (a LUT profile, not RGB, damaged)."""
    try:
        if len(icc_bytes) < 132 or icc_bytes[16:20] != b"RGB " or icc_bytes[20:24] != b"XYZ ":
            return None
        tags = _tags(icc_bytes)
        if "A2B0" in tags or not all(t in tags for t in ("rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC")):
            return None
        matrix = np.stack([_xyz(tags["rXYZ"]), _xyz(tags["gXYZ"]), _xyz(tags["bXYZ"])], axis=1)
        curves = tuple(_curve(tags[name]) for name in ("rTRC", "gTRC", "bTRC"))
        if abs(np.linalg.det(matrix)) < 1e-6:
            return None
        return MatrixTRC(matrix, curves, tuple(_inverse(c) for c in curves))
    except (ValueError, struct.error, IndexError):
        return None


@lru_cache(maxsize=1)
def _srgb() -> MatrixTRC:
    from app.professional_color import srgb_icc_bytes

    parsed = parse_matrix_trc(srgb_icc_bytes())
    if parsed is None:  # pragma: no cover - LittleCMS's built-in sRGB is matrix/TRC
        raise RuntimeError("the sRGB profile isn't matrix/TRC")
    return parsed


def to_srgb_float(rgb: np.ndarray, icc_bytes: bytes) -> Optional[np.ndarray]:
    """`rgb` (float 0..1, in the profile's space) converted to sRGB in float, or None when the
    profile isn't a matrix/TRC one (the caller then uses LittleCMS on 8 bits)."""
    source = parse_matrix_trc(icc_bytes)
    if source is None:
        return None
    target = _srgb()
    linear = np.stack([source.curves[i](rgb[..., i].astype(np.float64)) for i in range(3)], axis=-1)
    to_srgb = np.linalg.inv(target.matrix) @ source.matrix
    srgb_linear = np.clip(linear @ to_srgb.T, 0.0, 1.0)
    encoded = np.stack([target.inverse_curves[i](srgb_linear[..., i]) for i in range(3)], axis=-1)
    return np.clip(encoded, 0.0, 1.0).astype(np.float32)
