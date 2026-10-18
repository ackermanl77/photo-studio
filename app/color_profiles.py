"""Output color spaces for export: sRGB (the working space) and Display P3.

The pipeline normally works in sRGB (app.pipeline normalizes an embedded profile into it). A
Display P3 export -- the wide-gamut space of current phones, Macs and many monitors -- renders in a
Display P3 working space instead (see the bottom of this module, B14), so colours outside sRGB that
the source holds survive; the result is tagged with a Display P3 ICC profile. A result that is still
sRGB (an in-process render) is converted from sRGB into P3 at export.

No profile file is bundled: display_p3_icc_bytes() builds a standard ICC v4 matrix/TRC display
profile (the P3 primaries, a D65 white adapted to the D50 connection space with Bradford, and the
sRGB transfer curve that Display P3 shares), the same structure as Apple's Display P3.icc.
convert_srgb_to_display_p3() does the conversion itself in float math (linearize, 3x3 matrix,
re-encode), so 8-bit and 16-bit exports convert identically; tests/test_color_profiles.py checks it
against littlecms using the generated profile.
"""
from __future__ import annotations

import struct
from functools import lru_cache

import numpy as np

COLOR_SPACES = ("sRGB", "Display P3")

_D65 = (0.3127, 0.3290)
_SRGB_PRIMARIES = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
_P3_PRIMARIES = ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060))
_D50_XYZ = np.array([0.9642, 1.0, 0.8249])
_BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
])


def _xyz(xy) -> np.ndarray:
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def rgb_to_xyz_matrix(primaries, white=_D65) -> np.ndarray:
    """The RGB -> XYZ matrix of an RGB space with these chromaticities (Y of white = 1)."""
    columns = np.stack([_xyz(p) for p in primaries], axis=1)
    scale = np.linalg.solve(columns, _xyz(white))
    return columns * scale


def bradford_adaptation(source_white: np.ndarray, target_white: np.ndarray) -> np.ndarray:
    cone_source = _BRADFORD @ source_white
    cone_target = _BRADFORD @ target_white
    return np.linalg.inv(_BRADFORD) @ np.diag(cone_target / cone_source) @ _BRADFORD


SRGB_TO_XYZ = rgb_to_xyz_matrix(_SRGB_PRIMARIES)
P3_TO_XYZ = rgb_to_xyz_matrix(_P3_PRIMARIES)
# Linear sRGB -> linear Display P3 (same D65 white: no adaptation needed between them).
SRGB_TO_P3 = np.linalg.inv(P3_TO_XYZ) @ SRGB_TO_XYZ


def srgb_decode(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def srgb_encode(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1 / 2.4) - 0.055)


def convert_srgb_to_display_p3(rgb: np.ndarray) -> np.ndarray:
    """Float sRGB (0..1, ..., 3) -> float Display P3 (0..1), float32."""
    linear = srgb_decode(rgb[..., :3])
    p3 = linear @ SRGB_TO_P3.T
    return srgb_encode(p3).astype(np.float32)


def convert_srgb_to_display_p3_uint8(rgb8: np.ndarray) -> np.ndarray:
    """8-bit sRGB -> 8-bit Display P3 through a 256^3-free path: a per-channel lookup can't express
    the matrix, so the float conversion runs on the pixels (rounded back to 8 bits)."""
    out = convert_srgb_to_display_p3(np.asarray(rgb8, dtype=np.float32) / 255.0)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


# --- the ICC profile ---------------------------------------------------------------------------------

def _s15f16(value: float) -> bytes:
    return struct.pack(">i", int(round(value * 65536.0)))


def _mluc(text: str) -> bytes:
    encoded = text.encode("utf-16-be")
    return b"mluc" + b"\0" * 4 + struct.pack(">II", 1, 12) + b"enUS" + struct.pack(">II", len(encoded), 28) + encoded


def _xyz_tag(xyz) -> bytes:
    return b"XYZ " + b"\0" * 4 + b"".join(_s15f16(v) for v in xyz)


def _sf32(matrix: np.ndarray) -> bytes:
    return b"sf32" + b"\0" * 4 + b"".join(_s15f16(v) for v in np.asarray(matrix).ravel())


def _para_srgb() -> bytes:
    # Parametric curve type 3: Y = (aX + b)^g for X >= d, Y = cX below -- the sRGB curve.
    params = (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045)
    return b"para" + b"\0" * 4 + struct.pack(">HH", 3, 0) + b"".join(_s15f16(v) for v in params)


def _pad4(data: bytes) -> bytes:
    return data + b"\0" * (-len(data) % 4)


@lru_cache(maxsize=1)
def display_p3_icc_bytes() -> bytes:
    """An ICC v4.3 Display P3 display profile (see the module docstring)."""
    adapt = bradford_adaptation(_xyz(_D65), _D50_XYZ)
    colorants = adapt @ P3_TO_XYZ  # columns: red, green, blue in the D50 PCS
    trc = _para_srgb()
    tags = [
        (b"desc", _mluc("Display P3")),
        (b"cprt", _mluc("No copyright, use freely")),
        (b"wtpt", _xyz_tag(_D50_XYZ)),
        (b"chad", _sf32(adapt)),
        (b"rXYZ", _xyz_tag(colorants[:, 0])),
        (b"gXYZ", _xyz_tag(colorants[:, 1])),
        (b"bXYZ", _xyz_tag(colorants[:, 2])),
        (b"rTRC", trc),
        (b"gTRC", trc),
        (b"bTRC", trc),
    ]
    table_size = 4 + 12 * len(tags)
    offset = 128 + table_size
    table = struct.pack(">I", len(tags))
    data = b""
    placed: dict[bytes, int] = {}
    for signature, body in tags:
        if body in placed:  # the three TRCs share one curve, as ICC allows
            start = placed[body]
        else:
            start = offset + len(data)
            placed[body] = start
            data += _pad4(body)
        table += signature + struct.pack(">II", start, len(body))
    size = offset + len(data)
    header = (
        struct.pack(">I", size) + b"lcms" + struct.pack(">I", 0x04300000) + b"mntr" + b"RGB " + b"XYZ "
        + struct.pack(">6H", 2026, 1, 1, 0, 0, 0) + b"acsp" + b"\0" * 4 + struct.pack(">I", 0)
        + b"\0" * 8 + b"\0" * 8 + struct.pack(">I", 0) + b"".join(_s15f16(v) for v in _D50_XYZ)
        + b"\0" * 4 + b"\0" * 16 + b"\0" * 28
    )
    assert len(header) == 128
    return header + table + data


def output_icc(color_space: str, current_icc: bytes | None) -> bytes | None:
    """The profile an exported file carries: Display P3's for a P3 export, else the image's own."""
    return display_p3_icc_bytes() if color_space == "Display P3" else current_icc


# --- Display P3 as the working space (B14) ---------------------------------------------------------
# An sRGB working space clips every colour outside sRGB on the way in (normalize_to_srgb(), and RAW
# decoded straight to sRGB), so converting the finished pixels to P3 at export could never put a
# colour outside sRGB into the file. A Display P3 export therefore renders in a Display P3 working
# space instead: RAW decoded to P3 by LibRaw, an embedded profile converted to P3 rather than to sRGB,
# an untagged (sRGB) source converted to P3 numerically. Display P3 shares sRGB's transfer curve, so
# every later step sees encoded values on the same scale; hue-based tools (HSL bands, the skin mask)
# see P3 hues, which differ a little from sRGB ones for strongly saturated colours.

WORKING_SPACES = COLOR_SPACES


def is_display_p3_icc(icc: bytes | None) -> bool:
    """True when `icc` is this module's own Display P3 profile -- how app.export recognises pixels
    the pipeline already rendered in P3 (and must not convert from sRGB a second time)."""
    return bool(icc) and icc == display_p3_icc_bytes()


def normalize_to_display_p3(image, icc_bytes: bytes):
    """`image` (8-bit RGB PIL) converted from the profile `icc_bytes` describes into Display P3,
    relative colorimetric like normalize_to_srgb() -- but a wide-gamut source (Adobe RGB, ProPhoto,
    a camera profile) keeps everything inside P3 instead of being clipped to sRGB."""
    import io

    from PIL import ImageCms

    source = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
    target = ImageCms.ImageCmsProfile(io.BytesIO(display_p3_icc_bytes()))
    return ImageCms.profileToProfile(
        image.convert("RGB"), source, target,
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB",
    )
