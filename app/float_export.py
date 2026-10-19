"""Floating-point exports (F24): OpenEXR and PFM from the 16-bit path's float pixels.

Both files hold *linear-light* RGB (the sRGB / Display P3 transfer curve undone), the convention
compositing, 3D and HDR tools expect of a float image. Written here in NumPy + zlib -- OpenCV's
wheel ships no EXR encoder and the OpenEXR bindings would be a new native dependency:

- ``encode_exr()``: a single-part scanline OpenEXR 2 file, channels B/G/R (an EXR's channel list is
  sorted by name) as HALF floats, ZIP compression (zlib over blocks of 16 scanlines, after the
  format's byte split and delta predictor; a block that doesn't shrink is stored raw, as the spec
  allows). The colour space travels in the standard ``chromaticities`` attribute -- Rec. 709/sRGB
  primaries, or Display P3's for a P3 export -- since an EXR carries no ICC profile.
- ``encode_pfm()``: the Portable Float Map (``PF``, 32-bit little-endian floats, rows bottom-up).

Neither format has a place for EXIF, so the metadata policy doesn't apply to them.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

EXR_MAGIC = 20000630
EXR_HALF = 1
EXR_FLOAT = 2
EXR_NO_COMPRESSION = 0
EXR_ZIP_COMPRESSION = 3
ZIP_BLOCK_LINES = 16

# (red x, y, green x, y, blue x, y, white x, y)
CHROMATICITIES = {
    "sRGB": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290),
    "Display P3": (0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290),
}


def srgb_to_linear(array: np.ndarray) -> np.ndarray:
    """The sRGB transfer curve undone (Display P3 uses the same curve)."""
    array = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    return np.where(array <= 0.04045, array / 12.92, ((array + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _attribute(name: str, kind: str, value: bytes) -> bytes:
    return name.encode("ascii") + b"\0" + kind.encode("ascii") + b"\0" + struct.pack("<i", len(value)) + value


def _zip_predict(raw: bytes) -> bytes:
    """OpenEXR's ZIP pre-pass: even bytes then odd bytes, then each byte as a delta (+128) from the previous."""
    data = np.frombuffer(raw, dtype=np.uint8)
    split = np.concatenate([data[0::2], data[1::2]]).astype(np.int16)
    out = split.copy()
    out[1:] = (split[1:] - split[:-1] + 128) & 0xFF
    return out.astype(np.uint8).tobytes()


def encode_exr(linear_rgb: np.ndarray, color_space: str = "sRGB", pixel_type: int = EXR_HALF, compression: int = EXR_ZIP_COMPRESSION) -> bytes:
    """A linear HxWx3 float array -> an OpenEXR file's bytes."""
    array = np.asarray(linear_rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("encode_exr needs an HxWx3 array")
    if compression not in (EXR_NO_COMPRESSION, EXR_ZIP_COMPRESSION) or pixel_type not in (EXR_HALF, EXR_FLOAT):
        raise ValueError("unsupported EXR pixel type or compression")
    height, width = array.shape[:2]
    sample = np.dtype("<f2") if pixel_type == EXR_HALF else np.dtype("<f4")
    channels = b"".join(name + b"\0" + struct.pack("<iB3xii", pixel_type, 0, 1, 1) for name in (b"B", b"G", b"R")) + b"\0"
    header = struct.pack("<ii", EXR_MAGIC, 2) + b"".join([
        _attribute("channels", "chlist", channels),
        _attribute("chromaticities", "chromaticities", struct.pack("<8f", *CHROMATICITIES.get(color_space, CHROMATICITIES["sRGB"]))),
        _attribute("compression", "compression", bytes([compression])),
        _attribute("dataWindow", "box2i", struct.pack("<4i", 0, 0, width - 1, height - 1)),
        _attribute("displayWindow", "box2i", struct.pack("<4i", 0, 0, width - 1, height - 1)),
        _attribute("lineOrder", "lineOrder", b"\0"),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    ]) + b"\0"
    # Per scanline: all of B, then G, then R (the channel list's order).
    planar = np.ascontiguousarray(array[:, :, ::-1].transpose(0, 2, 1)).astype(sample)
    lines = ZIP_BLOCK_LINES if compression == EXR_ZIP_COMPRESSION else 1
    blocks = []
    for y in range(0, height, lines):
        raw = planar[y:y + lines].tobytes()
        data = raw
        if compression == EXR_ZIP_COMPRESSION:
            packed = zlib.compress(_zip_predict(raw), 6)
            if len(packed) < len(raw):
                data = packed
        blocks.append(struct.pack("<ii", y, len(data)) + data)
    offset = len(header) + 8 * len(blocks)
    table = bytearray()
    for block in blocks:
        table += struct.pack("<Q", offset)
        offset += len(block)
    return header + bytes(table) + b"".join(blocks)


def encode_pfm(linear_rgb: np.ndarray) -> bytes:
    """A linear HxWx3 float array -> a little-endian colour PFM (rows stored bottom-up)."""
    array = np.asarray(linear_rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("encode_pfm needs an HxWx3 array")
    height, width = array.shape[:2]
    return f"PF\n{width} {height}\n-1.0\n".encode("ascii") + np.ascontiguousarray(array[::-1]).astype("<f4").tobytes()
