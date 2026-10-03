"""High-precision ("16-bit") working data: decode, carry and quantize a float32 copy of a photo.

The normal pipeline works on 8-bit PIL images, and the interactive preview stays that way. When an
export asks for 16 bits (B4 in the improvement report), app.pipeline.run_pipeline() runs with
`options["bit_depth"] == 16`: PhotoProcessor then keeps a float32 HxWx3 array (sRGB-encoded,
0..1) alongside its 8-bit image and the tone, colour, mask-blend, geometry, crop and vignette steps
compute on that array directly (see PhotoProcessor's "high-precision mode"). The array rides on the
result PIL image as an attribute (attach()/attached()) so app.export.encode_export() can write a
genuine 16-bit PNG or TIFF from it.

Sources: decode_high_bit() returns the float array for a file that really holds more than 8 bits
-- 16-bit PNG/TIFF (RGB or grayscale) and camera RAW (rawpy at output_bps=16) -- and None for an
8-bit file, whose float copy is then simply its 8-bit values / 255 (still worth it: the float
steps no longer re-quantize between each other, which is where banding comes from).
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

_ATTRIBUTE = "_photo_high_bit"
_ORIENTATION = 0x0112


def attach(image: Image.Image, array: np.ndarray | None) -> Image.Image:
    """Store `array` (float32 HxWx3, 0..1) on `image` -- lost on image.copy(), deliberately: a
    copy that later gets edited in 8 bits must not export stale high-precision pixels."""
    setattr(image, _ATTRIBUTE, array)
    return image


def attached(image: Image.Image) -> np.ndarray | None:
    array = getattr(image, _ATTRIBUTE, None)
    if array is None or array.shape[:2] != (image.height, image.width):
        return None
    return array


def quantize8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(array, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def to_uint16(array: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(array, dtype=np.float32) * 65535.0), 0, 65535).astype(np.uint16)


def from_image(image: Image.Image) -> np.ndarray:
    """An 8-bit image's float copy (0..1)."""
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def to_image(array: np.ndarray, info_from: Image.Image | None = None) -> Image.Image:
    """The 8-bit PIL image for a float array, with `info_from`'s EXIF/ICC (and the array attached)."""
    image = Image.fromarray(quantize8(array), mode="RGB")
    if info_from is not None:
        image.info.update({key: value for key, value in info_from.info.items() if key in ("exif", "icc_profile", "dpi")})
    return attach(image, np.asarray(array, dtype=np.float32))


# EXIF Orientation -> the same pixel transform ImageOps.exif_transpose() applies to the 8-bit decode.
def _orient(array: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 2:
        return array[:, ::-1]
    if orientation == 3:
        return array[::-1, ::-1]
    if orientation == 4:
        return array[::-1]
    if orientation == 5:
        return np.swapaxes(array, 0, 1)
    if orientation == 6:
        return np.rot90(array, -1)
    if orientation == 7:
        return np.swapaxes(array, 0, 1)[::-1, ::-1]
    if orientation == 8:
        return np.rot90(array, 1)
    return array


def _decode_standard_high_bit(content: bytes) -> np.ndarray | None:
    import cv2

    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype == np.uint8:
        return None
    if decoded.dtype == np.uint16:
        values = decoded.astype(np.float32) / 65535.0
    elif decoded.dtype in (np.float32, np.float64):
        values = np.nan_to_num(decoded.astype(np.float32))
        peak = float(values.max()) if values.size else 1.0
        values = values / peak if peak > 1.0 else values
    else:
        return None
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=2)
    elif values.shape[2] == 1:
        values = np.repeat(values, 3, axis=2)
    else:
        values = values[..., 2::-1] if values.shape[2] >= 3 else np.repeat(values[..., :1], 3, axis=2)  # BGR(A) -> RGB
    try:
        orientation = int(Image.open(BytesIO(content)).getexif().get(_ORIENTATION, 1))
    except (OSError, ValueError, SyntaxError, TypeError):
        orientation = 1
    return np.ascontiguousarray(np.clip(_orient(values, orientation), 0.0, 1.0), dtype=np.float32)


def decode_high_bit(content: bytes, filename: str) -> np.ndarray | None:
    """The float32 (0..1) pixels of a file holding more than 8 bits per channel, else None."""
    from app.raw_import import decode_raw_array, is_raw_path

    if is_raw_path(filename):
        return decode_raw_array(BytesIO(content), bits=16).astype(np.float32) / 65535.0
    return _decode_standard_high_bit(content)
