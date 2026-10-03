from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".dng", ".pef", ".srw"}
# Every non-RAW format the decoder below opens. SUPPORTED_EXTENSIONS is the one list every place
# that accepts a photo derives from (the main uploader, the Batch uploader, the Library scan, the
# CLI's --input-dir) -- they used to keep separate hand-written lists, so the main uploader refused
# Fuji/Panasonic/Olympus/Pentax/Samsung RAW files the decoder supports and Batch took only JPG/PNG.
STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = STANDARD_EXTENSIONS | RAW_EXTENSIONS


def uploader_types(include_raw: bool = True) -> list[str]:
    """st.file_uploader(type=...) form of the supported extensions (no leading dot), sorted."""
    extensions = SUPPORTED_EXTENSIONS if include_raw else STANDARD_EXTENSIONS
    return sorted(extension.lstrip(".") for extension in extensions)


def is_raw_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def _decode_standard(source) -> Image.Image:
    """PIL decode for every non-RAW format. Applies the EXIF Orientation tag at decode time -- a
    phone photo is usually stored landscape with an "Orientation: rotate 90" tag rather than with
    rotated pixels, and every step downstream (masks, crop, the canvas, export) works on raw pixel
    geometry, so without this it showed up sideways. exif_transpose() also removes the tag from the
    image's own EXIF, so an exported file carrying that EXIF isn't rotated a second time by a
    viewer. (RAW needs no equivalent: rawpy's postprocess() already applies the camera's flip.)"""
    image = Image.open(source)
    ImageOps.exif_transpose(image, in_place=True)
    if image.mode in _HIGH_BIT_GRAY_MODES:
        return _high_bit_gray_to_rgb(image)
    return image.convert("RGB")


# Pillow's own convert("RGB") *clips* these single-channel high-bit modes to 0-255 instead of
# scaling them, so a 16-bit grayscale PNG/TIFF (scans, astro, scientific images) decoded with
# ~99.6 % of its pixels pure white.
_HIGH_BIT_GRAY_MODES = {"I;16", "I;16L", "I;16B", "I;16N", "I", "F"}


def _high_bit_gray_to_rgb(image: Image.Image) -> Image.Image:
    values = np.asarray(image, dtype=np.float64)
    if image.mode.startswith("I;16"):
        peak = 65535.0
    elif image.mode == "F":
        finite_max = float(np.nanmax(values)) if values.size else 1.0
        peak = 1.0 if finite_max <= 1.0 else finite_max
    else:  # 32-bit integer: usually a 16-bit file Pillow opened as "I"
        finite_max = float(values.max()) if values.size else 255.0
        peak = 255.0 if finite_max <= 255 else (65535.0 if finite_max <= 65535 else finite_max)
    scaled = np.clip(np.nan_to_num(values) / peak * 255.0 + 0.5, 0, 255).astype(np.uint8)
    result = Image.fromarray(scaled, mode="L").convert("RGB")
    result.info.update({key: value for key, value in image.info.items() if key in ("exif", "icc_profile", "dpi")})
    return result


def decode_raw_array(source, bits: int = 8) -> np.ndarray:
    """`source` is a path string or a file-like object -- rawpy.imread() accepts either. Bytes go
    through BytesIO rather than a NamedTemporaryFile: on Windows a NamedTemporaryFile can't be
    reopened by name while it's still open, so LibRaw failed to read it (LibRawIOError) for every
    RAW upload. LibRaw's own errors (a corrupt or unsupported RAW) are re-raised as RuntimeError --
    the exception type the frontend's file-open handler and app/backend.py's /process route both
    already turn into a clean message, instead of a traceback / HTML 500.
    `bits` is 8 (the normal decode) or 16 (app.high_bit's 16-bit export path, B4): the same
    demosaic, white balance and highlight handling, just without the 8-bit quantization."""
    try:
        import rawpy
    except ImportError as error:
        raise RuntimeError(
            "RAW support is optional. Install requirements-raw.txt to open camera RAW files."
        ) from error
    try:
        with rawpy.imread(source) as raw:
            # highlight_mode=Blend: blend the clipped channels back in instead of LibRaw's
            # default hard clip, so blown highlights fade to neutral rather than turning magenta
            # or flat white at decode time (the Highlights slider can't recover clipped data).
            blend = getattr(getattr(rawpy, "HighlightMode", None), "Blend", None)
            extra = {"highlight_mode": blend} if blend is not None else {}
            return raw.postprocess(use_camera_wb=True, output_bps=16 if bits == 16 else 8, **extra)
    except rawpy.LibRawError as error:
        raise RuntimeError(f"Could not decode this camera RAW file: {error}") from error


def _decode_raw(source) -> Image.Image:
    return Image.fromarray(decode_raw_array(source), mode="RGB")


def open_image(path: str | Path) -> Image.Image:
    source = Path(path)
    if not is_raw_path(source):
        return _decode_standard(source)
    return _decode_raw(str(source))


def open_bytes(content: bytes, filename: str) -> Image.Image:
    if not is_raw_path(filename):
        return _decode_standard(BytesIO(content))
    return _decode_raw(BytesIO(content))
