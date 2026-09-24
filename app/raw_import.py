from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".dng", ".pef", ".srw"}


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
    return image.convert("RGB")


def _decode_raw(source) -> Image.Image:
    """`source` is a path string or a file-like object -- rawpy.imread() accepts either. Bytes go
    through BytesIO rather than a NamedTemporaryFile: on Windows a NamedTemporaryFile can't be
    reopened by name while it's still open, so LibRaw failed to read it (LibRawIOError) for every
    RAW upload. LibRaw's own errors (a corrupt or unsupported RAW) are re-raised as RuntimeError --
    the exception type the frontend's file-open handler and app/backend.py's /process route both
    already turn into a clean message, instead of a traceback / HTML 500."""
    try:
        import rawpy
    except ImportError as error:
        raise RuntimeError(
            "RAW support is optional. Install requirements-raw.txt to open camera RAW files."
        ) from error
    try:
        with rawpy.imread(source) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
    except rawpy.LibRawError as error:
        raise RuntimeError(f"Could not decode this camera RAW file: {error}") from error
    return Image.fromarray(rgb, mode="RGB")


def open_image(path: str | Path) -> Image.Image:
    source = Path(path)
    if not is_raw_path(source):
        return _decode_standard(source)
    return _decode_raw(str(source))


def open_bytes(content: bytes, filename: str) -> Image.Image:
    if not is_raw_path(filename):
        return _decode_standard(BytesIO(content))
    return _decode_raw(BytesIO(content))
