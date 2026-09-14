from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".dng", ".pef", ".srw"}


def is_raw_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def open_image(path: str | Path) -> Image.Image:
    source = Path(path)
    if not is_raw_path(source):
        return Image.open(source).convert("RGB")
    try:
        import rawpy
    except ImportError as error:
        raise RuntimeError(
            "RAW support is optional. Install requirements-raw.txt to open camera RAW files."
        ) from error
    with rawpy.imread(str(source)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
    return Image.fromarray(rgb, mode="RGB")


def open_bytes(content: bytes, filename: str) -> Image.Image:
    if not is_raw_path(filename):
        return Image.open(BytesIO(content)).convert("RGB")
    try:
        import rawpy
    except ImportError as error:
        raise RuntimeError(
            "RAW support is optional. Install requirements-raw.txt to open camera RAW files."
        ) from error
    import tempfile

    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(content)
        temporary.flush()
        with rawpy.imread(temporary.name) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
    return Image.fromarray(rgb, mode="RGB")
