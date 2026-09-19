from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.major_features import PhotoCatalog

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def scan_folder(folder: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(folder)
    paths = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def image_info(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        with Image.open(source) as image:
            return {"path": str(source), "name": source.name, "width": image.width, "height": image.height, "format": image.format, "bytes": source.stat().st_size}
    except (OSError, UnidentifiedImageError) as error:
        return {"path": str(source), "name": source.name, "error": str(error)}


def thumbnail(path: str | Path, destination: str | Path, size: tuple[int, int] = (320, 320)) -> Path:
    source = Path(path)
    target_root = Path(destination)
    target_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    target = target_root / f"{digest}.jpg"
    with Image.open(source) as image:
        preview = image.convert("RGB")
        preview.thumbnail(size, Image.Resampling.LANCZOS)
        preview.save(target, format="JPEG", quality=84, optimize=True)
    return target


def index_folder(folder: str | Path, database: str | Path, recursive: bool = True, progress=None) -> list[dict[str, object]]:
    """`progress`, when given, is called as `progress(index, total, info)` after each file --
    same shape as app.batch_release.process_files()'s own `progress` callback -- so the frontend's
    "Scan Folder" button can drive an st.progress() bar the same way Run Batch already does."""
    catalog = PhotoCatalog(database)
    paths = scan_folder(folder, recursive)
    total = len(paths)
    records = []
    for index, path in enumerate(paths, 1):
        info = image_info(path)
        if "error" not in info:
            catalog.upsert(path)
            records.append(info)
        if progress:
            progress(index, total, info)
    return records
