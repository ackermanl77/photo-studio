from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.major_features import PhotoCatalog
from app.raw_import import SUPPORTED_EXTENSIONS, is_raw_path
from app.xmp_sidecar import REJECTED_TAG, read_sidecar, write_sidecar

# Every format the editor opens, camera RAW included (it used to skip RAW entirely).
IMAGE_EXTENSIONS = SUPPORTED_EXTENSIONS


def scan_folder(folder: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(folder)
    paths = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def image_info(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if is_raw_path(source):
        try:
            import rawpy

            with rawpy.imread(str(source)) as raw:
                sizes = raw.sizes
            return {"path": str(source), "name": source.name, "width": sizes.width, "height": sizes.height, "format": "RAW", "bytes": source.stat().st_size}
        except ImportError:
            return {"path": str(source), "name": source.name, "width": None, "height": None, "format": "RAW", "bytes": source.stat().st_size}
        except Exception as error:  # noqa: BLE001 -- LibRaw's own error types; a bad file is just skipped
            return {"path": str(source), "name": source.name, "error": str(error)}
    try:
        with Image.open(source) as image:
            return {"path": str(source), "name": source.name, "width": image.width, "height": image.height, "format": image.format, "bytes": source.stat().st_size}
    except (OSError, UnidentifiedImageError) as error:
        return {"path": str(source), "name": source.name, "error": str(error)}


def save_rating(catalog: PhotoCatalog, path: str | Path, rating: int, tags, write_xmp: bool = False) -> str | None:
    """Store a rating/tags change in the catalog and, with `write_xmp`, in the photo's XMP
    sidecar too (the "rejected" tag becomes xmp:Rating -1). Returns a message when the sidecar
    couldn't be written -- the catalog is updated either way."""
    tags = list(tags)
    catalog.upsert(path, rating=rating, tags=tags)
    if not write_xmp:
        return None
    try:
        write_sidecar(path, rating, [t for t in tags if t != REJECTED_TAG], rejected=REJECTED_TAG in tags)
    except (OSError, ValueError) as error:
        return f"Couldn't write the XMP sidecar: {error}"
    return None


def thumbnail(path: str | Path, destination: str | Path, size: tuple[int, int] = (320, 320)) -> Path:
    """A cached JPEG thumbnail of `path` in `destination` (the Library's thumbnail grid). Keyed by
    the file's path, size and modification time, so an edited file gets a fresh one and an unchanged
    one is never decoded again. A RAW file uses its embedded preview (rawpy.extract_thumb, fast)
    and falls back to a half-size demosaic."""
    source = Path(path)
    target_root = Path(destination)
    target_root.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    digest = hashlib.sha1(f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{size}".encode("utf-8")).hexdigest()[:16]
    target = target_root / f"{digest}.jpg"
    if target.exists():
        return target
    preview = _raw_preview(source) if is_raw_path(source) else None
    if preview is None:
        with Image.open(source) as image:
            from PIL import ImageOps

            preview = ImageOps.exif_transpose(image).convert("RGB")
    preview.thumbnail(size, Image.Resampling.LANCZOS)
    preview.save(target, format="JPEG", quality=84, optimize=True)
    return target


def _raw_preview(source: Path) -> Image.Image | None:
    try:
        import io

        import rawpy

        with rawpy.imread(str(source)) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    return Image.open(io.BytesIO(thumb.data)).convert("RGB")
                return Image.fromarray(thumb.data).convert("RGB")
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                return Image.fromarray(raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8))
    except ImportError:
        return None



def index_folder(folder: str | Path, database: str | Path, recursive: bool = True, progress=None,
                 import_xmp: bool = True) -> list[dict[str, object]]:
    """`progress`, when given, is called as `progress(index, total, info)` after each file --
    same shape as app.batch_release.process_files()'s own `progress` callback -- so the frontend's
    "Scan Folder" button can drive an st.progress() bar the same way Run Batch already does.

    With `import_xmp`, a photo catalogued for the first time takes its rating, reject flag and
    keywords from its XMP sidecar (app.xmp_sidecar), if it has one; photos already in the catalog
    keep what was set here."""
    catalog = PhotoCatalog(database)
    paths = scan_folder(folder, recursive)
    total = len(paths)
    records = []
    for index, path in enumerate(paths, 1):
        info = image_info(path)
        if "error" not in info:
            if catalog.ensure(path) and import_xmp:
                sidecar = read_sidecar(path)
                if sidecar is not None:
                    tags = sidecar["tags"] + ([REJECTED_TAG] if sidecar["rejected"] else [])
                    catalog.upsert(path, rating=sidecar["rating"], tags=tags)
            records.append(info)
        if progress:
            progress(index, total, info)
    return records
