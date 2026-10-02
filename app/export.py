"""Metadata-aware export encoding -- the one writer every export path uses.

The main Save button, the backend's export-time encode (AI Upscale / full-resolution renders), the
CLI, and folder batch all go through encode_export(), so they agree on what an exported file
carries. Before this, the Save button wrote neither EXIF nor ICC, the CLI's JPEG branch dropped
EXIF, and folder batch saved with PIL defaults (JPEG quality 75, no metadata).

Metadata policy (settings["metadata"]):
- "all": every EXIF tag the source had (camera, lens, capture time, copyright, GPS...).
- "camera": copyright and camera/capture fields only (see _CAMERA_IFD0_TAGS/_CAMERA_EXIF_TAGS).
- "none": no EXIF at all.
plus settings["remove_gps"], which drops the GPS block under "all" too. In every case the written
EXIF is fixed up for the new file: Orientation is 1 (pixels are already upright -- decode applied
it), PixelXDimension/PixelYDimension are the exported size, and any embedded thumbnail (IFD1, which
would show the *unedited* photo) is dropped. The ICC profile is always kept -- it's color
information, not personal data, and dropping it changes how a viewer renders the colors.
"""
from __future__ import annotations

import io
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageFilter

EXPORT_FORMATS = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "WEBP": (".webp", "image/webp"),
    "TIFF": (".tif", "image/tiff"),
}
METADATA_POLICIES = ("all", "camera", "none")
METADATA_POLICY_LABELS = {"all": "Keep all", "camera": "Copyright & camera only", "none": "Strip all"}
# Output sharpening applied after the resize, in the exported file's own pixels -- the usual
# "sharpen for screen / matte paper / glossy paper" presets of a photo editor's export dialog.
OUTPUT_SHARPENING = {
    "None": None,
    "Screen": (0.6, 60, 2),
    "Matte paper": (1.0, 90, 2),
    "Glossy paper": (1.0, 120, 2),
}

DEFAULT_EXPORT_SETTINGS = {
    "format": "JPEG",
    "quality": 92,
    "long_edge": 0,
    "sharpen": "None",
    "metadata": "all",
    "remove_gps": False,
    "watermark": None,
}

_ORIENTATION = 0x0112
_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
_PIXEL_X_DIMENSION = 0xA002
_PIXEL_Y_DIMENSION = 0xA003
# Make, Model, Software, DateTime, Artist, Copyright
_CAMERA_IFD0_TAGS = {0x010F, 0x0110, 0x0131, 0x0132, 0x013B, 0x8298}
# ExposureTime, FNumber, ExposureProgram, ISO, DateTimeOriginal, DateTimeDigitized, OffsetTime*,
# ShutterSpeed, Aperture, ExposureBias, MaxAperture, MeteringMode, Flash, FocalLength,
# FocalLengthIn35mm, WhiteBalance, BodySerialNumber, LensSpecification, LensMake, LensModel
_CAMERA_EXIF_TAGS = {
    0x829A, 0x829D, 0x8822, 0x8827, 0x9003, 0x9004, 0x9010, 0x9011, 0x9012, 0x9201, 0x9202,
    0x9204, 0x9205, 0x9207, 0x9209, 0x920A, 0xA405, 0xA403, 0xA431, 0xA432, 0xA433, 0xA434,
}


def validate_export_settings(settings) -> dict:
    """Normalize an export settings dict (the frontend's Export panel, or /process's "export"
    option), filling defaults for missing keys. Raises ValueError for an unknown key or value."""
    if not isinstance(settings, dict):
        raise ValueError("export settings must be an object")
    unknown = sorted(set(settings) - set(DEFAULT_EXPORT_SETTINGS))
    if unknown:
        raise ValueError(f"Unknown export setting(s): {', '.join(unknown)}")
    clean = {**DEFAULT_EXPORT_SETTINGS, **settings}
    if clean["format"] not in EXPORT_FORMATS:
        raise ValueError(f"export format must be one of {', '.join(EXPORT_FORMATS)}")
    if clean["metadata"] not in METADATA_POLICIES:
        raise ValueError(f"export metadata must be one of {', '.join(METADATA_POLICIES)}")
    if clean["sharpen"] not in OUTPUT_SHARPENING:
        raise ValueError(f"export sharpen must be one of {', '.join(OUTPUT_SHARPENING)}")
    for key, low, high in (("quality", 1, 100), ("long_edge", 0, 100_000)):
        value = clean[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"export {key} must be between {low} and {high}")
        clean[key] = int(value)
    if not isinstance(clean["remove_gps"], bool):
        raise ValueError("export remove_gps must be true or false")
    watermark = clean["watermark"]
    if watermark is not None:
        if not isinstance(watermark, dict) or not str(watermark.get("text", "")).strip():
            clean["watermark"] = None
        else:
            clean["watermark"] = {
                "text": str(watermark["text"]),
                "position": str(watermark.get("position", "Bottom Right")),
                "opacity": int(watermark.get("opacity", 50)),
                "font_size": int(watermark.get("font_size", 32)),
            }
    return clean


def prepare_exif(exif_bytes: bytes | None, size: tuple[int, int], policy: str = "all", remove_gps: bool = False) -> bytes | None:
    """EXIF bytes to embed in an exported file of `size`, per `policy` (see the module docstring),
    or None when nothing should be written."""
    if not exif_bytes or policy == "none":
        return None
    source = Image.Exif()
    try:
        source.load(exif_bytes)
    except (SyntaxError, ValueError, OSError, TypeError):
        return None  # unreadable EXIF block -- export without it rather than fail the export
    exif_ifd = dict(source.get_ifd(_EXIF_IFD))
    gps_ifd = dict(source.get_ifd(_GPS_IFD))

    result = Image.Exif()
    for tag, value in source.items():
        if tag in (_EXIF_IFD, _GPS_IFD):
            continue
        if policy == "camera" and tag not in _CAMERA_IFD0_TAGS:
            continue
        result[tag] = value
    if policy == "camera":
        exif_ifd = {tag: value for tag, value in exif_ifd.items() if tag in _CAMERA_EXIF_TAGS}
    result[_ORIENTATION] = 1
    exif_ifd[_PIXEL_X_DIMENSION] = int(size[0])
    exif_ifd[_PIXEL_Y_DIMENSION] = int(size[1])
    target_exif_ifd = result.get_ifd(_EXIF_IFD)
    target_exif_ifd.update(exif_ifd)
    if gps_ifd and policy == "all" and not remove_gps:
        result.get_ifd(_GPS_IFD).update(gps_ifd)
    return result.tobytes()


def _resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    if not long_edge or max(image.size) <= long_edge:
        return image
    scale = long_edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def encode_export(image: Image.Image, settings: dict | None = None, *, exif: bytes | None = None, icc: bytes | None = None) -> tuple[bytes, str, str]:
    """Encode `image` for delivery: resize (long edge), output sharpening, watermark, then the
    chosen format with the metadata policy applied. `exif`/`icc` default to `image.info`'s own.
    Returns (file bytes, extension, mime type)."""
    settings = validate_export_settings(settings or {})
    exif = exif if exif is not None else image.info.get("exif")
    icc = icc if icc is not None else image.info.get("icc_profile")

    result = _resize_long_edge(image, settings["long_edge"])
    sharpening = OUTPUT_SHARPENING[settings["sharpen"]]
    if sharpening:
        radius, percent, threshold = sharpening
        result = result.convert("RGB").filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    watermark = settings["watermark"]
    if watermark:
        from app.professional_tools import apply_watermark

        result = apply_watermark(result, watermark["text"], watermark["position"], watermark["opacity"], watermark["font_size"])

    format_name = settings["format"]
    save_kwargs: dict = {"format": format_name}
    if format_name in ("JPEG", "WEBP"):
        result = result.convert("RGB")
        save_kwargs["quality"] = settings["quality"]
        if format_name == "JPEG":
            save_kwargs["optimize"] = True
    elif result.mode not in ("RGB", "RGBA", "L"):
        result = result.convert("RGB")
    exif_out = prepare_exif(exif, result.size, settings["metadata"], settings["remove_gps"])
    if exif_out:
        save_kwargs["exif"] = exif_out
    if icc:
        save_kwargs["icc_profile"] = icc
    buffer = io.BytesIO()
    result.save(buffer, **save_kwargs)
    extension, mime = EXPORT_FORMATS[format_name]
    return buffer.getvalue(), extension, mime


_EXPORT_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
_EXPORT_CACHE_LOCK = threading.Lock()
_EXPORT_CACHE_ENTRIES = 2


def cached_export(key: str, produce: Callable[[], bytes]) -> bytes:
    """Return the export file for `key` (a digest of the photo, the edit and the export settings),
    calling `produce` only on a miss. The frontend's Save button hands Streamlit a data *callable*
    (run only when clicked, on a separate thread), and this makes a repeat click of an unchanged
    export free. A plain lock-guarded LRU rather than st.cache_data, which needs the script thread.
    Two entries: exports are whole files, possibly tens of MB each."""
    with _EXPORT_CACHE_LOCK:
        if key in _EXPORT_CACHE:
            _EXPORT_CACHE.move_to_end(key)
            return _EXPORT_CACHE[key]
    content = produce()
    with _EXPORT_CACHE_LOCK:
        _EXPORT_CACHE[key] = content
        _EXPORT_CACHE.move_to_end(key)
        while len(_EXPORT_CACHE) > _EXPORT_CACHE_ENTRIES:
            _EXPORT_CACHE.popitem(last=False)
    return content


DEFAULT_NAME_TEMPLATE = "{stem}-edited"
_UNSAFE_NAME_CHARACTERS = '<>:"/\\|?*'


def render_export_name(template: str, stem: str, size: tuple[int, int], format_name: str, extension: str, today=None) -> str:
    """The Save button's file name from a template: {stem} (the original's name), {date}
    (YYYY-MM-DD), {width}, {height}, {format}. Path separators and characters Windows forbids are
    replaced, an unknown field falls back to the default template, and the extension is the
    format's own."""
    import datetime

    fields = {
        "stem": stem, "date": (today or datetime.date.today()).isoformat(), "width": size[0], "height": size[1],
        "format": format_name.lower(),
    }
    try:
        name = (template or DEFAULT_NAME_TEMPLATE).format(**fields)
    except (KeyError, IndexError, ValueError):
        name = DEFAULT_NAME_TEMPLATE.format(**fields)
    name = "".join("_" if character in _UNSAFE_NAME_CHARACTERS or ord(character) < 32 else character for character in name).strip(" .")
    return f"{name or stem}{extension}"


class ExportPresetStore:
    """Named export settings (format, quality, resize, sharpening, metadata, GPS, file name) saved
    under the user data folder -- F6's "saveable export presets". Written atomically; an unreadable
    file reads as empty rather than breaking the Export panel."""

    FIELDS = ("format", "quality", "long_edge", "sharpen", "metadata", "remove_gps", "name_template")

    def __init__(self, path: str | Path | None = None):
        if path is None:
            from app.config import user_data_file

            path = user_data_file("export_presets.json")
        self.path = Path(path)

    def load(self) -> dict[str, dict]:
        import json

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(name): {key: value for key, value in preset.items() if key in self.FIELDS} for name, preset in payload.items() if isinstance(preset, dict)} if isinstance(payload, dict) else {}

    def save(self, name: str, settings: dict) -> None:
        import json

        from app.config import atomic_write_text

        presets = self.load()
        presets[name] = {key: settings[key] for key in self.FIELDS if key in settings}
        atomic_write_text(self.path, json.dumps(presets, indent=2, sort_keys=True))

    def delete(self, name: str) -> None:
        import json

        from app.config import atomic_write_text

        presets = self.load()
        presets.pop(name, None)
        atomic_write_text(self.path, json.dumps(presets, indent=2, sort_keys=True))


def format_for_path(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    for name, (extension, _mime) in EXPORT_FORMATS.items():
        if suffix == extension or (name == "JPEG" and suffix == ".jpeg") or (name == "TIFF" and suffix == ".tiff"):
            return name
    raise ValueError(f"Unsupported output format: {suffix or '(no extension)'} -- use .jpg, .png, .webp or .tif")


def write_export(image: Image.Image, output_path: str | Path, settings: dict | None = None) -> Path:
    """encode_export() straight to a file whose format follows its extension (CLI and batch)."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content, _extension, _mime = encode_export(image, {**(settings or {}), "format": format_for_path(output)})
    output.write_bytes(content)
    return output


def save_for_delivery(
    processor,
    output_path: str | Path,
    quality: int = 92,
    max_size: tuple[int, int] | None = None,
    strip_metadata: bool = False,
) -> Path:
    """Kept for existing callers: now a thin wrapper over write_export(), so a JPEG keeps its EXIF
    too (the old JPEG branch dropped it even with strip_metadata=False)."""
    image = processor.image.copy()
    if max_size:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    settings = {"quality": min(100, max(1, int(quality))), "metadata": "none" if strip_metadata else "all"}
    return write_export(image, output_path, settings)
