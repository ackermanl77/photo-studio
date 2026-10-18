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

Bit depth (settings["bit_depth"], PNG and TIFF only): 16 writes a genuine 16-bit file from the
float pixels app.pipeline attaches to its result in high-precision mode (app.high_bit -- B4); an
image without them is widened from its 8 bits. A 16-bit PNG carries ICC and EXIF (iCCP/eXIf
chunks); a 16-bit TIFF carries ICC and the IFD0 text tags (camera make/model, date, artist,
copyright, software) but not the EXIF sub-IFD (exposure details, GPS) -- tifffile can't write it.

Color space (settings["color_space"]): "sRGB" (the working space; the file keeps the image's own
profile) or "Display P3", which converts the finished pixels into P3 and embeds a Display P3 profile
(app.color_profiles) -- last, after resize, sharpening and the watermark, at 8 or 16 bits alike.
"""
from __future__ import annotations

import io
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from app.color_profiles import (
    COLOR_SPACES, convert_srgb_to_display_p3, convert_srgb_to_display_p3_uint8, is_display_p3_icc, output_icc,
)

_logger = logging.getLogger(__name__)

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
    "bit_depth": 8,
    "gpu_export": False,
    "color_space": "sRGB",
}
# Formats that can hold 16 bits per channel.
HIGH_BIT_FORMATS = ("PNG", "TIFF")

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
    if not isinstance(clean["gpu_export"], bool):
        raise ValueError("export gpu_export must be true or false")
    if clean["bit_depth"] not in (8, 16) or isinstance(clean["bit_depth"], bool):
        raise ValueError("export bit_depth must be 8 or 16")
    if clean["color_space"] not in COLOR_SPACES:
        raise ValueError(f"export color_space must be one of {', '.join(COLOR_SPACES)}")
    if clean["bit_depth"] == 16 and clean["format"] not in HIGH_BIT_FORMATS:
        raise ValueError(f"16-bit export needs {' or '.join(HIGH_BIT_FORMATS)}, not {clean['format']}")
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
                "font": str(watermark.get("font") or ""),
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


def _resized_size(size: tuple[int, int], long_edge: int) -> tuple[int, int]:
    width, height = size
    if not long_edge or max(width, height) <= long_edge:
        return size
    scale = long_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def gpu_export_transform(
    pixels: np.ndarray,
    output_size: tuple[int, int],
    sharpening: tuple[float, int, int] | None = None,
) -> np.ndarray | None:
    """Try the optional CUDA export path; return None and log when it must fall back to CPU."""
    try:
        from app.gpu_export import resize_and_sharpen

        result = resize_and_sharpen(pixels, output_size, sharpening)
        if result is None:
            _logger.warning("CUDA export requested but unavailable; using the CPU export path.")
        return result
    except Exception:
        _logger.warning("CUDA export failed; using the CPU export path.", exc_info=True)
        return None


def encode_export(image: Image.Image, settings: dict | None = None, *, exif: bytes | None = None, icc: bytes | None = None) -> tuple[bytes, str, str]:
    """Encode `image` for delivery: resize (long edge), output sharpening, watermark, then the
    chosen format with the metadata policy applied. `exif`/`icc` default to `image.info`'s own.
    Returns (file bytes, extension, mime type)."""
    settings = validate_export_settings(settings or {})
    exif = exif if exif is not None else image.info.get("exif")
    icc = icc if icc is not None else image.info.get("icc_profile")
    if settings["bit_depth"] == 16:
        return _encode_16bit(image, settings, exif, icc)

    sharpening = OUTPUT_SHARPENING[settings["sharpen"]]
    output_size = _resized_size(image.size, settings["long_edge"])
    result = None
    if settings["gpu_export"] and (output_size != image.size or sharpening):
        gpu_input = image.convert("RGB") if sharpening else image
        if gpu_input.mode not in ("RGB", "RGBA", "L"):
            gpu_input = gpu_input.convert("RGB")
        pixels = gpu_export_transform(np.asarray(gpu_input), output_size, sharpening)
        if pixels is not None:
            result = Image.fromarray(pixels)
    if result is None:
        result = _resize_long_edge(image, settings["long_edge"])
        if sharpening:
            radius, percent, threshold = sharpening
            result = result.convert("RGB").filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    watermark = settings["watermark"]
    if watermark:
        from app.professional_tools import apply_watermark

        result = apply_watermark(result, watermark["text"], watermark["position"], watermark["opacity"], watermark["font_size"],
                                 font=watermark.get("font", ""))

    format_name = settings["format"]
    save_kwargs: dict = {"format": format_name}
    if format_name in ("JPEG", "WEBP"):
        result = result.convert("RGB")
        save_kwargs["quality"] = settings["quality"]
        if format_name == "JPEG":
            save_kwargs["optimize"] = True
    elif result.mode not in ("RGB", "RGBA", "L"):
        result = result.convert("RGB")
    if settings["color_space"] == "Display P3":
        # Rendered in the Display P3 working space already (B14, tagged with our P3 profile): keep
        # the pixels -- they may hold colours outside sRGB. Otherwise convert the sRGB result.
        if not is_display_p3_icc(icc):
            result = _to_display_p3(result)
        icc = output_icc("Display P3", icc)
    exif_out = prepare_exif(exif, result.size, settings["metadata"], settings["remove_gps"])
    if exif_out:
        save_kwargs["exif"] = exif_out
    if icc:
        save_kwargs["icc_profile"] = icc
    buffer = io.BytesIO()
    result.save(buffer, **save_kwargs)
    extension, mime = EXPORT_FORMATS[format_name]
    return buffer.getvalue(), extension, mime


def _to_display_p3(image: Image.Image) -> Image.Image:
    """`image` (sRGB) converted to Display P3 pixels, alpha kept; grayscale stays gray (neutral
    colors are the same in both spaces)."""
    if image.mode == "L":
        return image
    rgba = image.mode == "RGBA"
    rgb = image.convert("RGB")
    converted = Image.fromarray(convert_srgb_to_display_p3_uint8(np.asarray(rgb)), mode="RGB")
    if rgba:
        converted.putalpha(image.getchannel("A"))
    converted.info = dict(image.info)
    return converted


# --- 16-bit output (B4) -------------------------------------------------------------------------------

def _resize_float(array: np.ndarray, long_edge: int) -> np.ndarray:
    """_resize_long_edge() on float pixels: the same Pillow Lanczos filter, run per channel on 32-bit
    float ("F") images, so the 16-bit file is resampled exactly like the 8-bit one."""
    height, width = array.shape[:2]
    if not long_edge or max(width, height) <= long_edge:
        return array
    scale = long_edge / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    channels = [np.asarray(Image.fromarray(np.ascontiguousarray(array[..., index], dtype=np.float32), mode="F").resize(size, Image.Resampling.LANCZOS)) for index in range(3)]
    return np.clip(np.stack(channels, axis=-1), 0.0, 1.0)


def _unsharp_float(array: np.ndarray, radius: float, percent: int, threshold: int) -> np.ndarray:
    """PIL's UnsharpMask (radius, percent, threshold in 8-bit levels) on float pixels."""
    from app.high_bit import unsharp_float

    return unsharp_float(array, radius, percent, threshold)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    import struct
    import zlib

    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _png16(array16: np.ndarray, exif: bytes | None, icc: bytes | None) -> bytes:
    """A 16-bit RGB PNG (OpenCV writes the pixels), with iCCP/eXIf chunks spliced in after IHDR --
    OpenCV writes no metadata, and Pillow can't write 16-bit RGB."""
    import cv2
    import zlib

    ok, encoded = cv2.imencode(".png", np.ascontiguousarray(array16[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError("Could not encode the 16-bit PNG")
    content = encoded.tobytes()
    chunks = b""
    if icc:
        chunks += _png_chunk(b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(icc))
    if exif:
        chunks += _png_chunk(b"eXIf", exif[6:] if exif.startswith(b"Exif\x00\x00") else exif)
    ihdr_end = 8 + 4 + 4 + 13 + 4  # signature, then the IHDR chunk (length, type, 13 bytes, CRC)
    return content[:ihdr_end] + chunks + content[ihdr_end:]


# IFD0 text tags a 16-bit TIFF keeps from the prepared EXIF: Make, Model, Software, DateTime, Artist, Copyright.
_TIFF_TEXT_TAGS = (0x010F, 0x0110, 0x0131, 0x0132, 0x013B, 0x8298)


def _tiff16(array16: np.ndarray, exif: bytes | None, icc: bytes | None) -> bytes:
    """A 16-bit RGB TIFF (tifffile; Pillow can't write one) with the ICC profile, the IFD0 text tags
    and the EXIF and GPS sub-IFDs of the prepared EXIF. tifffile only writes top-level tags, so the
    sub-IFD pointers are written as placeholders and the sub-IFDs -- serialized by Pillow's own
    TIFF tag writer -- are appended and linked afterwards (_link_sub_ifds). tifffile refuses the
    pointer tags themselves, so the placeholders use the next tag number (unassigned in IFD0, and
    nothing sorts between the two), renumbered when linking."""
    import tifffile

    extratags, software, sub_ifds = [], None, {}
    if exif:
        parsed = Image.Exif()
        parsed.load(exif)
        for tag in _TIFF_TEXT_TAGS:
            value = parsed.get(tag)
            if not isinstance(value, str) or not value:
                continue
            if tag == 0x0131:
                software = value  # tifffile owns the Software tag; it takes it as a parameter
            else:
                extratags.append((tag, "s", 0, value, True))
        for pointer in _SUB_IFD_POINTERS:
            entries = dict(parsed.get_ifd(pointer))
            if entries:
                sub_ifds[pointer] = entries
                extratags.append((pointer + 1, "I", 1, 0, True))  # placeholder, linked below
    buffer = io.BytesIO()
    tifffile.imwrite(
        buffer, array16, photometric="rgb", compression="zlib", iccprofile=icc or None, extratags=extratags,
        metadata=None, software=software,
    )
    content = buffer.getvalue()
    return _link_sub_ifds(content, sub_ifds) if sub_ifds else content


# The EXIF (0x8769) and GPS (0x8825) sub-IFD pointer tags of IFD0.
_SUB_IFD_POINTERS = (0x8769, 0x8825)


def _link_sub_ifds(content: bytes, sub_ifds: dict[int, dict]) -> bytes:
    """Append each sub-IFD to a classic little-endian TIFF, then turn IFD0's placeholder entry (tag
    pointer + 1) into the real pointer tag holding its offset."""
    import struct

    from PIL import TiffImagePlugin

    data = bytearray(content)
    if data[:4] != b"II*\x00":
        return content  # not the classic little-endian layout tifffile writes by default
    ifd0 = struct.unpack_from("<I", data, 4)[0]
    count = struct.unpack_from("<H", data, ifd0)[0]
    entry_offsets = {struct.unpack_from("<H", data, ifd0 + 2 + 12 * i)[0]: ifd0 + 2 + 12 * i for i in range(count)}
    for pointer, entries in sub_ifds.items():
        entry = entry_offsets.get(pointer + 1)
        if entry is None:
            continue
        if len(data) % 2:
            data += b"\x00"  # IFDs start on a word boundary
        directory = TiffImagePlugin.ImageFileDirectory_v2(group=pointer)
        for tag, value in entries.items():
            directory[tag] = value
        offset = len(data)
        data += directory.tobytes(offset)
        struct.pack_into("<HHII", data, entry, pointer, 4, 1, offset)  # tag, LONG, count 1, offset
    return bytes(data)


def _encode_16bit(image: Image.Image, settings: dict, exif: bytes | None, icc: bytes | None) -> tuple[bytes, str, str]:
    from app.high_bit import attached, from_image, quantize8

    array = attached(image)
    array = from_image(image) if array is None else np.asarray(array, dtype=np.float32)
    sharpening = OUTPUT_SHARPENING[settings["sharpen"]]
    output_size = _resized_size((array.shape[1], array.shape[0]), settings["long_edge"])
    gpu_array = None
    if settings["gpu_export"] and (output_size != (array.shape[1], array.shape[0]) or sharpening):
        gpu_array = gpu_export_transform(array, output_size, sharpening)
    if gpu_array is not None:
        array = gpu_array
    else:
        array = _resize_float(array, settings["long_edge"])
        if sharpening:
            array = _unsharp_float(array, *sharpening)
    watermark = settings["watermark"]
    if watermark:
        from app.professional_tools import apply_watermark

        base = Image.fromarray(quantize8(array), mode="RGB")
        marked = apply_watermark(base, watermark["text"], watermark["position"], watermark["opacity"], watermark["font_size"],
                                 font=watermark.get("font", "")).convert("RGB")
        array = np.clip(array + (np.asarray(marked, dtype=np.float32) - np.asarray(base, dtype=np.float32)) / 255.0, 0.0, 1.0)
    from app.high_bit import to_uint16

    if settings["color_space"] == "Display P3":
        if not is_display_p3_icc(icc):
            array = convert_srgb_to_display_p3(array)
        icc = output_icc("Display P3", icc)
    array16 = to_uint16(array)
    size = (array16.shape[1], array16.shape[0])
    exif_out = prepare_exif(exif, size, settings["metadata"], settings["remove_gps"])
    format_name = settings["format"]
    content = _png16(array16, exif_out, icc) if format_name == "PNG" else _tiff16(array16, exif_out, icc)
    extension, mime = EXPORT_FORMATS[format_name]
    return content, extension, mime


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

    FIELDS = ("format", "quality", "long_edge", "sharpen", "metadata", "remove_gps", "name_template", "bit_depth", "gpu_export",
              "color_space")

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
    """encode_export() straight to a file whose format follows its extension (CLI and batch). A
    16-bit request applies where the format can hold it (PNG/TIFF); a JPEG/WEBP output stays 8-bit."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = {**(settings or {}), "format": format_for_path(output)}
    if settings["format"] not in HIGH_BIT_FORMATS:
        settings.pop("bit_depth", None)
    content, _extension, _mime = encode_export(image, settings)
    output.write_bytes(content)
    return output
