"""Layered Photoshop documents (F22): read a PSD's raster layers, write a layered PSD.

Standard library + NumPy + Pillow only. psd-tools was the obvious choice, but its compositing
extra pulls in scikit-image/scipy/aggdraw for what amounts to a few hundred lines of a documented
format, and Pillow's own PSD plugin reads layers but can't write them. Coverage:

Reading (``read_psd``):
- PSD (version 1) and PSB (version 2, the Large Document Format: 8-byte lengths, 4-byte RLE counts);
- RGB, Grayscale, CMYK, Lab and Duotone, converted to RGB (CMYK and Lab by the textbook formulas,
  not an ICC profile; Duotone as its grey plane), 8 or 16 bits per channel (16 is reduced to 8 -- the
  Layers stack is 8-bit). A 16-bit document's layers are in the "Lr16" block after the layer info;
- channel data raw, RLE (PackBits), ZIP and ZIP-with-prediction;
- per layer: name (the Unicode ``luni`` name when present), position, opacity, blend mode,
  visibility, transparency, fill opacity, its user mask and a clipping mask (both applied to the
  alpha). Layer effects (drop shadow, glow, stroke...) and vector masks aren't rendered;
- groups: their visibility is carried down to the layers inside them and each layer records its
  group path; the group markers themselves (section dividers) aren't layers;
- adjustment, fill-without-pixels and other empty layers are skipped and counted (adjustments by
  name), since they have no pixels to import. Text, shape and smart-object layers import as the pixels
  Photoshop stored with them; a text layer also carries its text, font, size and colour (``text``);
- the composite ("maximize compatibility" image) is returned too.

Writing (``write_psd``): an 8-bit RGB PSD with one raster layer per entry (bottom first, the order
Photoshop's file stores them), each cropped to its visible pixels, with its name (Pascal and
Unicode), opacity, blend mode and visibility, plus the composite image.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

SIGNATURE = b"8BPS"
# Photo Studio's blend modes (app.major_features.BLEND_MODE_LABELS) <-> Photoshop's blend keys.
BLEND_KEYS = {
    "normal": b"norm", "darken": b"dark", "multiply": b"mul ", "color_burn": b"idiv", "linear_burn": b"lbrn",
    "darker_color": b"dkCl", "lighten": b"lite", "screen": b"scrn", "color_dodge": b"div ", "linear_dodge": b"lddg",
    "lighter_color": b"lgCl", "overlay": b"over", "soft_light": b"sLit", "hard_light": b"hLit", "vivid_light": b"vLit",
    "linear_light": b"lLit", "pin_light": b"pLit", "hard_mix": b"hMix", "difference": b"diff", "exclusion": b"smud",
    "subtract": b"fsub", "divide": b"fdiv", "hue": b"hue ", "saturation": b"sat ", "color": b"colr", "luminosity": b"lum ",
}
BLEND_MODES = {key: mode for mode, key in BLEND_KEYS.items()}
MAX_DIMENSION = 30000  # PSD's own limit
MAX_PSB_DIMENSION = 300000  # PSB (Large Document Format)
# Colour modes: the channel ids that hold colour, per mode. Duotone stores one grey plane.
MODE_CHANNELS = {1: (0,), 3: (0, 1, 2), 4: (0, 1, 2, 3), 8: (0,), 9: (0, 1, 2)}
MODE_NAMES = {0: "Bitmap", 2: "Indexed", 7: "Multichannel"}
# In a PSB these tagged blocks have an 8-byte length instead of 4 (Adobe's file format spec).
PSB_LONG_KEYS = {b"LMsk", b"Lr16", b"Lr32", b"Layr", b"Mt16", b"Mt32", b"Mtrn", b"Alph", b"FMsk", b"lnk2", b"FEid",
                 b"FXid", b"PxSD"}
# Adjustment and fill layers: their tagged-block keys, for naming what an import skips.
ADJUSTMENT_KEYS = {
    b"brit": "Brightness/Contrast", b"levl": "Levels", b"curv": "Curves", b"expA": "Exposure", b"vibA": "Vibrance",
    b"hue ": "Hue/Saturation", b"hue2": "Hue/Saturation", b"blnc": "Color Balance", b"blwh": "Black & White",
    b"phfl": "Photo Filter", b"mixr": "Channel Mixer", b"clrL": "Color Lookup", b"nvrt": "Invert",
    b"post": "Posterize", b"thrs": "Threshold", b"grdm": "Gradient Map", b"selc": "Selective Color",
    b"SoCo": "Solid Color", b"GdFl": "Gradient Fill", b"PtFl": "Pattern Fill",
}


class PsdError(ValueError):
    """The file isn't a PSD this reader handles."""


@dataclass
class PsdLayer:
    name: str
    image: Image.Image  # RGBA, the layer's own pixels (not the whole canvas)
    left: int = 0
    top: int = 0
    opacity: float = 1.0
    blend_mode: str = "normal"
    visible: bool = True
    group: str = ""  # "Outer / Inner" for a layer inside groups, "" at the top level
    # A text layer's own text, for importing it as editable text: {"contents", "font", "size_px",
    # "color" (r, g, b)}. The pixels above are what Photoshop rendered for it.
    text: dict | None = None

    def on_canvas(self, size: tuple[int, int]) -> Image.Image:
        """The layer placed on a transparent canvas of the document's size."""
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        canvas.paste(self.image, (self.left, self.top))
        return canvas


@dataclass
class PsdDocument:
    width: int
    height: int
    layers: list[PsdLayer] = field(default_factory=list)  # bottom first
    composite: Image.Image | None = None
    skipped: int = 0  # layers without pixels (adjustments, empty layers)
    skipped_adjustments: list[str] = field(default_factory=list)  # e.g. ["Hue/Saturation"], one per skipped layer
    large: bool = False  # a PSB (Large Document Format) file


# --- reading -------------------------------------------------------------------------------------

class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.pos + count > len(self.data):
            raise PsdError("The PSD file is truncated.")
        chunk = self.data[self.pos:self.pos + count]
        self.pos += count
        return chunk

    def unpack(self, fmt: str):
        values = struct.unpack(">" + fmt, self.take(struct.calcsize(">" + fmt)))
        return values if len(values) > 1 else values[0]


def _unpackbits(data: bytes, expected: int) -> bytes:
    out = bytearray()
    pos, size = 0, len(data)
    while pos < size and len(out) < expected:
        header = data[pos]
        pos += 1
        if header < 128:
            out += data[pos:pos + header + 1]
            pos += header + 1
        elif header > 128:
            out += data[pos:pos + 1] * (257 - header)
            pos += 1
    if len(out) < expected:
        raise PsdError("A layer's compressed pixel data is truncated.")
    return bytes(out[:expected])


def _decode_channel(compression: int, payload: bytes, width: int, height: int, depth: int, count_size: int = 2) -> np.ndarray:
    """One channel as uint8. `count_size`: the bytes per RLE row count (2 in a PSD, 4 in a PSB)."""
    bytes_per = depth // 8
    row = width * bytes_per
    if compression == 0:
        raw = payload[:row * height]
        if len(raw) < row * height:
            raise PsdError("A layer's pixel data is truncated.")
    elif compression == 1:
        count_format = "H" if count_size == 2 else "I"
        counts = struct.unpack(f">{height}{count_format}", payload[:count_size * height])
        pos, rows = count_size * height, []
        for count in counts:
            rows.append(_unpackbits(payload[pos:pos + count], row))
            pos += count
        raw = b"".join(rows)
    elif compression in (2, 3):
        raw = zlib.decompress(payload)
        if compression == 3:
            dtype = np.uint8 if bytes_per == 1 else np.dtype(">u2")
            deltas = np.frombuffer(raw, dtype=dtype).reshape(height, width)
            raw = np.cumsum(deltas, axis=1, dtype=np.uint64).astype(deltas.dtype).tobytes()
    else:
        raise PsdError(f"Unsupported PSD compression {compression}.")
    if bytes_per == 1:
        return np.frombuffer(raw, dtype=np.uint8, count=width * height).reshape(height, width)
    values = np.frombuffer(raw, dtype=">u2", count=width * height).reshape(height, width)
    return (values.astype(np.uint32) * 255 // 65535).astype(np.uint8)


def _to_rgb(planes: list[np.ndarray], mode: int) -> list[np.ndarray]:
    """A mode's colour planes (MODE_CHANNELS order) as three 8-bit R, G, B planes. CMYK and Lab are
    converted with the plain textbook formulas, not through an ICC profile, so colours are close to
    Photoshop's but not identical."""
    if mode in (1, 8):  # Grayscale; Duotone keeps its grey plane (the inks aren't simulated)
        return [planes[0]] * 3
    if mode == 3:
        return planes
    if mode == 4:
        # PSD stores CMYK inverted: 255 is no ink. RGB = (1 - C)(1 - K) on the ink fractions.
        c, m, y, k = (plane.astype(np.uint32) for plane in planes)
        return [(channel * k // 255).astype(np.uint8) for channel in (c, m, y)]
    if mode == 9:
        import cv2

        lab = np.dstack([
            planes[0].astype(np.float32) * (100.0 / 255.0),
            planes[1].astype(np.float32) - 128.0,
            planes[2].astype(np.float32) - 128.0,
        ])
        rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return [rgb[..., 0], rgb[..., 1], rgb[..., 2]]
    raise PsdError("Unsupported PSD colour mode.")


def _pascal(reader: _Reader, pad: int) -> str:
    length = reader.unpack("B")
    name = reader.take(length).decode("latin-1")
    total = 1 + length
    reader.take((-total) % pad)
    return name


def _length(reader: _Reader, large: bool) -> int:
    return reader.unpack("Q" if large else "I")


def _tagged_blocks(reader: _Reader, end: int, large: bool, align: int):
    """(key, value) for each "8BIM"/"8B64" tagged block up to `end`. `align`: the block padding (2
    inside a layer record, 4 for the document-level blocks after the layer info)."""
    while reader.pos + 12 <= end:
        if reader.data[reader.pos:reader.pos + 4] not in (b"8BIM", b"8B64"):
            return
        reader.take(4)
        key = reader.take(4)
        length = _length(reader, large and key in PSB_LONG_KEYS)
        value = reader.take(length)
        reader.take(min((-length) % align, max(0, end - reader.pos)))
        yield key, value


def read_psd(content: bytes) -> PsdDocument:
    reader = _Reader(content)
    if reader.take(4) != SIGNATURE:
        raise PsdError("Not a Photoshop file.")
    version = reader.unpack("H")
    if version not in (1, 2):
        raise PsdError("Unknown Photoshop file version.")
    large = version == 2
    reader.take(6)
    channels, height, width, depth, mode = reader.unpack("HIIHH")
    if mode not in MODE_CHANNELS:
        raise PsdError(f"{MODE_NAMES.get(mode, 'This colour mode')} PSDs aren't supported; convert to RGB in Photoshop.")
    if depth not in (8, 16):
        raise PsdError("Only 8- and 16-bit PSDs are supported (32-bit files aren't).")
    limit = MAX_PSB_DIMENSION if large else MAX_DIMENSION
    if not (0 < width <= limit and 0 < height <= limit):
        raise PsdError("The PSD's size is invalid.")
    reader.take(reader.unpack("I"))  # colour mode data
    reader.take(reader.unpack("I"))  # image resources
    layer_mask_length = _length(reader, large)
    section_end = reader.pos + layer_mask_length
    document = PsdDocument(width, height, large=large)
    if layer_mask_length:
        info_length = _length(reader, large)
        info_end = reader.pos + info_length
        if info_length:
            _read_layers(reader, document, depth, mode, large)
        reader.pos = info_end
        if reader.pos + 4 <= section_end:
            reader.take(reader.unpack("I"))  # global layer mask info
            # A 16-bit document keeps its layers in an "Lr16" block here (the layer info above is
            # empty), a 32-bit one in "Lr32".
            for key, value in _tagged_blocks(reader, section_end, large, 4):
                if key == b"Lr16" and not document.layers and not document.skipped:
                    _read_layers(_Reader(value), document, depth, mode, large)
    reader.pos = section_end
    document.composite = _read_composite(reader, width, height, channels, depth, mode, large)
    return document


def _read_layers(reader: _Reader, document: PsdDocument, depth: int, mode: int, large: bool = False) -> None:
    count = abs(reader.unpack("h"))
    records = []
    for _ in range(count):
        top, left, bottom, right = reader.unpack("iiii")
        channel_count = reader.unpack("H")
        channels = [(reader.unpack("h"), _length(reader, large)) for _ in range(channel_count)]
        if reader.take(4) != b"8BIM":
            raise PsdError("A layer record is damaged.")
        blend_key = reader.take(4)
        opacity, clipping, flags, _filler = reader.unpack("BBBB")
        extra_length = reader.unpack("I")
        extra_end = reader.pos + extra_length
        mask = None
        mask_length = reader.unpack("I")
        if mask_length >= 18:
            mask_top, mask_left, mask_bottom, mask_right = reader.unpack("iiii")
            default_color, mask_flags = reader.unpack("BB")
            mask = (mask_top, mask_left, mask_bottom, mask_right, default_color, mask_flags)
            reader.take(mask_length - 18)
        else:
            reader.take(mask_length)
        reader.take(reader.unpack("I"))  # blending ranges
        name = _pascal(reader, 4)
        section_type, fill, text, adjustment = 0, 255, None, None
        for key, value in _tagged_blocks(reader, extra_end, large, 2):
            if key == b"luni" and len(value) >= 4:
                (chars,) = struct.unpack(">I", value[:4])
                name = value[4:4 + 2 * chars].decode("utf-16-be", "replace").rstrip("\x00")
            elif key in (b"lsct", b"lsdk") and len(value) >= 4:
                (section_type,) = struct.unpack(">I", value[:4])
            elif key == b"iOpa" and value:
                fill = value[0]
            elif key == b"TySh":
                text = _read_text(value)
            elif key in ADJUSTMENT_KEYS:
                adjustment = ADJUSTMENT_KEYS[key]
        reader.pos = extra_end
        records.append({
            "rect": (top, left, bottom, right), "channels": channels, "blend": blend_key, "opacity": opacity,
            "visible": not flags & 2, "name": name, "section": section_type, "mask": mask,
            "clipping": bool(clipping), "fill": fill, "text": text, "adjustment": adjustment,
        })
    count_size = 4 if large else 2
    # Channel image data, record by record, channel by channel.
    for record in records:
        top, left, bottom, right = record["rect"]
        pixels = {}
        for channel_id, length in record["channels"]:
            payload = reader.take(length)
            if length < 2:
                continue
            if channel_id < -2:
                continue  # -3: the "real" user mask (a vector/density pair's second mask), not imported
            compression = struct.unpack(">H", payload[:2])[0]
            if channel_id == -2 and record["mask"]:
                mask_top, mask_left, mask_bottom, mask_right = record["mask"][:4]
                w, h = mask_right - mask_left, mask_bottom - mask_top
            else:
                w, h = right - left, bottom - top
            if w <= 0 or h <= 0:
                continue
            pixels[channel_id] = _decode_channel(compression, payload[2:], w, h, depth, count_size)
        record["pixels"] = pixels
    _build_layers(records, document, mode)


def _build_layers(records: list[dict], document: PsdDocument, mode: int) -> None:
    # Groups: read top-down, a folder record (section 1/2) opens a group and the bounding divider
    # (section 3) below its children closes it. A hidden group hides everything inside it.
    hidden_depth, stack = 0, []
    visible_flags = [True] * len(records)
    group_paths = [""] * len(records)
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        if record["section"] in (1, 2):
            stack.append((not record["visible"], record["name"] or "Group"))
            hidden_depth += stack[-1][0]
        elif record["section"] == 3:
            if stack:
                hidden_depth -= stack.pop()[0]
        else:
            visible_flags[index] = record["visible"] and hidden_depth == 0
            group_paths[index] = " / ".join(group_name for _hidden, group_name in stack)
    colour_ids = MODE_CHANNELS[mode]
    base = None  # (left, top, alpha) of the last unclipped layer -- what clipped layers above it clip to
    for record, visible, group in zip(records, visible_flags, group_paths):
        if record["section"]:
            continue
        top, left, bottom, right = record["rect"]
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0 or any(channel not in record["pixels"] for channel in colour_ids):
            document.skipped += 1
            if record["adjustment"]:
                document.skipped_adjustments.append(record["adjustment"])
            continue
        colour = _to_rgb([record["pixels"][channel] for channel in colour_ids], mode)
        alpha = record["pixels"].get(-1)
        alpha = np.full((height, width), 255, dtype=np.uint8) if alpha is None else alpha.copy()
        mask = record["mask"]
        if mask and -2 in record["pixels"] and not mask[5] & 2:  # flag bit 1: mask disabled
            mask_top, mask_left, mask_bottom, mask_right, default_color = mask[:5]
            full = np.full((height, width), default_color, dtype=np.uint8)
            y0, x0 = max(mask_top, top), max(mask_left, left)
            y1, x1 = min(mask_bottom, bottom), min(mask_right, right)
            if y1 > y0 and x1 > x0:
                full[y0 - top:y1 - top, x0 - left:x1 - left] = record["pixels"][-2][y0 - mask_top:y1 - mask_top, x0 - mask_left:x1 - mask_left]
            alpha = (alpha.astype(np.uint16) * full // 255).astype(np.uint8)
        if record["fill"] < 255:  # fill opacity: the pixels' own opacity (layer effects aren't imported)
            alpha = (alpha.astype(np.uint16) * record["fill"] // 255).astype(np.uint8)
        if record["clipping"] and base is not None:
            # Clipped to the layer below: visible only where that base layer has pixels.
            base_alpha = np.zeros((height, width), dtype=np.uint8)
            b_left, b_top, b_alpha = base
            y0, x0 = max(top, b_top), max(left, b_left)
            y1, x1 = min(bottom, b_top + b_alpha.shape[0]), min(right, b_left + b_alpha.shape[1])
            if y1 > y0 and x1 > x0:
                base_alpha[y0 - top:y1 - top, x0 - left:x1 - left] = b_alpha[y0 - b_top:y1 - b_top, x0 - b_left:x1 - b_left]
            alpha = (alpha.astype(np.uint16) * base_alpha // 255).astype(np.uint8)
        elif not record["clipping"]:
            base = (left, top, alpha)
        image = Image.fromarray(np.dstack([*colour, alpha]), mode="RGBA")
        text = record["text"]
        if text and (mode != 3 or text["color"] is None):
            # EngineData stores the colour in the document's mode (CMYK, grey, Lab values), not RGB:
            # take it from what Photoshop rendered instead -- the median of the solid glyph pixels.
            solid = alpha >= 200
            if solid.any():
                text = {**text, "color": tuple(int(np.median(plane[solid])) for plane in colour)}
        document.layers.append(PsdLayer(
            record["name"] or "Layer", image, left, top, record["opacity"] / 255.0,
            BLEND_MODES.get(record["blend"], "normal"), visible, group, text,
        ))


def _read_composite(reader: _Reader, width: int, height: int, channels: int, depth: int, mode: int,
                    large: bool = False) -> Image.Image | None:
    if reader.pos + 2 > len(reader.data):
        return None
    compression = reader.unpack("H")
    plane = width * height * depth // 8
    colour = len(MODE_CHANNELS[mode])
    count_size = 4 if large else 2
    count_format = "H" if count_size == 2 else "I"
    try:
        if compression == 1:
            counts = struct.unpack(f">{height * channels}{count_format}", reader.take(count_size * height * channels))
            planes = []
            for index in range(colour):
                rows = counts[index * height:(index + 1) * height]
                start = reader.pos + sum(counts[:index * height])
                payload = reader.data[start:start + sum(rows)]
                planes.append(_decode_channel(1, struct.pack(f">{height}{count_format}", *rows) + payload, width, height,
                                              depth, count_size))
        elif compression == 0:
            planes = [_decode_channel(0, reader.take(plane), width, height, depth) for _ in range(colour)]
        else:
            return None
    except PsdError:
        return None
    return Image.fromarray(np.dstack(_to_rgb(planes, mode)), mode="RGB")


# --- text layers ---------------------------------------------------------------------------------

def _read_text(value: bytes) -> dict | None:
    """A "TySh" block: the layer's text, font, size and colour, or None if it can't be read. The
    text sits in the descriptor's "Txt " item; font, size and colour of its first run in the
    EngineData (a PostScript-like dictionary) of the same descriptor."""
    try:
        reader = _Reader(value)
        reader.unpack("H")  # version
        xx, _xy, _yx, yy, _tx, _ty = reader.unpack("6d")
        reader.unpack("H")  # text version
        reader.unpack("I")  # descriptor version
        items = _read_descriptor(reader)
    except (PsdError, struct.error, UnicodeDecodeError, ValueError):
        return None
    contents = items.get(b"Txt ")
    if not isinstance(contents, str):
        return None
    engine = items.get(b"EngineData", b"")
    engine = engine if isinstance(engine, bytes) else b""
    size = _engine_number(engine, b"/FontSize")
    scale = (abs(xx) + abs(yy)) / 2 or 1.0  # Photoshop keeps a resized text as size x transform
    return {
        "contents": contents.replace("\r", "\n").rstrip("\x00"),
        "font": _engine_font(engine),
        "size_px": round(size * scale, 2) if size else None,
        "color": _engine_color(engine),
    }


def _unicode(reader: _Reader) -> str:
    length = reader.unpack("I")
    return reader.take(2 * length).decode("utf-16-be").rstrip("\x00")


def _key(reader: _Reader) -> bytes:
    length = reader.unpack("I")
    return reader.take(length or 4)


def _read_descriptor(reader: _Reader, depth: int = 0) -> dict:
    """An Action Descriptor (the structure Photoshop stores text and layer settings in) as
    {key: value}, for the item types text layers use."""
    if depth > 16:
        raise PsdError("A text layer's settings are nested too deeply.")
    _unicode(reader)  # class name
    _key(reader)  # class id
    items = {}
    for _ in range(reader.unpack("I")):
        key = _key(reader)
        items[key] = _read_item(reader, reader.take(4), depth)
    return items


def _read_item(reader: _Reader, kind: bytes, depth: int):
    if kind in (b"Objc", b"GlbO"):
        return _read_descriptor(reader, depth + 1)
    if kind == b"VlLs":
        return [_read_item(reader, reader.take(4), depth) for _ in range(reader.unpack("I"))]
    if kind == b"TEXT":
        return _unicode(reader)
    if kind in (b"tdta", b"alis", b"Pth "):
        return reader.take(reader.unpack("I"))
    if kind == b"doub":
        return reader.unpack("d")
    if kind == b"UntF":
        reader.take(4)
        return reader.unpack("d")
    if kind == b"long":
        return reader.unpack("i")
    if kind == b"comp":
        return reader.unpack("q")
    if kind == b"bool":
        return bool(reader.unpack("B"))
    if kind == b"enum":
        _key(reader)
        return _key(reader)
    if kind in (b"type", b"GlbC"):
        _unicode(reader)
        return _key(reader)
    raise PsdError(f"Unsupported descriptor item {kind!r}.")


def _engine_number(engine: bytes, name: bytes) -> float | None:
    import re

    match = re.search(re.escape(name) + rb"\s+(-?[0-9.]+)", engine)
    return float(match.group(1)) if match else None


def _engine_font(engine: bytes) -> str:
    """The first font of the FontSet that isn't Photoshop's invisible placeholder ("AdobeInvisFont")."""
    import re

    start = engine.find(b"/FontSet")
    for match in re.finditer(rb"/Name\s+\(((?:\\.|[^\\)])*)\)", engine[start:] if start >= 0 else b""):
        raw = re.sub(rb"\\(.)", rb"\1", match.group(1))
        name = raw[2:].decode("utf-16-be", "replace") if raw.startswith(b"\xfe\xff") else raw.decode("latin-1")
        name = name.rstrip("\x00")
        if name and name != "AdobeInvisFont":
            return name
    return ""


def _engine_color(engine: bytes) -> tuple[int, int, int] | None:
    """The first run's fill colour: /FillColor << /Type 1 /Values [ a r g b ] >> with 0-1 floats."""
    import re

    match = re.search(rb"/FillColor\s*<<\s*/Type\s+1\s*/Values\s*\[\s*([-0-9. ]+)\]", engine)
    if not match:
        return None
    values = [float(value) for value in match.group(1).split()]
    if len(values) != 4:
        return None
    return tuple(max(0, min(255, round(value * 255))) for value in values[1:])


# --- writing -------------------------------------------------------------------------------------

def _pascal_bytes(name: str, pad: int) -> bytes:
    encoded = name.encode("latin-1", "replace")[:255]
    data = bytes([len(encoded)]) + encoded
    return data + b"\0" * ((-len(data)) % pad)


def write_psd(composite: Image.Image, layers: list[PsdLayer]) -> bytes:
    """An 8-bit RGB PSD: `layers` (bottom first, each an RGBA image at left/top on the canvas) over
    nothing, with `composite` as the flattened image Photoshop shows without reading layers."""
    composite = composite.convert("RGB")
    width, height = composite.size
    if not (0 < width <= MAX_DIMENSION and 0 < height <= MAX_DIMENSION):
        raise ValueError("A PSD can be at most 30000 pixels on a side.")
    records, channel_data = [], []
    for layer in layers:
        image = layer.image.convert("RGBA")
        left, top = layer.left, layer.top
        bbox = image.getchannel("A").getbbox()
        if bbox is None:
            image, left, top = Image.new("RGBA", (1, 1), (0, 0, 0, 0)), max(0, left), max(0, top)
        else:
            image = image.crop(bbox)
            left, top = left + bbox[0], top + bbox[1]
        w, h = image.size
        array = np.asarray(image)
        channels = []
        for channel_id, index in ((-1, 3), (0, 0), (1, 1), (2, 2)):
            data = struct.pack(">H", 0) + np.ascontiguousarray(array[:, :, index]).tobytes()
            channels.append((channel_id, data))
        name = layer.name or "Layer"
        luni = struct.pack(">I", len(name)) + name.encode("utf-16-be")
        extra = (
            struct.pack(">I", 0)  # no layer mask
            + struct.pack(">I", 0)  # no blending ranges
            + _pascal_bytes(name, 4)
            + b"8BIMluni" + struct.pack(">I", len(luni)) + luni
        )
        flags = 8 | (0 if layer.visible else 2)
        record = (
            struct.pack(">iiiiH", top, left, top + h, left + w, len(channels))
            + b"".join(struct.pack(">hI", channel_id, len(data)) for channel_id, data in channels)
            + b"8BIM" + BLEND_KEYS.get(layer.blend_mode, b"norm")
            + struct.pack(">BBBB", round(max(0.0, min(1.0, layer.opacity)) * 255), 0, flags, 0)
            + struct.pack(">I", len(extra)) + extra
        )
        records.append(record)
        channel_data.extend(data for _id, data in channels)
    layer_info = struct.pack(">h", len(layers)) + b"".join(records) + b"".join(channel_data) if layers else b""
    layer_info += b"\0" * (len(layer_info) % 2)
    layer_section = (struct.pack(">I", len(layer_info)) + layer_info if layers else b"") + struct.pack(">I", 0)
    if not layers:
        layer_section = struct.pack(">I", 0) + struct.pack(">I", 0)
    array = np.asarray(composite)
    image_data = struct.pack(">H", 0) + b"".join(np.ascontiguousarray(array[:, :, index]).tobytes() for index in range(3))
    header = SIGNATURE + struct.pack(">H6xHIIHH", 1, 3, height, width, 8, 3)
    return (
        header
        + struct.pack(">I", 0)  # colour mode data
        + struct.pack(">I", 0)  # image resources
        + struct.pack(">I", len(layer_section)) + layer_section
        + image_data
    )
