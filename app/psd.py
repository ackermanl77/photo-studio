"""Layered Photoshop documents (F22): read a PSD's raster layers, write a layered PSD.

Standard library + NumPy + Pillow only. psd-tools was the obvious choice, but its compositing
extra pulls in scikit-image/scipy/aggdraw for what amounts to a few hundred lines of a documented
format, and Pillow's own PSD plugin reads layers but can't write them. Coverage:

Reading (``read_psd``):
- PSD version 1, RGB or Grayscale, 8 or 16 bits per channel (16 is reduced to 8 -- the Layers stack is 8-bit);
- channel data raw, RLE (PackBits), ZIP and ZIP-with-prediction;
- per layer: name (the Unicode ``luni`` name when present), position, opacity, blend mode,
  visibility, transparency, fill opacity, its user mask and a clipping mask (both applied to the
  alpha). Layer effects (drop shadow, glow, stroke...) and vector masks aren't rendered;
- groups: their visibility is carried down to the layers inside them; the group markers themselves
  (section dividers) aren't layers;
- adjustment, fill-without-pixels and other empty layers are skipped and counted, since they have no
  pixels to import. Text, shape and smart-object layers import as the pixels Photoshop stored with
  them (they arrive rasterized);
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


def _decode_channel(compression: int, payload: bytes, width: int, height: int, depth: int) -> np.ndarray:
    bytes_per = depth // 8
    row = width * bytes_per
    if compression == 0:
        raw = payload[:row * height]
        if len(raw) < row * height:
            raise PsdError("A layer's pixel data is truncated.")
    elif compression == 1:
        counts = struct.unpack(f">{height}H", payload[:2 * height])
        pos, rows = 2 * height, []
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


def _pascal(reader: _Reader, pad: int) -> str:
    length = reader.unpack("B")
    name = reader.take(length).decode("latin-1")
    total = 1 + length
    reader.take((-total) % pad)
    return name


def read_psd(content: bytes) -> PsdDocument:
    reader = _Reader(content)
    if reader.take(4) != SIGNATURE:
        raise PsdError("Not a Photoshop file.")
    version = reader.unpack("H")
    if version != 1:
        raise PsdError("Large-document PSB files aren't supported; save as PSD.")
    reader.take(6)
    channels, height, width, depth, mode = reader.unpack("HIIHH")
    if mode not in (1, 3) or depth not in (8, 16):
        raise PsdError("Only RGB or Grayscale PSDs at 8 or 16 bits per channel are supported.")
    if not (0 < width <= MAX_DIMENSION and 0 < height <= MAX_DIMENSION):
        raise PsdError("The PSD's size is invalid.")
    reader.take(reader.unpack("I"))  # colour mode data
    reader.take(reader.unpack("I"))  # image resources
    layer_mask_length = reader.unpack("I")
    section_end = reader.pos + layer_mask_length
    document = PsdDocument(width, height)
    if layer_mask_length:
        info_length = reader.unpack("I")
        if info_length:
            info_end = reader.pos + info_length
            _read_layers(reader, document, depth, mode)
            reader.pos = info_end
    reader.pos = section_end
    document.composite = _read_composite(reader, width, height, channels, depth, mode)
    return document


def _read_layers(reader: _Reader, document: PsdDocument, depth: int, mode: int) -> None:
    count = abs(reader.unpack("h"))
    records = []
    for _ in range(count):
        top, left, bottom, right = reader.unpack("iiii")
        channel_count = reader.unpack("H")
        channels = [reader.unpack("hI") for _ in range(channel_count)]
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
        section_type, fill = 0, 255
        while reader.pos + 12 <= extra_end:
            if reader.take(4) not in (b"8BIM", b"8B64"):
                break
            key = reader.take(4)
            length = reader.unpack("I")
            value = reader.take(length)
            reader.take(length % 2)
            if key == b"luni" and len(value) >= 4:
                (chars,) = struct.unpack(">I", value[:4])
                name = value[4:4 + 2 * chars].decode("utf-16-be", "replace").rstrip("\x00")
            elif key in (b"lsct", b"lsdk") and len(value) >= 4:
                (section_type,) = struct.unpack(">I", value[:4])
            elif key == b"iOpa" and value:
                fill = value[0]
        reader.pos = extra_end
        records.append({
            "rect": (top, left, bottom, right), "channels": channels, "blend": blend_key, "opacity": opacity,
            "visible": not flags & 2, "name": name, "section": section_type, "mask": mask,
            "clipping": bool(clipping), "fill": fill,
        })
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
            pixels[channel_id] = _decode_channel(compression, payload[2:], w, h, depth)
        record["pixels"] = pixels
    _build_layers(records, document, mode)


def _build_layers(records: list[dict], document: PsdDocument, mode: int) -> None:
    # Group visibility: read top-down, a folder record (section 1/2) opens a group and the bounding
    # divider (section 3) below its children closes it.
    hidden_depth, stack = 0, []
    visible_flags = [True] * len(records)
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        if record["section"] in (1, 2):
            stack.append(not record["visible"])
            hidden_depth += stack[-1]
        elif record["section"] == 3:
            if stack:
                hidden_depth -= stack.pop()
        else:
            visible_flags[index] = record["visible"] and hidden_depth == 0
    base = None  # (left, top, alpha) of the last unclipped layer -- what clipped layers above it clip to
    for record, visible in zip(records, visible_flags):
        if record["section"]:
            continue
        top, left, bottom, right = record["rect"]
        width, height = right - left, bottom - top
        colour_ids = (0, 1, 2) if mode == 3 else (0,)
        if width <= 0 or height <= 0 or any(channel not in record["pixels"] for channel in colour_ids):
            document.skipped += 1
            continue
        colour = [record["pixels"][channel] for channel in colour_ids]
        if mode == 1:
            colour = colour * 3
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
        document.layers.append(PsdLayer(
            record["name"] or "Layer", image, left, top, record["opacity"] / 255.0,
            BLEND_MODES.get(record["blend"], "normal"), visible,
        ))


def _read_composite(reader: _Reader, width: int, height: int, channels: int, depth: int, mode: int) -> Image.Image | None:
    if reader.pos + 2 > len(reader.data):
        return None
    compression = reader.unpack("H")
    plane = width * height * depth // 8
    colour = 3 if mode == 3 else 1
    try:
        if compression == 1:
            counts = struct.unpack(f">{height * channels}H", reader.take(2 * height * channels))
            planes = []
            for index in range(colour):
                rows = counts[index * height:(index + 1) * height]
                start = reader.pos + sum(counts[:index * height])
                payload = reader.data[start:start + sum(rows)]
                planes.append(_decode_channel(1, struct.pack(f">{height}H", *rows) + payload, width, height, depth))
        elif compression == 0:
            planes = [_decode_channel(0, reader.take(plane), width, height, depth) for _ in range(colour)]
        else:
            return None
    except PsdError:
        return None
    if mode == 1:
        return Image.fromarray(planes[0], mode="L").convert("RGB")
    return Image.fromarray(np.dstack(planes), mode="RGB")


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
