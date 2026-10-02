from __future__ import annotations

import base64
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.major_features import Layer, LayerStack

# A short, well-known list of TrueType fonts likely to already be installed, checked in order --
# deliberately not a bundled font file (this app avoids adding binary assets/new dependencies
# elsewhere too, e.g. no bundled lens/ICC profile database) and not a font picker; Text layers get
# whatever real TrueType font this machine happens to have, falling back to Pillow's own built-in
# bitmap font (PIL.ImageFont.load_default()) when none of these exist -- legible but not scalable
# the way a real font is, and the one realistic gap this leaves (see docs/PROJECT_STATUS.md's
# "Known Issues"). Covers Windows, common Linux distros, and macOS.
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(1, int(size))
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1 -- a scalable fallback
    except TypeError:
        return ImageFont.load_default()  # older Pillow: a small, fixed-size bitmap font


def render_text_layer(
    canvas_size: tuple[int, int],
    text: str,
    font_size: float = 48,
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
    align: str = "left",
    x_pct: float = 50.0,
    y_pct: float = 50.0,
) -> Image.Image:
    """Render a text layer's transparent RGBA image at the source photo's actual pixel size --
    kept parametric (text/font_size/color/align/position) in the Layers tab's own session_state
    rather than only ever existing as a baked bitmap, the same "definitions stay editable, only
    rasterize on demand" approach app.advanced_masks.MaskLayer's parametric selections already use.
    `x_pct`/`y_pct` place the text block's own center (not its top-left corner) at that fraction of
    the canvas -- simpler to reason about when centering a title than tracking the block's size."""
    image = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    if not text:
        return image
    draw = ImageDraw.Draw(image)
    font = _load_font(font_size)
    x = canvas_size[0] * max(0.0, min(100.0, x_pct)) / 100.0
    y = canvas_size[1] * max(0.0, min(100.0, y_pct)) / 100.0
    draw.multiline_text((x, y), text, font=font, fill=tuple(color), align=align, anchor="mm")
    return image


def render_image_layer(
    canvas_size: tuple[int, int],
    image: Image.Image,
    x_pct: float = 50.0,
    y_pct: float = 50.0,
    scale_pct: float = 100.0,
    rotation_deg: float = 0.0,
) -> Image.Image:
    """Place an image layer on a transparent canvas-sized RGBA image, keeping its aspect ratio (F5
    in the improvement report -- every image layer used to be stretched to the canvas). At
    scale 100 % it's fitted inside the canvas; `x_pct`/`y_pct` place its center; `rotation_deg`
    turns it clockwise. Everything is relative to the canvas, so a proxy preview and the
    full-resolution export place it identically."""
    width, height = canvas_size
    content = image.convert("RGBA")
    fit = min(width / content.width, height / content.height) * max(0.01, float(scale_pct)) / 100.0
    size = (max(1, round(content.width * fit)), max(1, round(content.height * fit)))
    content = content.resize(size, Image.Resampling.LANCZOS)
    if rotation_deg:
        content = content.rotate(-float(rotation_deg), resample=Image.Resampling.BICUBIC, expand=True)
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    left = round(width * float(x_pct) / 100.0 - content.width / 2)
    top = round(height * float(y_pct) / 100.0 - content.height / 2)
    canvas.alpha_composite(content, (max(0, left), max(0, top)), (max(0, -left), max(0, -top)))
    return canvas


@dataclass
class WorkspaceLayer:
    name: str
    image: Image.Image
    opacity: float = 1.0
    blend_mode: str = "normal"
    visible: bool = True
    mask: Image.Image | None = None


def _layer_to_dict(layer: WorkspaceLayer) -> dict:
    image_buffer = io.BytesIO()
    layer.image.save(image_buffer, format="PNG")
    mask = None
    if layer.mask:
        mask_buffer = io.BytesIO()
        layer.mask.save(mask_buffer, format="PNG")
        mask = base64.b64encode(mask_buffer.getvalue()).decode("ascii")
    return {
        "name": layer.name,
        "image": base64.b64encode(image_buffer.getvalue()).decode("ascii"),
        "opacity": layer.opacity,
        "blend_mode": layer.blend_mode,
        "visible": layer.visible,
        "mask": mask,
    }


def _layer_from_dict(data: dict) -> WorkspaceLayer:
    image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGBA")
    mask = Image.open(io.BytesIO(base64.b64decode(data["mask"]))).convert("L") if data.get("mask") else None
    return WorkspaceLayer(data["name"], image, data.get("opacity", 1.0), data.get("blend_mode", "normal"), data.get("visible", True), mask)


def layers_to_json(layers: list[WorkspaceLayer]) -> str:
    """Serialize a plain list of overlay layers (no mandatory "Background" entry) -- the wire
    format app/backend_client.py sends to app/backend.py's /process endpoint for the Layers tab's
    now-multi-layer stack. Uses the exact same per-layer shape as LayerWorkspace.to_json() (see
    _layer_to_dict/_layer_from_dict above) so a saved .photoproject's layer_workspace field and a
    live edit's request body are never two different formats to maintain."""
    return json.dumps({"version": 1, "layers": [_layer_to_dict(layer) for layer in layers]}, sort_keys=True)


def layers_from_json(content: str | bytes) -> list[WorkspaceLayer]:
    payload = json.loads(content)
    return [_layer_from_dict(data) for data in payload.get("layers", [])]


class LayerWorkspace:
    def __init__(self, base: Image.Image):
        self.size = base.size
        self.layers = [WorkspaceLayer("Background", base.convert("RGBA"))]

    def add(self, layer: WorkspaceLayer) -> "LayerWorkspace":
        if layer.image.size != self.size:
            layer.image = layer.image.resize(self.size, Image.Resampling.LANCZOS)
        if layer.blend_mode not in LayerStack.MODES:
            raise ValueError(f"Unsupported blend mode: {layer.blend_mode}")
        self.layers.append(layer)
        return self

    def remove(self, name: str) -> "LayerWorkspace":
        if name == "Background":
            raise ValueError("The background layer cannot be removed")
        self.layers = [layer for layer in self.layers if layer.name != name]
        return self

    def duplicate(self, name: str) -> "LayerWorkspace":
        original = self._find(name)
        self.layers.append(WorkspaceLayer(f"{name} copy", original.image.copy(), original.opacity, original.blend_mode, original.visible, original.mask.copy() if original.mask else None))
        return self

    def move(self, name: str, offset: int) -> "LayerWorkspace":
        index = next(index for index, layer in enumerate(self.layers) if layer.name == name)
        target = max(1, min(len(self.layers) - 1, index + int(offset)))
        self.layers.insert(target, self.layers.pop(index))
        return self

    def composite(self) -> Image.Image:
        stack = LayerStack(self.layers[0].image)
        for layer in self.layers[1:]:
            stack.add(Layer(layer.name, layer.image, layer.opacity, layer.blend_mode, layer.mask, layer.visible))
        return stack.composite()

    def to_json(self) -> str:
        payload = {"version": 1, "size": self.size, "layers": [_layer_to_dict(layer) for layer in self.layers]}
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, content: str | bytes) -> "LayerWorkspace":
        payload = json.loads(content)
        if payload.get("version") != 1 or not payload.get("layers"):
            raise ValueError("Invalid layer workspace")
        layers = [_layer_from_dict(data) for data in payload["layers"]]
        workspace = cls(layers[0].image)
        workspace.layers = layers
        workspace.size = tuple(payload.get("size", workspace.size))
        return workspace

    def _find(self, name: str) -> WorkspaceLayer:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(name)
