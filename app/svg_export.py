"""SVG export of the Layers stack (F27): Shape, Path and Text layers as real vector elements,
Image layers embedded, optionally over the edited photo -- for Illustrator, Inkscape, Figma or a
web page.

The SVG's coordinate system is the full frame the layers are drawn on (the post-geometry photo);
the Crop step is a ``viewBox`` onto it, so nothing is resampled. Per layer: name (``id`` and
``inkscape:label``), opacity, visibility (hidden layers are written with ``display="none"``) and
blend mode (CSS ``mix-blend-mode``; the few Photoshop modes CSS lacks fall back to normal).
Solid and gradient fills (``linearGradient``/``radialGradient``), outlines and the Pen tool's Bezier
curves are exact. Text uses the layer's font family name (the viewer needs that font installed).
Layer Styles (shadows, glows, bevels...) and adjustment layers have no SVG equivalent here and are
left out -- the PNG/PSD exports carry them.
"""
from __future__ import annotations

import base64
import io
import math
from xml.sax.saxutils import escape, quoteattr

from PIL import Image

from app.vector_shapes import HIGHLIGHTER_OPACITY, _polygon_points

CSS_BLEND = {
    "normal": "normal", "multiply": "multiply", "screen": "screen", "overlay": "overlay", "darken": "darken",
    "lighten": "lighten", "color_dodge": "color-dodge", "color_burn": "color-burn", "hard_light": "hard-light",
    "soft_light": "soft-light", "difference": "difference", "exclusion": "exclusion", "hue": "hue",
    "saturation": "saturation", "color": "color", "luminosity": "luminosity",
}


def _hex(color) -> str:
    if isinstance(color, str):
        return color if color.startswith("#") else f"#{color}"
    r, g, b = (int(v) for v in tuple(color)[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _num(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _png_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


class _Defs:
    def __init__(self):
        self.items: list[str] = []

    def gradient(self, layer: dict, box: tuple[float, float, float, float]) -> str:
        gid = f"grad{len(self.items)}"
        start, end = _hex(layer.get("fill_color", "#FFFFFF")), _hex(layer.get("fill_color2", "#000000"))
        opacity = _num(max(0.0, min(1.0, float(layer.get("fill_opacity", 1.0)))))
        stops = (f'<stop offset="0" stop-color="{start}" stop-opacity="{opacity}"/>'
                 f'<stop offset="1" stop-color="{end}" stop-opacity="{opacity}"/>')
        x0, y0, x1, y1 = box
        if layer.get("fill_style") == "Radial gradient":
            cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, max(1.0, math.hypot(x1 - x0, y1 - y0) / 2)
            self.items.append(f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" cx="{_num(cx)}" cy="{_num(cy)}" '
                              f'r="{_num(r)}">{stops}</radialGradient>')
        else:
            angle = math.radians(float(layer.get("gradient_angle", 0.0)))
            dx, dy = math.cos(angle), -math.sin(angle)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            half = max(1.0, (abs(dx) * (x1 - x0) + abs(dy) * (y1 - y0)) / 2)
            self.items.append(f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{_num(cx - dx * half)}" '
                              f'y1="{_num(cy - dy * half)}" x2="{_num(cx + dx * half)}" y2="{_num(cy + dy * half)}">{stops}</linearGradient>')
        return f"url(#{gid})"

    def arrow(self, color: str) -> str:
        """An arrow-head marker (F33) in `color`, sized in stroke widths like app.vector_shapes'."""
        mid = f"arrow{len(self.items)}"
        self.items.append(f'<marker id="{mid}" viewBox="0 0 4 2.4" refX="4" refY="1.2" markerWidth="4" markerHeight="2.4" '
                          f'orient="auto-start-reverse" markerUnits="strokeWidth"><path d="M 0 0 L 4 1.2 L 0 2.4 Z" fill="{color}"/></marker>')
        return f"url(#{mid})"


def _paint(layer: dict, defs: _Defs, box) -> str:
    """fill/stroke attributes of a Shape or closed Path."""
    parts = []
    if layer.get("fill_enabled", True):
        if layer.get("fill_style", "Solid") != "Solid":
            parts.append(f'fill="{defs.gradient(layer, box)}"')
        else:
            parts.append(f'fill="{_hex(layer.get("fill_color", "#FFFFFF"))}" fill-opacity="{_num(float(layer.get("fill_opacity", 1.0)))}"')
    else:
        parts.append('fill="none"')
    if layer.get("stroke_enabled", False):
        parts.append(f'stroke="{_hex(layer.get("stroke_color", "#000000"))}" stroke-width="{_num(float(layer.get("stroke_width", 4)))}" '
                     'stroke-linejoin="round"')
    return " ".join(parts)


def _arrow_attributes(layer: dict, color: str, defs: _Defs) -> str:
    parts = []
    if layer.get("arrow_start", False):
        parts.append(f' marker-start="{defs.arrow(color)}"')
    if layer.get("arrow_end", False):
        parts.append(f' marker-end="{defs.arrow(color)}"')
    return "".join(parts)


def _shape_element(layer: dict, width: int, height: int, defs: _Defs) -> str:
    cx, cy = width * float(layer.get("x_pct", 50.0)) / 100, height * float(layer.get("y_pct", 50.0)) / 100
    hw, hh = width * max(0.5, float(layer.get("width_pct", 30.0))) / 200, height * max(0.5, float(layer.get("height_pct", 30.0))) / 200
    rotation = float(layer.get("rotation_deg", 0.0))
    transform = f' transform="rotate({_num(rotation)} {_num(cx)} {_num(cy)})"' if rotation else ""
    shape = layer.get("shape_type", "Rectangle")
    box = (cx - hw, cy - hh, cx + hw, cy + hh)
    if shape == "Line":
        color = _hex(layer.get("fill_color", "#FFFFFF"))
        return (f'<line x1="{_num(cx - hw)}" y1="{_num(cy)}" x2="{_num(cx + hw)}" y2="{_num(cy)}" stroke="{color}" '
                f'stroke-width="{_num(float(layer.get("line_thickness", 6)))}"{_arrow_attributes(layer, color, defs)}{transform}/>')
    paint = _paint(layer, defs, box)
    if shape == "Rectangle":
        return f'<rect x="{_num(cx - hw)}" y="{_num(cy - hh)}" width="{_num(2 * hw)}" height="{_num(2 * hh)}" {paint}{transform}/>'
    if shape == "Ellipse":
        return f'<ellipse cx="{_num(cx)}" cy="{_num(cy)}" rx="{_num(hw)}" ry="{_num(hh)}" {paint}{transform}/>'
    inner = max(0.05, min(0.95, float(layer.get("inner_ratio", 0.5)))) if shape == "Star" else None
    points = _polygon_points((cx, cy), hw, hh, int(layer.get("sides", 5)), rotation, inner_ratio=inner)
    return f'<polygon points="{" ".join(f"{_num(x)},{_num(y)}" for x, y in points)}" {paint}/>'


def path_d(points: list[dict], width: int, height: int, closed: bool) -> str:
    """The Pen path as SVG path data: one cubic Bezier per segment, with the same symmetric handles
    app.vector_shapes.path_outline_points samples."""
    anchors = [(p["x_pct"] / 100 * width, p["y_pct"] / 100 * height) for p in points]
    handles = [(p.get("handle_dx_pct", 0.0) / 100 * width, p.get("handle_dy_pct", 0.0) / 100 * height) for p in points]
    d = [f"M {_num(anchors[0][0])} {_num(anchors[0][1])}"]
    count = len(anchors)
    for i in range(count if closed else count - 1):
        p0, p1 = anchors[i], anchors[(i + 1) % count]
        h0, h1 = handles[i], handles[(i + 1) % count]
        d.append(f"C {_num(p0[0] + h0[0])} {_num(p0[1] + h0[1])} {_num(p1[0] - h1[0])} {_num(p1[1] - h1[1])} {_num(p1[0])} {_num(p1[1])}")
    if closed:
        d.append("Z")
    return " ".join(d)


def _path_element(layer: dict, width: int, height: int, defs: _Defs) -> str | None:
    points = layer.get("points") or []
    if len(points) < 2:
        return None
    closed = bool(layer.get("closed", False))
    xs = [p["x_pct"] / 100 * width for p in points]
    ys = [p["y_pct"] / 100 * height for p in points]
    paint_layer = {**layer, "fill_enabled": closed and layer.get("fill_enabled", False), "stroke_enabled": layer.get("stroke_enabled", True),
                   "stroke_width": layer.get("stroke_width", 3), "stroke_color": layer.get("stroke_color", "#FFFFFF")}
    extra = ""
    if not closed and paint_layer["stroke_enabled"]:
        color = _hex(paint_layer["stroke_color"])
        extra = _arrow_attributes(layer, color, defs)
        if layer.get("highlighter", False):
            extra += f' stroke-opacity="{_num(HIGHLIGHTER_OPACITY)}"'
    return f'<path d="{path_d(points, width, height, closed)}" {_paint(paint_layer, defs, (min(xs), min(ys), max(xs), max(ys)))} stroke-linecap="round"{extra}/>'


def _text_element(layer: dict, width: int, height: int) -> str | None:
    text = layer.get("text", "")
    if not text:
        return None
    size = float(layer.get("font_size_pct", 6.0)) / 100 * min(width, height)
    x, y = width * float(layer.get("x_pct", 50.0)) / 100, height * float(layer.get("y_pct", 50.0)) / 100
    lines = text.split("\n")
    anchor = {"left": "start", "center": "middle", "right": "end"}.get(layer.get("align", "left"), "start")
    # The editor centres the whole block on (x, y): the block's left/right edge for start/end text
    # is estimated from the longest line (about 0.55 em per character).
    block_half = max(len(line) for line in lines) * size * 0.55 / 2
    x_line = x - block_half if anchor == "start" else (x + block_half if anchor == "end" else x)
    first_dy = -(len(lines) - 1) * 1.2 * size / 2
    family = layer.get("font") or "sans-serif"
    spans = "".join(
        f'<tspan x="{_num(x_line)}" dy="{_num(first_dy if i == 0 else 1.2 * size)}">{escape(line) or " "}</tspan>'
        for i, line in enumerate(lines))
    return (f'<text x="{_num(x_line)}" y="{_num(y)}" font-size="{_num(size)}" font-family={quoteattr(family)} '
            f'fill="{_hex(layer.get("color", "#FFFFFF"))}" text-anchor="{anchor}" dominant-baseline="middle">{spans}</text>')


def _image_element(layer: dict, width: int, height: int) -> str | None:
    if not layer.get("image_b64"):
        return None
    content = Image.open(io.BytesIO(base64.b64decode(layer["image_b64"])))
    href = f"data:image/png;base64,{layer['image_b64']}"
    if layer.get("stretch", "x_pct" not in layer):
        return f'<image href="{href}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none"/>'
    fit = min(width / content.width, height / content.height) * float(layer.get("scale_pct", 100.0)) / 100
    w, h = content.width * fit, content.height * fit
    cx, cy = width * float(layer.get("x_pct", 50.0)) / 100, height * float(layer.get("y_pct", 50.0)) / 100
    rotation = float(layer.get("rotation_deg", 0.0))
    transform = f' transform="rotate({_num(rotation)} {_num(cx)} {_num(cy)})"' if rotation else ""
    return f'<image href="{href}" x="{_num(cx - w / 2)}" y="{_num(cy - h / 2)}" width="{_num(w)}" height="{_num(h)}"{transform}/>'


def layers_to_svg(layers: list[dict], frame_size: tuple[int, int], crop: tuple[int, int, int, int] | None = None,
                  background: Image.Image | None = None) -> str:
    """The Layers stack (bottom first, the session's layer dicts) as an SVG document. `frame_size`
    is the full frame the layers are placed on; `crop` (left, top, right, bottom on that frame)
    becomes the viewBox; `background` (the edited photo, already cropped) goes underneath."""
    width, height = frame_size
    left, top, right, bottom = crop or (0, 0, width, height)
    defs = _Defs()
    body = []
    if background is not None:
        body.append(f'<image id="photo" href="{_png_data_uri(background.convert("RGB"), "JPEG")}" x="{left}" y="{top}" '
                    f'width="{right - left}" height="{bottom - top}" preserveAspectRatio="none"/>')
    for index, layer in enumerate(layers):
        kind = layer.get("kind", "image")
        if kind == "shape":
            element = _shape_element(layer, width, height, defs)
        elif kind == "path":
            element = _path_element(layer, width, height, defs)
        elif kind == "text":
            element = _text_element(layer, width, height)
        elif kind == "image":
            element = _image_element(layer, width, height)
        else:
            element = None  # adjustment layers: no SVG counterpart
        if element is None:
            continue
        name = str(layer.get("name") or f"Layer {index + 1}")
        style = []
        blend = CSS_BLEND.get(layer.get("blend_mode", "normal"), "normal")
        if blend != "normal":
            style.append(f"mix-blend-mode:{blend}")
        attrs = [f'id="layer{index + 1}"', f"inkscape:label={quoteattr(name)}", 'inkscape:groupmode="layer"']
        if float(layer.get("opacity", 1.0)) < 1.0:
            attrs.append(f'opacity="{_num(float(layer["opacity"]))}"')
        if not layer.get("visible", True):
            attrs.append('display="none"')
        if style:
            attrs.append(f'style="{";".join(style)}"')
        body.append(f'<g {" ".join(attrs)}>{element}</g>')
    defs_xml = f"<defs>{''.join(defs.items)}</defs>" if defs.items else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{right - left}" height="{bottom - top}" viewBox="{left} {top} {right - left} {bottom - top}">'
        f"{defs_xml}{''.join(body)}</svg>\n"
    )
