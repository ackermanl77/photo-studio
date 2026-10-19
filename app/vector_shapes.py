from __future__ import annotations

import math

from PIL import Image, ImageDraw

# Two families of vector content live here: the named-primitive Shapes below (Rectangle/Ellipse/
# Line/Polygon/Star, straight-edged, parametrized by position/size/rotation) and the freeform Pen
# tool path further down (render_path_layer/PATH_POINT_DEFAULTS -- arbitrary anchor points with
# draggable Bezier curve handles). Both stay parametric in the Layers tab's own session_state (never
# only a baked bitmap), the same "definitions stay editable, only rasterize on demand" approach
# app.advanced_masks.MaskLayer's selections and Text layers (app.layer_workspace.render_text_layer)
# already use.
SHAPE_TYPES = ("Rectangle", "Ellipse", "Line", "Polygon", "Star")
# Fill styles of a Shape or closed Path (F27): a flat colour, or a gradient from fill_color to
# fill_color2 across the shape's own bounds -- linear at gradient_angle degrees (0 = left to right,
# 90 = bottom to top), or radial from its centre outwards.
FILL_STYLES = ("Solid", "Linear gradient", "Radial gradient")


def gradient_fill(alpha: Image.Image, params: dict) -> Image.Image:
    """The gradient filling `alpha`'s shape (RGBA, alpha scaled by fill_opacity)."""
    import numpy as np

    from app.layer_styles import _gradient_overlay_layer

    start = tuple(params.get("fill_color", (255, 255, 255)))[:3]
    end = tuple(params.get("fill_color2", (0, 0, 0)))[:3]
    opacity = max(0.0, min(1.0, params.get("fill_opacity", 1.0)))
    if params.get("fill_style") != "Radial gradient":
        return _gradient_overlay_layer(alpha, start, end, params.get("gradient_angle", 0.0), opacity)
    width, height = alpha.size
    x0, y0, x1, y1 = alpha.getbbox() or (0, 0, width, height)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = max(1.0, math.hypot(x1 - x0, y1 - y0) / 2.0)
    position = np.clip(np.hypot(xx - (x0 + x1) / 2.0, yy - (y0 + y1) / 2.0) / radius, 0.0, 1.0)[..., None]
    rgb = np.asarray(start, np.float32) * (1 - position) + np.asarray(end, np.float32) * position
    a = np.asarray(alpha, np.float32) * opacity
    return Image.fromarray(np.dstack([rgb, a]).round().astype(np.uint8), mode="RGBA")


def _with_gradient(render, canvas_size: tuple[int, int], params: dict) -> Image.Image:
    """Render `params` with its gradient fill: the shape's filled silhouette drives gradient_fill(),
    and the outline (when on) is drawn on top, as a solid fill would have it."""
    silhouette = render(canvas_size, {**params, "fill_style": "Solid", "fill_enabled": True, "fill_color": (255, 255, 255),
                                      "fill_opacity": 1.0, "stroke_enabled": False})
    image = gradient_fill(silhouette.getchannel("A"), params)
    if params.get("stroke_enabled", False):
        image = Image.alpha_composite(image, render(canvas_size, {**params, "fill_style": "Solid", "fill_enabled": False}))
    return image


def _rotate_point(point: tuple[float, float], center: tuple[float, float], angle_deg: float) -> tuple[float, float]:
    if not angle_deg:
        return point
    angle = math.radians(angle_deg)
    dx, dy = point[0] - center[0], point[1] - center[1]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (center[0] + dx * cos_a - dy * sin_a, center[1] + dx * sin_a + dy * cos_a)


def _polygon_points(
    center: tuple[float, float], radius_x: float, radius_y: float, sides: int, rotation_deg: float, inner_ratio: float | None = None,
) -> list[tuple[float, float]]:
    """Vertices of a regular polygon (or, with `inner_ratio`, a star -- alternating an outer vertex
    at full radius with an inner one scaled by `inner_ratio`), centered on `center` and rotated by
    `rotation_deg`. The first vertex always points straight up before rotation, matching how
    Photoshop's own Polygon/Star tool orients a shape at rotation 0."""
    sides = max(3, int(sides))
    count = sides * 2 if inner_ratio is not None else sides
    points = []
    for i in range(count):
        angle = -90 + i * (360 / count)
        scale = 1.0 if (inner_ratio is None or i % 2 == 0) else inner_ratio
        raw = (center[0] + radius_x * scale * math.cos(math.radians(angle)), center[1] + radius_y * scale * math.sin(math.radians(angle)))
        points.append(_rotate_point(raw, center, rotation_deg))
    return points


def _ellipse_points(center: tuple[float, float], radius_x: float, radius_y: float, rotation_deg: float) -> list[tuple[float, float]]:
    """A rotated ellipse as a polygon, about one vertex per 2 px of its outline (at least 64), so the
    facets stay under a pixel at the canvas's own resolution."""
    perimeter = math.pi * (3 * (radius_x + radius_y) - math.sqrt((3 * radius_x + radius_y) * (radius_x + 3 * radius_y)))
    count = max(64, int(perimeter / 2))
    cx, cy = center
    return [
        _rotate_point((cx + radius_x * math.cos(2 * math.pi * i / count), cy + radius_y * math.sin(2 * math.pi * i / count)),
                      center, rotation_deg)
        for i in range(count)
    ]


def render_shape_layer(canvas_size: tuple[int, int], params: dict) -> Image.Image:
    """Render one parametric vector shape into a transparent RGBA image at the source photo's real
    pixel size. `params` (matching one entry of st.session_state.image_layers for a "shape"-kind
    layer):
      - shape_type: one of SHAPE_TYPES
      - x_pct/y_pct: the shape's own center, as a fraction (0-100) of the canvas
      - width_pct/height_pct: the shape's bounding box size, as a fraction (0-100) of the canvas's
        own width/height (independent axes -- a Polygon/Star with width_pct != height_pct stretches
        into an ellipse-like variant rather than staying perfectly regular, a deliberate creative
        option rather than forcing symmetry)
      - rotation_deg: turns the shape about its center (an Ellipse at a multiple of 180 degrees is
        drawn with PIL's exact ellipse; any other angle as a finely sampled rotated polygon, since
        PIL has no rotated-ellipse primitive)
      - fill_enabled/fill_color/fill_opacity, stroke_enabled/stroke_color/stroke_width: the shape's
        own vector fill/outline -- distinct from a Layer Style's "Stroke" effect (app.layer_styles),
        which dilates whatever silhouette a layer already has rather than defining the shape itself
      - sides (Polygon/Star, 3-12), inner_ratio (Star only, 0.05-0.95): the inner vertices' radius
        as a fraction of the outer one
      - line_thickness (Line only): a Line has no separate fill/stroke, just one paint color/width
    """
    if (params.get("fill_style", "Solid") != "Solid" and params.get("fill_enabled", True)
            and params.get("shape_type", "Rectangle") != "Line"):
        return _with_gradient(render_shape_layer, canvas_size, params)
    width, height = canvas_size
    image = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    shape_type = params.get("shape_type", "Rectangle")
    cx = width * max(0.0, min(100.0, params.get("x_pct", 50.0))) / 100.0
    cy = height * max(0.0, min(100.0, params.get("y_pct", 50.0))) / 100.0
    half_w = width * max(0.5, params.get("width_pct", 30.0)) / 200.0
    half_h = height * max(0.5, params.get("height_pct", 30.0)) / 200.0
    rotation = params.get("rotation_deg", 0.0)
    fill_rgb = tuple(params.get("fill_color", (255, 255, 255)))
    fill = (*fill_rgb, int(255 * max(0.0, min(1.0, params.get("fill_opacity", 1.0))))) if params.get("fill_enabled", True) else None
    stroke_enabled = params.get("stroke_enabled", False)
    stroke_width = max(1, int(params.get("stroke_width", 2))) if stroke_enabled else 0
    outline = (*tuple(params.get("stroke_color", (0, 0, 0))), 255) if stroke_enabled else None

    if shape_type == "Line":
        thickness = max(1, int(params.get("line_thickness", 4)))
        p1 = _rotate_point((cx - half_w, cy), (cx, cy), rotation)
        p2 = _rotate_point((cx + half_w, cy), (cx, cy), rotation)
        draw.line([p1, p2], fill=fill or (*fill_rgb, 255), width=thickness)
        return image

    if shape_type == "Ellipse":
        if rotation % 180.0 == 0.0 or half_w == half_h:  # unrotated, or a circle: rotation changes nothing
            draw.ellipse([cx - half_w, cy - half_h, cx + half_w, cy + half_h], fill=fill, outline=outline, width=stroke_width or 1)
            return image
        points = _ellipse_points((cx, cy), half_w, half_h, rotation)
        draw.polygon(points, fill=fill, outline=outline, width=stroke_width or 1)
        return image

    if shape_type == "Rectangle":
        corners = [(cx - half_w, cy - half_h), (cx + half_w, cy - half_h), (cx + half_w, cy + half_h), (cx - half_w, cy + half_h)]
        points = [_rotate_point(point, (cx, cy), rotation) for point in corners]
    elif shape_type == "Star":
        inner_ratio = max(0.05, min(0.95, params.get("inner_ratio", 0.5)))
        points = _polygon_points((cx, cy), half_w, half_h, params.get("sides", 5), rotation, inner_ratio=inner_ratio)
    else:  # Polygon
        points = _polygon_points((cx, cy), half_w, half_h, params.get("sides", 5), rotation)

    draw.polygon(points, fill=fill, outline=outline, width=stroke_width or 1)
    return image


# --- Pen tool: a freeform path of anchor points connected by cubic Bezier curves -----------------
#
# Each point is a dict {"x_pct", "y_pct", "handle_dx_pct", "handle_dy_pct"} -- position plus one
# handle stored as an (dx, dy) OFFSET from the anchor, used symmetrically on both sides of that
# anchor (the curve arriving at the point and the curve leaving it share the same tangent line, in
# opposite directions). This models a "smooth" point in every vector editor's terms; a "corner"
# point (independent, non-mirrored in/out handles, e.g. Photoshop's Alt-drag) is a real, deliberate
# gap -- see docs/PROJECT_STATUS.md's "Known Issues". A zero-length handle degenerates the segment
# to a straight line, so corner-only paths (no curves at all) are still fully expressible.
PATH_POINT_DEFAULTS = {"x_pct": 50.0, "y_pct": 50.0, "handle_dx_pct": 0.0, "handle_dy_pct": 0.0}


def _cubic_bezier_points(p0: tuple[float, float], c1: tuple[float, float], c2: tuple[float, float], p1: tuple[float, float], steps: int = 24) -> list[tuple[float, float]]:
    """Sample a cubic Bezier segment into `steps + 1` straight-line points -- PIL's ImageDraw has
    no curve-fill/curve-stroke primitive, so every curved segment is rasterized as a fine polyline,
    the same "approximate a curve as enough short straight segments" approach OpenCV's own drawing
    functions use internally."""
    points = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t**2 * c2[0] + t**3 * p1[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t**2 * c2[1] + t**3 * p1[1]
        points.append((x, y))
    return points


def path_outline_points(points: list[dict], canvas_size: tuple[int, int], closed: bool, steps_per_segment: int = 24) -> list[tuple[float, float]]:
    """The full rasterized outline (in real pixel coordinates) of a Pen tool path -- every
    consecutive pair of anchors becomes one sampled Bezier segment; if `closed`, one more segment
    connects the last anchor back to the first. Shared between render_path_layer() below and the
    on-canvas Pen editor's own preview, so the path drawn while editing and the path actually
    rendered into the layer can never visually disagree."""
    width, height = canvas_size
    if len(points) < 2:
        return [(p["x_pct"] / 100.0 * width, p["y_pct"] / 100.0 * height) for p in points]

    anchors = [(p["x_pct"] / 100.0 * width, p["y_pct"] / 100.0 * height) for p in points]
    handles = [(p.get("handle_dx_pct", 0.0) / 100.0 * width, p.get("handle_dy_pct", 0.0) / 100.0 * height) for p in points]
    count = len(anchors)
    segment_count = count if closed else count - 1

    outline = [anchors[0]]
    for i in range(segment_count):
        p0, p1 = anchors[i], anchors[(i + 1) % count]
        h0, h1 = handles[i], handles[(i + 1) % count]
        control1 = (p0[0] + h0[0], p0[1] + h0[1])
        control2 = (p1[0] - h1[0], p1[1] - h1[1])
        segment = _cubic_bezier_points(p0, control1, control2, p1, steps_per_segment)
        outline.extend(segment[1:])  # [0] duplicates the previous segment's own last point
    return outline


def render_path_layer(canvas_size: tuple[int, int], params: dict) -> Image.Image:
    """Render a Pen tool path (`params["points"]`, `params["closed"]`) into a transparent RGBA
    image. An open path can only be stroked (there's no well-defined "inside" to fill); a closed
    one can be filled, stroked, or both -- same fill_enabled/fill_color/fill_opacity/stroke_enabled/
    stroke_color/stroke_width fields as a Shape layer's own vector fill+outline, so the two share
    the exact same UI vocabulary."""
    points = params.get("points", [])
    image = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    if len(points) < 2:
        return image

    closed = bool(params.get("closed", False))
    if closed and params.get("fill_enabled", False) and params.get("fill_style", "Solid") != "Solid":
        return _with_gradient(render_path_layer, canvas_size, params)
    outline = path_outline_points(points, canvas_size, closed)

    stroke_enabled = params.get("stroke_enabled", True)
    stroke_width = max(1, int(params.get("stroke_width", 3))) if stroke_enabled else 0
    stroke_color = (*tuple(params.get("stroke_color", (255, 255, 255))), 255) if stroke_enabled else None

    if closed:
        fill_rgb = tuple(params.get("fill_color", (255, 255, 255)))
        fill = (*fill_rgb, int(255 * max(0.0, min(1.0, params.get("fill_opacity", 1.0))))) if params.get("fill_enabled", False) else None
        draw = ImageDraw.Draw(image)
        draw.polygon(outline, fill=fill, outline=stroke_color, width=stroke_width or 1)
    elif stroke_enabled:
        draw = ImageDraw.Draw(image)
        draw.line(outline, fill=stroke_color, width=stroke_width, joint="curve")

    return image
