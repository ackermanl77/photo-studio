"""Liquify (F23): a parametric mesh-warp brush, applied with ``cv2.remap``.

The edit is a list of strokes, never pixels -- each ``{"tool", "size", "strength", "points"}`` with
the points as fractions of the frame (after the rest of the Geometry step), the brush radius
(``size``) as a fraction of the frame's short edge and ``strength`` 0.05..1. So a stroke survives
Undo/Redo, presets-free projects and a different preview size, and can be removed later.

Tools:
- ``push`` (Forward Warp): drags the pixels under the brush along the stroke;
- ``bloat`` / ``pinch``: pushes them away from / pulls them toward the brush centre;
- ``twirl_cw`` / ``twirl_ccw``: rotates them about the brush centre.

How: the strokes are replayed, stamp by stamp, into a displacement field on a fixed-size grid
(``GRID_LONG_EDGE`` on the long side, whatever the image's resolution -- so a proxy preview and the
full-resolution export warp identically). Each stamp is an inverse map ``f`` local to the brush
(smooth ``(1 - (r/R)^2)^2`` falloff), composed onto the field: ``D'(x) = f(x) - x + D(f(x))`` with the
old field sampled bilinearly, so later strokes move the already-warped pixels, as in an interactive
liquify. The field is then scaled up to the image and applied once with ``cv2.remap`` (bilinear,
edge pixels reflected), in row chunks, for 8-bit images and the 16-bit path's float arrays alike.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

TOOLS = ("push", "bloat", "pinch", "twirl_cw", "twirl_ccw")
TOOL_LABELS = {"push": "Forward Warp", "bloat": "Bloat", "pinch": "Pucker", "twirl_cw": "Twirl clockwise", "twirl_ccw": "Twirl counter-clockwise"}
GRID_LONG_EDGE = 512
MAX_STROKES = 200
MAX_POINTS = 512
SIZE_RANGE = (0.01, 0.5)
STRENGTH_RANGE = (0.05, 1.0)
# Per stamp: how far bloat/pinch scale toward/away from the centre, and twirl's angle (radians), at full strength.
_RADIAL_RATE = 0.12
_TWIRL_RATE = 0.12
_STAMP_SPACING = 0.2  # of the radius
_CHUNK_ROWS = 512


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def clean_stroke(stroke) -> dict | None:
    """One stroke made safe (clamped), or None when it isn't one."""
    if not isinstance(stroke, dict) or stroke.get("tool") not in TOOLS:
        return None
    size, strength = _number(stroke.get("size")), _number(stroke.get("strength"))
    points = stroke.get("points")
    if size is None or strength is None or not isinstance(points, (list, tuple)) or not points:
        return None
    clean_points = []
    for point in list(points)[:MAX_POINTS]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        x, y = _number(point[0]), _number(point[1])
        if x is None or y is None:
            return None
        clean_points.append([round(min(max(x, 0.0), 1.0), 5), round(min(max(y, 0.0), 1.0), 5)])
    return {
        "tool": stroke["tool"],
        "size": round(min(max(size, SIZE_RANGE[0]), SIZE_RANGE[1]), 4),
        "strength": round(min(max(strength, STRENGTH_RANGE[0]), STRENGTH_RANGE[1]), 3),
        "points": clean_points,
    }


def clean_strokes(value) -> list[dict]:
    """A recipe's/option's stroke list, malformed entries dropped, at most MAX_STROKES."""
    if not isinstance(value, (list, tuple)):
        return []
    return [stroke for stroke in map(clean_stroke, list(value)[:MAX_STROKES]) if stroke is not None]


def grid_size(width: int, height: int) -> tuple[int, int]:
    scale = GRID_LONG_EDGE / max(width, height, 1)
    return max(2, round(width * scale)), max(2, round(height * scale))


def _stamps(stroke: dict, grid_w: int, grid_h: int):
    """(centre x, centre y, push dx, push dy) per stamp, in grid pixels."""
    radius = stroke["size"] * min(grid_w, grid_h)
    points = [(x * (grid_w - 1), y * (grid_h - 1)) for x, y in stroke["points"]]
    spacing = max(0.5, radius * _STAMP_SPACING)
    if stroke["tool"] != "push" and len(points) == 1:
        for _ in range(4):  # a click holds the brush down for a moment
            yield points[0][0], points[0][1], 0.0, 0.0
        return
    if stroke["tool"] != "push":
        yield points[0][0], points[0][1], 0.0, 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, math.ceil(length / spacing))
        dx, dy = (x1 - x0) / steps, (y1 - y0) / steps
        for step in range(1, steps + 1):
            yield x0 + dx * (step - 1), y0 + dy * (step - 1), dx, dy


def displacement_field(strokes: list[dict], width: int, height: int) -> np.ndarray | None:
    """The grid's (gh, gw, 2) inverse displacement in grid pixels, or None without a stroke."""
    strokes = clean_strokes(strokes)
    if not strokes:
        return None
    grid_w, grid_h = grid_size(width, height)
    field = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    for stroke in strokes:
        tool, strength = stroke["tool"], stroke["strength"]
        radius = max(1.0, stroke["size"] * min(grid_w, grid_h))
        for cx, cy, push_x, push_y in _stamps(stroke, grid_w, grid_h):
            x0, x1 = max(0, int(cx - radius)), min(grid_w, int(cx + radius) + 2)
            y0, y1 = max(0, int(cy - radius)), min(grid_h, int(cy + radius) + 2)
            if x0 >= x1 or y0 >= y1:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            rx, ry = xs - cx, ys - cy
            falloff = np.clip(1.0 - (rx * rx + ry * ry) / (radius * radius), 0.0, 1.0) ** 2 * strength
            if tool == "push":
                fx, fy = xs - falloff * push_x, ys - falloff * push_y
            elif tool in ("bloat", "pinch"):
                rate = _RADIAL_RATE * falloff * (1.0 if tool == "bloat" else -1.0)
                fx, fy = xs - rx * rate, ys - ry * rate
            else:
                # Image y points down, so a positive angle here turns the content clockwise on screen.
                angle = _TWIRL_RATE * falloff * (1.0 if tool == "twirl_cw" else -1.0)
                cos, sin = np.cos(-angle), np.sin(-angle)
                fx, fy = cx + rx * cos - ry * sin, cy + rx * sin + ry * cos
            old = cv2.remap(field, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            field[y0:y1, x0:x1, 0] = fx - xs + old[:, :, 0]
            field[y0:y1, x0:x1, 1] = fy - ys + old[:, :, 1]
    return field


def apply_liquify(image, strokes) -> Image.Image | np.ndarray:
    """`image` (PIL, or the 16-bit path's float HxWx3 array) warped by `strokes`; unchanged without any."""
    is_array = isinstance(image, np.ndarray)
    width, height = (image.shape[1], image.shape[0]) if is_array else image.size
    field = displacement_field(strokes, width, height)
    if field is None:
        return image
    grid_h, grid_w = field.shape[:2]
    if is_array:
        array = np.ascontiguousarray(image, dtype=np.float32)
    else:
        mode = image.mode if image.mode in ("RGB", "RGBA", "L") else "RGB"
        array = np.asarray(image.convert(mode))
    scale_x, scale_y = (width - 1) / max(grid_w - 1, 1), (height - 1) / max(grid_h - 1, 1)
    xs = np.arange(width, dtype=np.float32)
    grid_x = np.broadcast_to(xs / scale_x, (1, width))
    out = np.empty_like(array)
    for top in range(0, height, _CHUNK_ROWS):
        rows = np.arange(top, min(height, top + _CHUNK_ROWS), dtype=np.float32)
        gx = np.ascontiguousarray(np.broadcast_to(grid_x, (rows.size, width)))
        gy = np.ascontiguousarray(np.broadcast_to((rows / scale_y)[:, None], (rows.size, width)))
        local = cv2.remap(field, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        map_x = xs[None, :] + local[:, :, 0] * scale_x
        map_y = rows[:, None] + local[:, :, 1] * scale_y
        out[top:top + rows.size] = cv2.remap(array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if is_array:
        return np.clip(out, 0.0, 1.0)
    result = Image.fromarray(out, mode=mode)
    result.info = dict(image.info)
    return result
