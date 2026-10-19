"""Align and distribute layers (F27), Streamlit-free.

Works on each layer's bounds as percentages of the frame, ``(x0, y0, x1, y1)`` -- measured from the
layer as it renders (app.ui.layers.layer_bounds), so text, rotated shapes, stars and Pen paths all
align by what you see. Alignment moves every chosen layer's left/centre/right (or top/middle/bottom)
edge onto the target's: the photo, or the chosen layers' combined bounds. Distribution keeps the
two outermost layers and spaces the ones between so the gaps are equal (3 or more layers).
`move_layer` then shifts a layer dict: a Shape/Text/placed Image by its position, a Pen path by all
of its points.
"""
from __future__ import annotations

Bounds = tuple[float, float, float, float]
ALIGN_MODES = ("left", "center", "right", "top", "middle", "bottom")
MOVABLE_KINDS = ("text", "shape", "path", "image")


def movable(layer: dict) -> bool:
    if layer.get("kind", "image") not in MOVABLE_KINDS:
        return False
    return not (layer.get("kind", "image") == "image" and layer.get("stretch", "x_pct" not in layer))


def union(bounds: dict[str, Bounds]) -> Bounds:
    values = list(bounds.values())
    return min(b[0] for b in values), min(b[1] for b in values), max(b[2] for b in values), max(b[3] for b in values)


def alignment_moves(bounds: dict[str, Bounds], mode: str, target: Bounds | None = None) -> dict[str, tuple[float, float]]:
    """{layer id: (dx, dy)} in frame percent. `target` None = the layers' own combined bounds."""
    if mode not in ALIGN_MODES:
        raise ValueError(f"Unknown alignment {mode!r}")
    if not bounds:
        return {}
    tx0, ty0, tx1, ty1 = target or union(bounds)
    moves = {}
    for key, (x0, y0, x1, y1) in bounds.items():
        dx = dy = 0.0
        if mode == "left":
            dx = tx0 - x0
        elif mode == "center":
            dx = (tx0 + tx1) / 2 - (x0 + x1) / 2
        elif mode == "right":
            dx = tx1 - x1
        elif mode == "top":
            dy = ty0 - y0
        elif mode == "middle":
            dy = (ty0 + ty1) / 2 - (y0 + y1) / 2
        else:
            dy = ty1 - y1
        moves[key] = (dx, dy)
    return moves


def distribution_moves(bounds: dict[str, Bounds], axis: str) -> dict[str, tuple[float, float]]:
    """Equal gaps between the layers along `axis` ("horizontal" or "vertical"), outermost two fixed."""
    if len(bounds) < 3:
        return {key: (0.0, 0.0) for key in bounds}
    lo, hi = (0, 2) if axis == "horizontal" else (1, 3)
    ordered = sorted(bounds, key=lambda key: (bounds[key][lo] + bounds[key][hi]) / 2)
    sizes = [bounds[key][hi] - bounds[key][lo] for key in ordered]
    start, end = bounds[ordered[0]][lo], bounds[ordered[-1]][hi]
    gap = (end - start - sum(sizes)) / (len(ordered) - 1)
    moves, position = {}, start
    for key, size in zip(ordered, sizes):
        delta = position - bounds[key][lo]
        moves[key] = (delta, 0.0) if axis == "horizontal" else (0.0, delta)
        position += size + gap
    return moves


def move_layer(layer: dict, dx: float, dy: float) -> dict:
    """Shift a layer dict by (dx, dy) frame percent, in place; returns it."""
    if layer.get("kind") == "path":
        for point in layer.get("points", []):
            point["x_pct"] = float(point["x_pct"]) + dx
            point["y_pct"] = float(point["y_pct"]) + dy
    else:
        layer["x_pct"] = max(0.0, min(100.0, float(layer.get("x_pct", 50.0)) + dx))
        layer["y_pct"] = max(0.0, min(100.0, float(layer.get("y_pct", 50.0)) + dy))
    return layer
