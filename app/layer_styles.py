from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

# Photoshop applies a layer's effects in a fixed back-to-front stacking order relative to the
# layer's own (possibly Color-Overlaid) content: Drop Shadow furthest back, then Outer Glow, then
# the content itself, then Stroke drawn on top of the content's edge. Each effect's own shape is
# computed from the layer's ORIGINAL alpha silhouette independently, not from whatever another
# effect already produced -- otherwise, e.g., a wide Stroke's dilated edge would feed into Outer
# Glow, making the glow's shape depend on the stroke width instead of the layer's real silhouette.


def _colorize_silhouette(alpha: Image.Image, color: tuple[int, int, int], opacity: float) -> Image.Image:
    """A solid `color` everywhere `alpha` is nonzero, itself scaled by `opacity` -- the shared base
    shape Drop Shadow/Outer Glow/Stroke/Color Overlay each start from."""
    scaled_alpha = alpha.point(lambda value: int(value * max(0.0, min(1.0, opacity))))
    solid = Image.new("RGBA", alpha.size, (*color, 0))
    solid.putalpha(scaled_alpha)
    return solid


def _dilated_alpha(alpha: Image.Image, width: float) -> Image.Image:
    width = max(1, int(width))
    kernel = np.ones((width * 2 + 1, width * 2 + 1), np.uint8)
    dilated = cv2.dilate(np.asarray(alpha), kernel)
    return Image.fromarray(dilated, mode="L")


def _drop_shadow_layer(alpha: Image.Image, offset_x: float, offset_y: float, blur: float, color: tuple[int, int, int], opacity: float) -> Image.Image:
    shape = _colorize_silhouette(alpha, color, opacity)
    layer = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    layer.paste(shape, (int(offset_x), int(offset_y)), shape)
    return layer.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else layer


def _outer_glow_layer(alpha: Image.Image, blur: float, color: tuple[int, int, int], opacity: float) -> Image.Image:
    shape = _colorize_silhouette(alpha, color, opacity)
    return shape.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else shape


def _stroke_layer(alpha: Image.Image, width: float, color: tuple[int, int, int], opacity: float) -> Image.Image:
    return _colorize_silhouette(_dilated_alpha(alpha, width), color, opacity)


def apply_color_overlay(image: Image.Image, color: tuple[int, int, int] = (255, 0, 0), opacity: float = 0.6) -> Image.Image:
    """Recolors the layer's own visible pixels -- at opacity 1.0, every non-transparent pixel
    becomes flat `color`; below that, it's a tint blended with the original content."""
    image = image.convert("RGBA")
    return Image.alpha_composite(image, _colorize_silhouette(image.split()[3], color, opacity))


def apply_stroke(image: Image.Image, width: float = 4, color: tuple[int, int, int] = (255, 255, 255), opacity: float = 1.0) -> Image.Image:
    """An outline traced around the layer's own silhouette, drawn just outside its edge (an
    "Outside" stroke position, the one that doesn't hide any of the layer's own content)."""
    image = image.convert("RGBA")
    return Image.alpha_composite(_stroke_layer(image.split()[3], width, color, opacity), image)


def apply_outer_glow(image: Image.Image, blur: float = 10, color: tuple[int, int, int] = (255, 220, 120), opacity: float = 0.75) -> Image.Image:
    """A soft halo radiating outward from the layer's silhouette -- a blurred, recolored copy of
    its own alpha shape, sitting behind the layer's real content."""
    image = image.convert("RGBA")
    return Image.alpha_composite(_outer_glow_layer(image.split()[3], blur, color, opacity), image)


def apply_drop_shadow(
    image: Image.Image, offset_x: float = 8, offset_y: float = 8, blur: float = 6,
    color: tuple[int, int, int] = (0, 0, 0), opacity: float = 0.6,
) -> Image.Image:
    """A blurred, offset, recolored copy of the layer's own silhouette, sitting behind it -- drawn
    within the same canvas-sized image the layer already occupies, so an offset/blur large enough
    to push the shadow past the canvas edge is simply clipped, the same as Photoshop clips a shadow
    that falls outside a layer's own bounding canvas."""
    image = image.convert("RGBA")
    return Image.alpha_composite(_drop_shadow_layer(image.split()[3], offset_x, offset_y, blur, color, opacity), image)


def _clip_to(layer: Image.Image, alpha: Image.Image) -> Image.Image:
    """`layer` limited to the layer's own visible pixels (inner effects never spill outside)."""
    clipped = np.asarray(layer).copy()
    clipped[..., 3] = (clipped[..., 3].astype(np.uint16) * np.asarray(alpha, dtype=np.uint16) // 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGBA")


def _inner_shadow_layer(alpha: Image.Image, offset_x: float, offset_y: float, blur: float,
                        color: tuple[int, int, int], opacity: float) -> Image.Image:
    """A shadow cast inward from the silhouette's edge (F28): the area OUTSIDE the shape, offset and
    blurred, seen only inside the shape -- as if the layer were cut out of the surface."""
    outside = Image.fromarray(255 - np.asarray(alpha), mode="L")
    shifted = Image.new("L", alpha.size, 255)  # beyond the canvas counts as outside too
    shifted.paste(outside, (int(offset_x), int(offset_y)))
    if blur > 0:
        shifted = shifted.filter(ImageFilter.GaussianBlur(blur))
    return _clip_to(_colorize_silhouette(shifted, color, opacity), alpha)


def _gradient_overlay_layer(alpha: Image.Image, color_start: tuple[int, int, int], color_end: tuple[int, int, int],
                            angle: float, opacity: float) -> Image.Image:
    """A linear gradient across the layer's own bounds at `angle` degrees (0 = left to right,
    90 = bottom to top, as in Photoshop), clipped to its pixels (F28)."""
    width, height = alpha.size
    box = alpha.getbbox() or (0, 0, width, height)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radians = np.radians(angle)
    dx, dy = np.cos(radians), -np.sin(radians)
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    half = max(1.0, (abs(dx) * (box[2] - box[0]) + abs(dy) * (box[3] - box[1])) / 2.0)
    position = np.clip(((xx - cx) * dx + (yy - cy) * dy) / (2 * half) + 0.5, 0.0, 1.0)[..., None]
    rgb = np.asarray(color_start, np.float32) * (1 - position) + np.asarray(color_end, np.float32) * position
    a = np.asarray(alpha, np.float32) * max(0.0, min(1.0, opacity))
    return Image.fromarray(np.dstack([rgb, a]).round().astype(np.uint8), mode="RGBA")


def _bevel_layers(alpha: Image.Image, size: float, angle: float, depth: float,
                  highlight: tuple[int, int, int], shadow: tuple[int, int, int]) -> tuple[Image.Image, Image.Image]:
    """Inner Bevel & Emboss (F28): the silhouette's softened alpha used as a height map, lit from
    `angle` degrees (Photoshop's light direction, 30 deg altitude): faces turned toward the light get
    `highlight`, the others `shadow`, both scaled by `depth` and limited to the layer's pixels.
    Returns (highlight layer, shadow layer)."""
    size = max(1.0, float(size))
    sigma = size / 2.0
    height = cv2.GaussianBlur(np.asarray(alpha, np.float32) / 255.0, (0, 0), sigma)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    radians = np.radians(angle)
    lx, ly = np.cos(radians), -np.sin(radians)  # image y points down
    # Slope toward the light, normalised by the steepest slope the blurred edge has (a Gaussian
    # step's peak gradient, 1 / (sigma * sqrt(2 pi)), times Sobel's gain of 8), so 1 = facing the light.
    lit = (-(gx * lx + gy * ly)) * sigma * np.sqrt(2 * np.pi) / 8.0
    strength = max(0.0, min(1.0, depth))
    inside = np.asarray(alpha, np.float32) / 255.0
    up = np.clip(lit, 0.0, 1.0) * strength * inside * 255.0
    down = np.clip(-lit, 0.0, 1.0) * strength * inside * 255.0

    def tinted(color, amount):
        layer = np.zeros((*amount.shape, 4), np.uint8)
        layer[..., :3] = color
        layer[..., 3] = amount.round().astype(np.uint8)
        return Image.fromarray(layer, mode="RGBA")

    return tinted(highlight, up), tinted(shadow, down)


def apply_layer_styles(image: Image.Image, styles: dict) -> Image.Image:
    """Apply any combination of the four styles above, in Photoshop's own back-to-front order, to
    one layer's rendered RGBA content -- called before that content becomes a WorkspaceLayer, so the
    normal opacity/blend-mode compositing in app.layer_workspace.LayerWorkspace runs unmodified on
    top of whatever this produces. `styles` uses one `{effect}_enabled` flag per effect (Color
    Overlay/Stroke/Outer Glow/Drop Shadow) plus that effect's own parameters; a missing or falsy
    `_enabled` flag leaves that effect off, and an empty/missing `styles` dict is a total no-op.

    Every effect's shape is computed from the ORIGINAL alpha (captured once, up front) rather than
    chained through the previous effect's output -- see the module docstring for why that matters
    for Stroke in particular."""
    base = image.convert("RGBA")
    original_alpha = base.split()[3]
    content = base
    # Photoshop's order, back to front: Drop Shadow, Outer Glow, (Stroke, outside), the content,
    # Gradient Overlay, Color Overlay, Inner Shadow, Bevel & Emboss.
    if styles.get("gradient_overlay_enabled"):
        content = Image.alpha_composite(content, _gradient_overlay_layer(
            original_alpha, tuple(styles.get("gradient_overlay_start", (0, 0, 0))), tuple(styles.get("gradient_overlay_end", (255, 255, 255))),
            styles.get("gradient_overlay_angle", 90.0), styles.get("gradient_overlay_opacity", 1.0)))
    if styles.get("color_overlay_enabled"):
        content = apply_color_overlay(content, tuple(styles.get("color_overlay_color", (255, 0, 0))), styles.get("color_overlay_opacity", 0.6))
    if styles.get("inner_shadow_enabled"):
        content = Image.alpha_composite(content, _inner_shadow_layer(
            original_alpha, styles.get("inner_shadow_offset_x", 5), styles.get("inner_shadow_offset_y", 5),
            styles.get("inner_shadow_blur", 5), tuple(styles.get("inner_shadow_color", (0, 0, 0))), styles.get("inner_shadow_opacity", 0.75)))
    if styles.get("bevel_enabled"):
        light, dark = _bevel_layers(
            original_alpha, styles.get("bevel_size", 6), styles.get("bevel_angle", 120.0), styles.get("bevel_depth", 0.75),
            tuple(styles.get("bevel_highlight", (255, 255, 255))), tuple(styles.get("bevel_shadow", (0, 0, 0))))
        content = Image.alpha_composite(Image.alpha_composite(content, dark), light)

    stack = []
    if styles.get("drop_shadow_enabled"):
        stack.append(_drop_shadow_layer(
            original_alpha, styles.get("drop_shadow_offset_x", 8), styles.get("drop_shadow_offset_y", 8),
            styles.get("drop_shadow_blur", 6), tuple(styles.get("drop_shadow_color", (0, 0, 0))), styles.get("drop_shadow_opacity", 0.6),
        ))
    if styles.get("outer_glow_enabled"):
        stack.append(_outer_glow_layer(original_alpha, styles.get("outer_glow_blur", 10), tuple(styles.get("outer_glow_color", (255, 220, 120))), styles.get("outer_glow_opacity", 0.75)))
    if styles.get("stroke_enabled"):
        stack.append(_stroke_layer(original_alpha, styles.get("stroke_width", 4), tuple(styles.get("stroke_color", (255, 255, 255))), styles.get("stroke_opacity", 1.0)))
    stack.append(content)

    result = stack[0]
    for layer in stack[1:]:
        result = Image.alpha_composite(result, layer)
    return result
