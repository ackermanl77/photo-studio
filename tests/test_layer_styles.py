import numpy as np
from PIL import Image

from app.layer_styles import apply_color_overlay, apply_drop_shadow, apply_layer_styles, apply_outer_glow, apply_stroke


def _square_layer(size=(60, 60), square_box=(20, 20, 40, 40), color=(0, 200, 0, 255)) -> Image.Image:
    """A small opaque square on an otherwise fully transparent canvas -- the standard probe shape
    for every test below, since its silhouette's edges/interior are trivial to reason about."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    x0, y0, x1, y1 = square_box
    pixels = np.asarray(image).copy()
    pixels[y0:y1, x0:x1] = color
    return Image.fromarray(pixels, mode="RGBA")


def test_apply_layer_styles_with_no_styles_is_a_no_op():
    layer = _square_layer()

    result = apply_layer_styles(layer, {})

    assert np.array_equal(np.asarray(result), np.asarray(layer))


def test_color_overlay_recolors_the_square_and_leaves_transparent_pixels_alone():
    layer = _square_layer()

    result = apply_color_overlay(layer, color=(255, 0, 0), opacity=1.0)
    pixels = np.asarray(result)

    assert tuple(pixels[30, 30]) == (255, 0, 0, 255)  # inside the square: fully recolored
    assert tuple(pixels[0, 0]) == (0, 0, 0, 0)  # outside it: still fully transparent


def test_stroke_adds_a_ring_outside_the_square_without_covering_its_interior():
    layer = _square_layer(color=(0, 200, 0, 255))

    result = apply_stroke(layer, width=4, color=(0, 0, 255), opacity=1.0)
    pixels = np.asarray(result)

    assert tuple(pixels[30, 30]) == (0, 200, 0, 255)  # interior: original green, untouched
    assert tuple(pixels[19, 30]) == (0, 0, 255, 255)  # just outside the square's top edge: blue stroke
    assert pixels[0, 0, 3] == 0  # far outside: still fully transparent (RGB there is meaningless)


def test_outer_glow_adds_color_outside_the_shape_and_leaves_the_interior_untouched():
    layer = _square_layer(color=(0, 200, 0, 255))

    result = apply_outer_glow(layer, blur=4, color=(255, 220, 120), opacity=1.0)
    pixels = np.asarray(result)

    assert tuple(pixels[30, 30]) == (0, 200, 0, 255)  # interior: original content on top, untouched
    assert pixels[19, 30, 3] > 0  # a real blur genuinely extends visible (non-transparent) glow past the edge
    assert pixels[19, 30, 0] > pixels[19, 30, 2]  # and it leans toward the warm glow color (more red than blue)


def test_drop_shadow_places_a_colored_silhouette_offset_behind_the_shape():
    layer = _square_layer(square_box=(20, 20, 40, 40), color=(0, 200, 0, 255))

    result = apply_drop_shadow(layer, offset_x=10, offset_y=10, blur=0, color=(0, 0, 0), opacity=1.0)
    pixels = np.asarray(result)

    assert tuple(pixels[30, 30]) == (0, 200, 0, 255)  # interior: the layer's own content on top
    assert tuple(pixels[45, 45]) == (0, 0, 0, 255)  # shadow visible where it's offset past the square's edge
    assert pixels[0, 0, 3] == 0  # far from both shape and shadow: still fully transparent


def test_apply_layer_styles_combines_color_overlay_and_stroke_in_the_right_layering():
    layer = _square_layer(color=(0, 200, 0, 255))

    result = apply_layer_styles(layer, {
        "color_overlay_enabled": True, "color_overlay_color": (255, 0, 0), "color_overlay_opacity": 1.0,
        "stroke_enabled": True, "stroke_width": 4, "stroke_color": (0, 0, 255), "stroke_opacity": 1.0,
    })
    pixels = np.asarray(result)

    assert tuple(pixels[30, 30]) == (255, 0, 0, 255)  # interior: recolored red by Color Overlay
    assert tuple(pixels[19, 30]) == (0, 0, 255, 255)  # ring: blue Stroke


def test_a_wide_stroke_does_not_change_outer_glows_shape():
    """Regression test for a real bug caught before shipping: an earlier version chained each
    effect through the previous one's output, so Outer Glow ended up haloing Stroke's already-
    dilated silhouette instead of the layer's real one -- a wide stroke would visibly push the glow
    further out. Both effects must independently start from the same original silhouette.

    Probed at a pixel neither stroke width reaches (so Stroke itself can't be what's covering it in
    either case), but that a real Outer Glow blur does -- if Glow's shape were built from Stroke's
    already-dilated silhouette instead of the original one, a wider stroke would push Glow's visible
    reach outward too, and this pixel would differ between the two runs."""
    layer = _square_layer(color=(0, 200, 0, 255))
    common_glow = {"outer_glow_enabled": True, "outer_glow_blur": 6, "outer_glow_color": (255, 220, 120), "outer_glow_opacity": 1.0}

    with_narrow_stroke = apply_layer_styles(layer, {"stroke_enabled": True, "stroke_width": 1, "stroke_color": (0, 0, 255), "stroke_opacity": 1.0, **common_glow})
    with_wide_stroke = apply_layer_styles(layer, {"stroke_enabled": True, "stroke_width": 3, "stroke_color": (0, 0, 255), "stroke_opacity": 1.0, **common_glow})

    probe = (12, 30)  # 8px outside the square's edge (y=20) -- past both stroke widths, within the glow's blur
    narrow_pixel = tuple(np.asarray(with_narrow_stroke)[probe])
    wide_pixel = tuple(np.asarray(with_wide_stroke)[probe])
    assert narrow_pixel[3] > 0  # sanity check: the probe is actually within the glow's visible reach
    assert narrow_pixel == wide_pixel
