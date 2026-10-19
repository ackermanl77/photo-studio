"""Inner Shadow, Bevel & Emboss and Gradient Overlay layer styles (F28)."""
import numpy as np
from PIL import Image, ImageDraw

from app.layer_styles import apply_layer_styles
from app.ui.layers import layer_dict_to_image, resolved_layer_styles, scaled_styles


def _square(size=(120, 100), box=(20, 20, 100, 80), color=(60, 120, 220, 255)):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(box, fill=color)
    return image


def test_inner_effects_stay_inside_the_silhouette():
    image = _square()
    for styles in ({"inner_shadow_enabled": True}, {"bevel_enabled": True, "bevel_size": 10},
                   {"gradient_overlay_enabled": True}):
        out = np.asarray(apply_layer_styles(image, styles))
        np.testing.assert_array_equal(out[..., 3], np.asarray(image)[..., 3])  # no spill, no holes


def test_inner_shadow_darkens_the_edge_toward_the_offset():
    out = np.asarray(apply_layer_styles(_square(), {"inner_shadow_enabled": True, "inner_shadow_offset_x": 8,
                                                     "inner_shadow_offset_y": 8, "inner_shadow_blur": 2})).astype(int)
    top_left, bottom_right, centre = out[22, 22, :3].sum(), out[78, 98, :3].sum(), out[50, 60, :3].sum()
    assert top_left < centre - 100  # the shadow falls inside the top-left edge
    assert abs(bottom_right - centre) < 10


def test_bevel_lights_the_side_facing_the_light():
    out = np.asarray(apply_layer_styles(_square(), {"bevel_enabled": True, "bevel_size": 10, "bevel_angle": 120,
                                                     "bevel_depth": 1.0})).astype(int)
    lit, shaded, flat = out[23, 60, :3].sum(), out[77, 60, :3].sum(), out[50, 60, :3].sum()
    assert lit > flat + 60 and shaded < flat - 60  # light from the upper left: top edge bright, bottom dark
    assert out[50, 60, :3].tolist() == [60, 120, 220]  # the flat middle is untouched


def test_gradient_overlay_runs_across_the_layer_bounds():
    styles = {"gradient_overlay_enabled": True, "gradient_overlay_start": (255, 0, 0), "gradient_overlay_end": (0, 0, 255),
              "gradient_overlay_angle": 0, "gradient_overlay_opacity": 1.0}
    out = np.asarray(apply_layer_styles(_square(), styles))
    assert out[50, 20, 0] > 240 and out[50, 20, 2] < 15  # left edge of the shape: the start colour
    assert out[50, 100, 2] > 240 and out[50, 100, 0] < 15  # right edge: the end colour
    up = np.asarray(apply_layer_styles(_square(), {**styles, "gradient_overlay_angle": 90}))
    assert up[80, 60, 0] > 240 and up[20, 60, 2] > 240  # 90 degrees: bottom to top


def test_ui_helpers_resolve_colors_and_scale_sizes():
    styles = {"inner_shadow_color": "#102030", "bevel_highlight": "#FFFFFF", "gradient_overlay_start": "#FF0000",
              "inner_shadow_blur": 10, "bevel_size": 8}
    resolved = resolved_layer_styles(styles)
    assert resolved["inner_shadow_color"] == (16, 32, 48) and resolved["gradient_overlay_start"] == (255, 0, 0)
    assert scaled_styles(styles, 0.5)["bevel_size"] == 4 and scaled_styles(styles, 0.5)["inner_shadow_blur"] == 5
    layer = {"kind": "shape", "shape_type": "Rectangle", "x_pct": 50.0, "y_pct": 50.0, "width_pct": 50.0, "height_pct": 50.0,
             "rotation_deg": 0.0, "fill_enabled": True, "fill_color": "#3366CC", "fill_opacity": 1.0, "stroke_enabled": False,
             "styles": {"bevel_enabled": True, "bevel_size": 6, "bevel_angle": 120, "bevel_depth": 1.0,
                        "bevel_highlight": "#FFFFFF", "bevel_shadow": "#000000"}}
    rendered = np.asarray(layer_dict_to_image(layer, (200, 100))).astype(int)
    plain = np.asarray(layer_dict_to_image({**layer, "styles": {}}, (200, 100))).astype(int)
    assert np.abs(rendered - plain).max() > 50  # the bevel reaches the rendered layer
