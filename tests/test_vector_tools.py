"""Gradient fills, SVG export and align/distribute (F27)."""
import base64
import io
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from app.layer_align import alignment_moves, distribution_moves, move_layer, movable
from app.svg_export import layers_to_svg, path_d
from app.vector_shapes import render_path_layer, render_shape_layer

SVG = "{http://www.w3.org/2000/svg}"


def _shape(**extra):
    return {"kind": "shape", "name": "Box", "shape_type": "Rectangle", "x_pct": 50.0, "y_pct": 50.0, "width_pct": 40.0, "height_pct": 40.0,
            "rotation_deg": 0.0, "fill_enabled": True, "fill_color": "#FF0000", "fill_opacity": 1.0, "stroke_enabled": False,
            "opacity": 1.0, "blend_mode": "normal", "visible": True, **extra}


# --- gradient fills ---------------------------------------------------------------------------------

def test_linear_and_radial_gradient_fills():
    base = {"shape_type": "Rectangle", "x_pct": 50, "y_pct": 50, "width_pct": 80, "height_pct": 80,
            "fill_color": (255, 0, 0), "fill_color2": (0, 0, 255), "fill_opacity": 1.0}
    linear = np.asarray(render_shape_layer((200, 100), {**base, "fill_style": "Linear gradient", "gradient_angle": 0}))
    assert linear[50, 25, 0] > 230 and linear[50, 174, 2] > 230  # red on the left, blue on the right
    radial = np.asarray(render_shape_layer((200, 100), {**base, "fill_style": "Radial gradient"}))
    assert radial[50, 100, 0] > 230 and radial[12, 22, 2] > radial[50, 100, 2]  # red centre, bluer corners
    assert radial[2, 2, 3] == 0  # still only inside the shape
    outlined = np.asarray(render_shape_layer((200, 100), {**base, "fill_style": "Linear gradient", "stroke_enabled": True,
                                                          "stroke_color": (0, 255, 0), "stroke_width": 3}))
    assert outlined[10, 100, 1] > 200  # the outline is drawn over the gradient
    path = {"points": [{"x_pct": 10, "y_pct": 10}, {"x_pct": 90, "y_pct": 10}, {"x_pct": 50, "y_pct": 90}], "closed": True,
            "fill_enabled": True, "fill_style": "Linear gradient", "fill_color": (255, 255, 0), "fill_color2": (0, 0, 0),
            "gradient_angle": 90, "stroke_enabled": False}
    filled = np.asarray(render_path_layer((100, 100), path))
    assert filled[80, 50, 0] > filled[15, 50, 0]  # 90 degrees: the start colour at the bottom


# --- SVG ------------------------------------------------------------------------------------------

def test_svg_has_one_element_per_layer_with_crop_as_viewbox():
    star = _shape(name="Star", shape_type="Star", fill_style="Radial gradient", fill_color2="#0000FF", rotation_deg=15)
    text = {"kind": "text", "name": "Title", "text": "Hello <world>\nsecond", "font_size_pct": 10, "color": "#FFFFFF",
            "align": "center", "x_pct": 50, "y_pct": 20, "font": "Arial", "opacity": 0.5, "blend_mode": "screen", "visible": True}
    path = {"kind": "path", "name": "Curve", "points": [{"x_pct": 10, "y_pct": 80, "handle_dx_pct": 5, "handle_dy_pct": 0},
                                                       {"x_pct": 90, "y_pct": 80}], "closed": False, "stroke_enabled": True,
            "stroke_color": "#00FF00", "stroke_width": 4, "opacity": 1.0, "blend_mode": "normal", "visible": False}
    adjustment = {"kind": "adjustment", "name": "Warm", "adjustments": {}, "opacity": 1.0, "blend_mode": "normal", "visible": True}
    svg = layers_to_svg([_shape(), star, text, path, adjustment], (400, 300), crop=(50, 0, 350, 300), background=Image.new("RGB", (300, 300)))
    root = ET.fromstring(svg)
    assert root.get("viewBox") == "50 0 300 300" and root.get("width") == "300"
    groups = root.findall(f"{SVG}g")
    assert len(groups) == 4  # the adjustment layer has no SVG counterpart
    assert groups[0].find(f"{SVG}rect").get("fill") == "#FF0000"
    assert groups[1].find(f"{SVG}polygon").get("fill").startswith("url(#grad")
    assert root.find(f"{SVG}defs/{SVG}radialGradient") is not None
    assert groups[2].get("opacity") == "0.5" and "mix-blend-mode:screen" in groups[2].get("style")
    assert [span.text for span in groups[2].iter(f"{SVG}tspan")] == ["Hello <world>", "second"]
    assert groups[3].get("display") == "none" and groups[3].find(f"{SVG}path").get("d").startswith("M 40 240 C")
    assert root.find(f"{SVG}image").get("x") == "50"  # the photo sits in the crop window


def test_path_data_matches_the_raster_curve_handles():
    d = path_d([{"x_pct": 0, "y_pct": 50, "handle_dx_pct": 10, "handle_dy_pct": 0}, {"x_pct": 100, "y_pct": 50}], 200, 100, closed=True)
    assert d == "M 0 50 C 20 50 200 50 200 50 C 200 50 -20 50 0 50 Z"


def test_placed_image_layers_are_embedded():
    pixels = Image.new("RGBA", (40, 20), (255, 0, 0, 255))
    buffer = io.BytesIO()
    pixels.save(buffer, format="PNG")
    layer = {"kind": "image", "name": "Logo", "image_b64": base64.b64encode(buffer.getvalue()).decode(), "x_pct": 25.0, "y_pct": 50.0,
             "scale_pct": 50.0, "rotation_deg": 0.0, "stretch": False, "opacity": 1.0, "blend_mode": "normal", "visible": True}
    image = ET.fromstring(layers_to_svg([layer], (200, 100))).find(f"{SVG}g/{SVG}image")
    assert (image.get("width"), image.get("height"), image.get("x")) == ("100", "50", "0")


# --- align / distribute ---------------------------------------------------------------------------

def test_alignment_and_distribution_moves():
    bounds = {"a": (10, 10, 20, 20), "b": (40, 30, 60, 50), "c": (80, 5, 90, 15)}
    assert alignment_moves(bounds, "left") == {"a": (0, 0), "b": (-30, 0), "c": (-70, 0)}
    assert alignment_moves(bounds, "center", (0, 0, 100, 100))["b"] == (0, 0)
    assert alignment_moves(bounds, "bottom")["c"] == (0, 35)
    moves = distribution_moves(bounds, "horizontal")
    assert moves["a"] == (0, 0) and moves["c"] == (0, 0)
    new_b = (40 + moves["b"][0], 60 + moves["b"][0])
    assert pytest.approx(new_b[0] - 20) == pytest.approx(80 - new_b[1])  # equal gaps
    with pytest.raises(ValueError):
        alignment_moves(bounds, "diagonal")


def test_move_layer_and_what_can_move():
    shape = move_layer(_shape(), 10, -60)
    assert (shape["x_pct"], shape["y_pct"]) == (60.0, 0.0)  # clamped to the frame
    path = move_layer({"kind": "path", "points": [{"x_pct": 10, "y_pct": 10}, {"x_pct": 20, "y_pct": 30}]}, 5, 5)
    assert path["points"][1] == {"x_pct": 25.0, "y_pct": 35.0}
    assert movable(_shape()) and not movable({"kind": "adjustment"}) and not movable({"kind": "image", "stretch": True})


@pytest.mark.slow
def test_align_buttons_move_layers_and_their_sliders(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import backend_client
    from tests.test_frontend_unified_app import APP_PATH, _fake_process_via_backend, image_bytes, open_tab

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("photo.png", image_bytes(size=(200, 100)), "image/png"))
    at.run(timeout=30)
    open_tab(at, "◈ Layers")
    at.session_state["image_layers"] = [{**_shape(), "id": "s1", "x_pct": 70.0, "width_pct": 20.0}]
    at.run(timeout=30)
    at.multiselect(key="l-align-sel").set_value(["s1"])
    at.run(timeout=30)
    at.button(key="l-align-left").click()
    at.run(timeout=30)
    assert not at.exception
    layer = at.session_state["image_layers"][0]
    assert layer["x_pct"] == pytest.approx(10.0, abs=1.0)  # its left edge on the photo's left edge
    assert at.slider(key="l-shp-x-s1").value == pytest.approx(layer["x_pct"])
    json.dumps(layer)
