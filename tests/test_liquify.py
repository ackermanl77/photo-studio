"""Liquify (F23): the parametric warp (app.liquify), its option/recipe validation, the pipeline's
Geometry step, the editor's JS and the Geometry tab."""
import json

import numpy as np
import pytest
from PIL import Image

from app import high_bit
from app.edit_schema import OptionsError, parse_options, sanitize_recipe
from app.geometry import apply_geometry
from app.liquify import MAX_STROKES, apply_liquify, clean_strokes, displacement_field
from app.pipeline import run_pipeline


def _dot_image(width=300, height=200, at=(150, 70), radius=4) -> Image.Image:
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    array[(xx - at[0]) ** 2 + (yy - at[1]) ** 2 <= radius ** 2] = 0
    return Image.fromarray(array)


def _dot_centre(image) -> tuple[float, float]:
    array = np.asarray(image.convert("L")) if isinstance(image, Image.Image) else high_bit.quantize8(image)[:, :, 0]
    ys, xs = np.nonzero(array < 128)
    return float(xs.mean()), float(ys.mean())


def test_push_drags_pixels_along_the_stroke():
    image = _dot_image(at=(100, 100))
    stroke = {"tool": "push", "size": 0.25, "strength": 1.0, "points": [[100 / 299, 0.5], [140 / 299, 0.5]]}
    x, y = _dot_centre(apply_liquify(image, [stroke]))
    assert x > 120 and abs(y - 100) < 2


def test_bloat_pinch_and_twirl_move_pixels_the_right_way():
    centre = [0.5, 0.5]
    image = _dot_image(at=(150, 70))  # above the brush centre (149.5, 99.5)

    def moved(tool):
        return _dot_centre(apply_liquify(image, [{"tool": tool, "size": 0.4, "strength": 1.0, "points": [centre]}]))

    assert moved("bloat")[1] < 69  # pushed away from the centre (up)
    assert moved("pinch")[1] > 71  # pulled toward it
    assert moved("twirl_cw")[0] > 151  # above the centre, clockwise goes right
    assert moved("twirl_ccw")[0] < 149


def test_no_strokes_leaves_the_image_alone_and_strokes_compose():
    image = _dot_image()
    assert apply_liquify(image, []) is image
    assert displacement_field([], 100, 100) is None
    one = {"tool": "push", "size": 0.2, "strength": 1.0, "points": [[0.4, 0.35], [0.5, 0.35]]}
    twice = _dot_centre(apply_liquify(image, [one, one]))[0]
    once = _dot_centre(apply_liquify(image, [one]))[0]
    assert twice > once > 150


def test_the_warp_is_resolution_independent():
    strokes = [{"tool": "push", "size": 0.2, "strength": 0.8, "points": [[0.45, 0.35], [0.6, 0.4]]}]
    small = _dot_centre(apply_liquify(_dot_image(300, 200, (150, 70), 4), strokes))
    large = _dot_centre(apply_liquify(_dot_image(1200, 800, (600, 280), 16), strokes))
    assert large[0] / 4 == pytest.approx(small[0], abs=1.5) and large[1] / 4 == pytest.approx(small[1], abs=1.5)


def test_float_arrays_are_warped_like_8_bit_images():
    image = _dot_image()
    array = high_bit.from_image(image)
    strokes = [{"tool": "bloat", "size": 0.3, "strength": 1.0, "points": [[0.5, 0.4]]}]
    warped = apply_liquify(array, strokes)
    assert warped.dtype == np.float32 and 0.0 <= warped.min() and warped.max() <= 1.0
    assert _dot_centre(warped) == pytest.approx(_dot_centre(apply_liquify(image, strokes)), abs=0.5)


def test_validation_of_options_and_recipes():
    stroke = {"tool": "pinch", "size": 0.9, "strength": 0.0, "points": [[1.5, -1.0]]}
    assert clean_strokes([stroke, {"tool": "smudge"}, "x"]) == [{"tool": "pinch", "size": 0.5, "strength": 0.05, "points": [[1.0, 0.0]]}]
    assert parse_options({"liquify": [stroke]})["liquify"][0]["size"] == 0.5
    assert parse_options({"liquify": []})["liquify"] is None
    with pytest.raises(OptionsError):
        parse_options({"liquify": [{"tool": "smudge", "size": 0.1, "strength": 1, "points": [[0, 0]]}]})
    with pytest.raises(OptionsError):
        parse_options({"liquify": [stroke] * (MAX_STROKES + 1)})
    assert sanitize_recipe({"liquify_strokes": [stroke, 3]})["liquify_strokes"] == clean_strokes([stroke])


def test_the_pipeline_applies_liquify_in_the_geometry_step():
    image = _dot_image(at=(100, 100))
    strokes = [{"tool": "push", "size": 0.25, "strength": 1.0, "points": [[100 / 299, 0.5], [140 / 299, 0.5]]}]
    via_geometry = apply_geometry(image, {"liquify": strokes})
    result = run_pipeline(image.copy(), parse_options({"liquify": strokes}))
    assert _dot_centre(result) == pytest.approx(_dot_centre(via_geometry), abs=0.5)
    # Drawn on the rotated frame: the stroke's coordinates follow the oriented photo.
    rotated = apply_geometry(image, {"rotate": 90, "liquify": strokes})
    assert rotated.size == (200, 300)


def test_liquify_editor_commits_the_dragged_points(tmp_path):
    from tests.test_interaction_improvements import NODE, _background, _run_editor

    if NODE is None:
        pytest.skip("Node.js is not installed")
    from app.ui_editors import _liquify_editor_html

    html, _height = _liquify_editor_html(_background(), "liquify-editor", 0.1, 4)
    assert 'role="application"' in html and ".st-key-liquify-editor-hidden" in html
    scenario = """
      const canvas = document.getElementById('liq-canvas'), wrap = document.getElementById('liq-wrap');
      canvas.dispatch('pointerdown', {clientX: 40, clientY: 30});
      for (let x = 50; x <= 200; x += 10) canvas.dispatch('pointermove', {clientX: x, clientY: 30});
      canvas.dispatch('pointerup', {});
      key(wrap, 'ArrowRight', {shiftKey: true});
      key(wrap, ' ');
    """
    out = _run_editor(tmp_path, html, scenario, "document.getElementById('liq-wrap').appendChild(document.getElementById('liq-canvas'));")
    stroke, stamp = out["commits"]
    assert len(stroke["points"]) == 4  # thinned to maxPoints
    assert stroke["points"][0] == [0.1, 0.1] and stroke["points"][-1] == [0.5, 0.1]
    assert stamp["points"][0][0] == pytest.approx(0.5 + 10 / 400)  # the cursor moved 10 px from its last spot


@pytest.mark.slow
def test_the_geometry_tab_records_a_stroke_into_the_recipe(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import backend_client
    from tests.test_frontend_unified_app import APP_PATH, _fake_process_via_backend, image_bytes, open_tab

    sent = []

    def recording_backend(image_bytes_, filename, options, *args, **kwargs):
        sent.append(dict(options))
        return _fake_process_via_backend(image_bytes_, filename, options, *args, **kwargs)

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", recording_backend)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("photo.png", image_bytes(size=(60, 40)), "image/png"))
    at.run(timeout=30)
    open_tab(at, "⛶ Geometry")
    at.selectbox(key="liq-tool").set_value("bloat")
    at.run(timeout=30)
    at.text_area(key="liquify-editor-data").set_value(json.dumps({"points": [[0.5, 0.5]]}))
    at.run(timeout=30)
    assert not at.exception
    strokes = at.session_state["liquify_strokes"]
    assert strokes == [{"tool": "bloat", "size": 0.1, "strength": 0.5, "points": [[0.5, 0.5]]}]
    assert any(options.get("liquify") == strokes for options in sent)
    at.button(key="liq-undo-btn").click()
    at.run(timeout=30)
    assert at.session_state["liquify_strokes"] == []
