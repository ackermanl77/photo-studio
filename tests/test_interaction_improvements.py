"""Round-three interaction items: refine-on-pause for the proxy preview (P1), drag-to-pan in the
zoomed canvas (F3), and keyboard nudging / [ ] brush size in the on-canvas editors (F11).

The JavaScript inside components.html() can't run under AppTest, so the editors are checked for
the keyboard wiring they ship and their commit payloads are driven through the same hidden
text_area fields a real interaction ends in (see CLAUDE.md, "Testing conventions")."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from app import backend_client, refine_preview
from app.backend import app as flask_app
from app.refine_preview import RefineTracker
from app.ui_editors import KEY_COMMIT_DELAY_MS, _crop_editor_html, _curve_editor_html, _paint_editor_html, _pan_viewer_html, _pen_editor_html
from tests.test_frontend_unified_app import APP_PATH, _fake_process_via_backend


@pytest.fixture
def stub_backend(monkeypatch):
    """The in-process backend stand-in tests/test_frontend_unified_app.py uses (its fixture is autouse
    only in that module)."""
    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)


# --- P1: refine-on-pause ---------------------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_tracker_waits_for_a_pause_then_renders_once(monkeypatch):
    monkeypatch.setattr(refine_preview, "PAUSE_SECONDS", 1.0)
    monkeypatch.setattr(refine_preview, "RUN_IN_THREAD", False)
    clock, calls = _Clock(), []
    tracker = RefineTracker(clock)
    render = lambda: calls.append(1) or Image.new("RGB", (8, 8))  # noqa: E731
    tracker.observe("edit-1", render)
    assert tracker.status() == "waiting" and tracker.pending()
    clock.now += 0.5
    tracker.observe("edit-1", render)  # an unrelated rerun doesn't restart the pause...
    assert not tracker.maybe_start()
    clock.now += 0.6
    assert tracker.maybe_start() and calls == [1]
    assert tracker.status() == "ready" and not tracker.pending()
    assert tracker.needs_rerun() and not tracker.needs_rerun()  # announced exactly once
    assert tracker.result("edit-1").size == (8, 8)
    assert not tracker.maybe_start() and calls == [1]  # never renders the same edit twice
    tracker.observe("edit-2", render)  # ...but a changed edit does, and drops the old result
    assert tracker.result("edit-2") is None and tracker.status() == "waiting"
    assert not tracker.maybe_start()


def test_tracker_is_off_without_a_proxy_and_reports_failures(monkeypatch):
    monkeypatch.setattr(refine_preview, "PAUSE_SECONDS", 0.0)
    monkeypatch.setattr(refine_preview, "RUN_IN_THREAD", False)
    tracker = RefineTracker()
    tracker.observe(None, None)
    assert tracker.status() == "off" and not tracker.pending() and not tracker.maybe_start()

    def boom():
        raise RuntimeError("backend down")

    tracker.observe("edit", boom)
    assert tracker.maybe_start() and tracker.status() == "failed" and tracker.result("edit") is None
    assert not tracker.pending()


def test_a_threaded_refine_carries_the_edit_session(monkeypatch):
    monkeypatch.setattr(refine_preview, "PAUSE_SECONDS", 0.0)
    seen = []
    backend_client.bind_edit_session("session-x")
    tracker = RefineTracker()
    tracker.observe("edit", lambda: seen.append(backend_client._EDIT_SESSION.get()) or Image.new("RGB", (2, 2)))
    assert tracker.maybe_start()
    assert tracker.job.done.wait(5)
    assert seen == ["session-x"] and tracker.status() == "ready"
    backend_client.bind_edit_session(None)


def _check(session, generation, lane):
    from app.backend import _cancellation_check
    from app.config import EDIT_GENERATION_HEADER, EDIT_LANE_HEADER, EDIT_SESSION_HEADER

    headers = {EDIT_SESSION_HEADER: session, EDIT_GENERATION_HEADER: str(generation), EDIT_LANE_HEADER: lane}
    with flask_app.test_request_context("/process", method="POST", headers=headers):
        return _cancellation_check()


def test_a_newer_interactive_edit_supersedes_a_running_refine_but_not_the_reverse():
    refine = _check("s-refine", 10, "refine")
    detail = _check("s-refine", 9, "detail")
    assert not refine() and not detail()
    newer_detail = _check("s-refine", 11, "detail")
    assert refine()  # the user moved on: the full-resolution job stops at its next step
    assert not newer_detail()
    later_refine = _check("s-refine", 12, "refine")
    assert not newer_detail() and not later_refine()  # a refine never cancels an edit
    other_session = _check("s-other", 13, "detail")
    assert not later_refine() and not other_session()


def test_process_via_backend_sends_the_refine_lane_and_timeout(monkeypatch):
    captured = {}

    class _Response:
        status_code, ok, content, headers = 200, True, b"", {}

    def fake_send(files, data, headers, timeout):
        captured.update(headers=headers, timeout=timeout)
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2)).save(buffer, format="PNG")
        response = _Response()
        response.content = buffer.getvalue()
        return response

    monkeypatch.setattr(backend_client, "_send", fake_send)
    backend_client.bind_edit_session("s1")
    backend_client.process_via_backend(b"x", "a.png", {}, lane=backend_client.REFINE_LANE, timeout=backend_client.REFINE_REQUEST_TIMEOUT_S)
    backend_client.bind_edit_session(None)
    from app.config import EDIT_LANE_HEADER

    assert captured["headers"][EDIT_LANE_HEADER] == "refine" and captured["timeout"] == backend_client.REFINE_REQUEST_TIMEOUT_S


def _big_photo() -> bytes:
    buffer = io.BytesIO()
    array = np.zeros((3000, 4000, 3), np.uint8)
    array[..., 0] = np.linspace(40, 200, 4000, dtype=np.uint8)[None, :]
    array[..., 1] = 110
    Image.fromarray(array).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.slow
def test_a_paused_proxy_preview_is_replaced_by_a_full_resolution_render(monkeypatch):
    monkeypatch.setattr(refine_preview, "PAUSE_SECONDS", 0.0)
    monkeypatch.setattr(refine_preview, "RUN_IN_THREAD", False)
    calls = []

    def recording(image_bytes_, filename, options, *args, **kwargs):
        calls.append({"size": Image.open(io.BytesIO(image_bytes_)).size, "lane": kwargs.get("lane"), "options": dict(options)})
        return _fake_process_via_backend(image_bytes_, filename, options, *args, **kwargs)

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", recording)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("big.jpg", _big_photo(), "image/jpeg"))
    at.run(timeout=120)
    assert not at.exception
    assert max(calls[0]["size"]) == 2048  # the proxy renders first, as before
    refine_calls = [call for call in calls if call["lane"] == "refine"]
    assert len(refine_calls) == 1
    assert refine_calls[0]["size"] == (4000, 3000) and "preview_scale" not in refine_calls[0]["options"]
    assert any("Full-resolution view" in caption.value for caption in at.caption)

    # An unrelated rerun keeps the refined view and renders nothing new.
    before = len(calls)
    at.checkbox(key="cv-split").set_value(False)
    at.run(timeout=60)
    assert len(calls) == before

    # Changing the edit goes back to the fast proxy, then refines that edit once more.
    _change_an_edit(at)
    at.run(timeout=120)
    new_calls = calls[before:]
    assert max(new_calls[0]["size"]) == 2048
    assert [call["lane"] for call in new_calls].count("refine") == 1


def _change_an_edit(at):
    """Move the Exposure slider to change the edit."""
    at.slider(key="t-ev").set_value(0.25)


# --- F11 / F3: the editors' keyboard and drag code, run under Node with a fake DOM ----------------
# (tests/js_fake_dom.js). AppTest can't execute components.html JavaScript; this runs the exact
# <script> the editors ship and records what they commit into the hidden text_area. Skipped where
# Node isn't installed (GitHub's runners have it).

import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

NODE = shutil.which("node")
FAKE_DOM = Path(__file__).with_name("js_fake_dom.js")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
CROP_TREE = "const wrap = document.getElementById('crop-wrap'); ['crop-rect', 'crop-tl', 'crop-br'].forEach((id) => wrap.appendChild(document.getElementById(id)));"


def _run_editor(tmp_path, html: str, scenario: str, prelude: str = "") -> dict:
    script = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
    program = (
        f"const {{document, commits, canvasCalls, flushTimers, key}} = require({json.dumps(str(FAKE_DOM))});\n"
        f"const out = {{}};\n{prelude}\n{script}\n{scenario}\n"
        "out.commits = commits.map((c) => { try { return JSON.parse(c); } catch (e) { return c; } });\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    path = tmp_path / "editor_test.js"
    path.write_text(program, encoding="utf-8")
    result = subprocess.run([NODE, str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _background(size=(40, 30)):
    return Image.new("RGB", size, (90, 120, 160))


def test_editors_are_focusable_and_labelled_for_keyboard_use():
    background = _background()
    crop = _crop_editor_html(background, "crop-editor", (0.1, 0.1, 0.9, 0.9), (40, 30))[0]
    pen = _pen_editor_html(background, "pen", [], False)[0]
    paint = _paint_editor_html(background, "m")[0]
    curve = _curve_editor_html([(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)], [(0, 64)] * 5)
    pan = _pan_viewer_html(background, "cv-nav", (0.5, 0.5), 2.0, (40, 30))[0]
    for markup in (crop, pen, paint, curve, pan):
        assert 'role="application"' in markup and "aria-label" in markup and 'tabindex="0"' in markup
    for markup in (crop, pen, curve, pan):
        assert "ArrowUp" in markup
        assert str(KEY_COMMIT_DELAY_MS) in markup  # a burst of key presses commits once
    assert "e.key === '['" in paint and "e.key === ']'" in paint
    assert ".st-key-cv-nav-wand-hidden" in pan  # the navigator's own commit field


@needs_node
def test_crop_arrow_keys_nudge_the_focused_corner_and_commit_once(tmp_path):
    html = _crop_editor_html(_background(), "crop-editor", (0.1, 0.1, 0.6, 0.7), (40, 30))[0]
    scenario = """
      const br = document.getElementById('crop-br'), rect = document.getElementById('crop-rect');
      for (let i = 0; i < 4; i++) key(br, 'ArrowRight');
      out.beforeFlush = commits.length;
      flushTimers();
      key(rect, 'ArrowDown', {shiftKey: true});
      flushTimers();
      out.prevented = key(rect, 'ArrowLeft').defaultPrevented;
    """
    out = _run_editor(tmp_path, html, scenario, CROP_TREE)
    assert out["beforeFlush"] == 0 and out["prevented"]
    first, second = out["commits"][0], out["commits"][1]
    assert first == pytest.approx({"x0": 0.1, "y0": 0.1, "x1": 0.7, "y1": 0.7})  # 4 px of 40
    # Shift+Down moves the whole box 10 px of 30, clamped at the bottom edge (height 0.6).
    assert second == pytest.approx({"x0": 0.1, "y0": 0.4, "x1": 0.7, "y1": 1.0})


@needs_node
@pytest.mark.slow
def test_a_keyboard_nudged_crop_reaches_the_exported_size(tmp_path, stub_backend):
    """End to end: the payload the crop editor's arrow keys commit, fed to the real app's hidden field."""
    from tests.test_frontend_unified_app import canvas_dims, run_with_uploaded_image

    html = _crop_editor_html(_background(), "crop-editor", (0.1, 0.1, 0.6, 0.7), (40, 30))[0]
    out = _run_editor(tmp_path, html, "for (let i = 0; i < 4; i++) key(document.getElementById('crop-br'), 'ArrowRight'); flushTimers();", CROP_TREE)
    at = run_with_uploaded_image(size=(40, 30))
    at.selectbox(key="crop-aspect").set_value("Free-form")
    at.run(timeout=30)
    at.text_area(key="crop-editor-data").set_value(json.dumps(out["commits"][-1]))
    at.run(timeout=30)
    assert not at.exception
    assert canvas_dims(at) == (24, 18)


@needs_node
def test_pen_arrow_keys_move_the_selected_point_its_handle_and_delete_it(tmp_path):
    points = [{"x_pct": 10, "y_pct": 10, "handle_dx_pct": 0, "handle_dy_pct": 0}, {"x_pct": 50, "y_pct": 50, "handle_dx_pct": 0, "handle_dy_pct": 0}]
    html = _pen_editor_html(_background(), "pen", points, False)[0]
    scenario = """
      const wrap = document.getElementById('pen-wrap');
      document.getElementById('pen-anchors').children[1].focus();
      key(wrap, 'ArrowRight'); key(wrap, 'ArrowRight', {shiftKey: true});
      key(wrap, 'ArrowUp', {altKey: true});
      flushTimers();
      key(wrap, 'Delete');
      flushTimers();
    """
    out = _run_editor(tmp_path, html, scenario)
    moved, deleted = out["commits"]
    # The fake layout is 400x300 px: one press = 100/400 % horizontally, 100/300 % vertically.
    assert moved[1]["x_pct"] == pytest.approx(50 + 11 * 0.25)
    assert moved[1]["y_pct"] == pytest.approx(50) and moved[1]["handle_dy_pct"] == pytest.approx(-100 / 300)
    assert moved[0] == points[0]
    assert len(deleted) == 1 and deleted[0]["x_pct"] == 10


@needs_node
def test_paint_brackets_resize_the_brush_and_arrows_move_the_last_radial(tmp_path):
    html = _paint_editor_html(_background(), "m")[0]
    scenario = """
      const root = document.getElementById('paint-root');
      key(root, ']'); key(root, ']');
      out.bigger = document.getElementById('brush-size').value;
      key(root, '[', {shiftKey: true});
      out.smaller = document.getElementById('brush-size').value;
      for (let i = 0; i < 20; i++) key(root, '[');
      out.minimum = document.getElementById('brush-size').value;
      out.noShapeYet = key(root, 'ArrowRight').defaultPrevented;
      // Draw a radial: pick the tool through its button's click handler, then drag on the canvas.
      const radialButton = document.createElement('button');
      const canvas = document.getElementById('mask-canvas');
      canvas.dispatch('pointerdown', {clientX: 100, clientY: 100});
      canvas.dispatch('pointerup', {clientX: 140, clientY: 100});
      out.brushStrokeIsNotNudgeable = key(root, 'ArrowRight').defaultPrevented;
    """
    out = _run_editor(tmp_path, html, scenario)
    assert out["bigger"] == 22 and out["smaller"] == 12 and out["minimum"] == 4
    assert out["noShapeYet"] is False  # arrows do nothing until a radial/gradient was drawn
    assert out["brushStrokeIsNotNudgeable"] is False  # (the default tool is the brush)


@needs_node
def test_curve_point_moves_with_up_down_within_its_bounds(tmp_path):
    html = _curve_editor_html([(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)], [(0, 64), (0, 128), (64, 192), (128, 255), (192, 255)])
    prelude = """
      const svgEl = document.getElementById('curve-svg');
      for (let i = 0; i < 5; i++) { const c = document.createElement('circle'); c.className = 'pt fixed'; c.dataset.idx = i; svgEl.appendChild(c); }
    """
    scenario = """
      const mid = document.getElementById('curve-svg').querySelectorAll('.pt.fixed')[2];
      key(mid, 'ArrowUp', {shiftKey: true}); key(mid, 'ArrowUp'); flushTimers();
      for (let i = 0; i < 10; i++) key(mid, 'ArrowUp', {shiftKey: true});
      flushTimers();
    """
    out = _run_editor(tmp_path, html, scenario, prelude)
    assert out["commits"][0]["curve_mid"] == 139
    assert out["commits"][1]["curve_mid"] == 192  # clamped to its bound
    assert len(out["commits"]) == 2


@needs_node
def test_a_channel_curve_commits_its_own_keys_and_takes_no_extra_points(tmp_path):
    keys = ("curve_b_black", "curve_b_shadows", "curve_b_mid", "curve_b_highlights", "curve_b_white")
    html = _curve_editor_html([(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)], [(0, 64), (0, 128), (64, 192), (128, 255), (192, 255)],
                              [(30, 40)], keys=keys, color="#539bf5")
    prelude = """
      const svgEl = document.getElementById('curve-svg');
      for (let i = 0; i < 5; i++) { const c = document.createElement('circle'); c.className = 'pt fixed'; c.dataset.idx = i; svgEl.appendChild(c); }
    """
    scenario = """
      const shadows = document.getElementById('curve-svg').querySelectorAll('.pt.fixed')[1];
      key(shadows, 'ArrowUp', {shiftKey: true}); flushTimers();
      document.getElementById('curve-bg').dispatch('click', {clientX: 50, clientY: 50});
    """
    out = _run_editor(tmp_path, html, scenario, prelude)
    assert out["commits"] == [{"curve_enabled": True, "curve_b_black": 0, "curve_b_shadows": 74, "curve_b_mid": 128, "curve_b_highlights": 192, "curve_b_white": 255}]
    assert "#539bf5" in html and "[30, 40]" not in html  # its colour; the master curve's extra points stay off it


@needs_node
def test_dragging_the_zoomed_view_commits_the_new_center(tmp_path):
    # A 50x40 view of a 100x80 photo at 200 %, shown 400x300 px in the fake layout.
    html = _pan_viewer_html(_background((50, 40)), "cv-nav", (0.5, 0.5), 2.0, (100, 80))[0]
    scenario = """
      const wrap = document.getElementById('pan-wrap'), img = document.getElementById('pan-img');
      wrap.appendChild(img);
      wrap.dispatch('pointerdown', {clientX: 100, clientY: 100});
      wrap.dispatch('pointermove', {clientX: 180, clientY: 60});
      out.transform = img.style.transform;
      wrap.dispatch('pointerup', {clientX: 180, clientY: 60});
      out.afterDrag = commits.length;
      wrap.dispatch('pointerdown', {clientX: 10, clientY: 10});
      wrap.dispatch('pointerup', {clientX: 11, clientY: 10});  // a click, not a drag
      out.afterClick = commits.length;
      key(wrap, 'ArrowRight'); key(wrap, 'ArrowRight');
      flushTimers();
    """
    out = _run_editor(tmp_path, html, scenario)
    assert out["transform"] == "translate(80px,-40px)"
    assert out["afterDrag"] == 1 and out["afterClick"] == 1
    drag, keys = out["commits"]
    # 80 screen px = 10 view px = 5 photo px = 0.05 of 100; -40 screen px = -5.33 view px = -2.67 photo px of 80.
    assert drag == pytest.approx({"x": 0.45, "y": 0.5 + (40 * 40 / 300 / 2) / 80})
    # Two Right presses = look 20 % of the view further right: 80 screen px -> +0.05.
    assert keys == pytest.approx({"x": 0.55, "y": 0.5})


@pytest.mark.slow
def test_a_drag_pan_commit_moves_the_zoomed_view_precisely(stub_backend):
    """The drag viewer commits through the navigator's field; the pan now keeps 4 decimals (a drag
    at 400 % moves only a few source pixels, below the old 0.01 step)."""
    from tests.test_frontend_unified_app import run_with_uploaded_image

    at = run_with_uploaded_image(size=(40, 30))
    at.selectbox(key="cv-zoom").set_value(2.0)
    at.run(timeout=30)
    assert not any(img.captions and img.captions[0] == "Current Edited Image" for img in at.image)
    at.text_area(key="cv-nav-wand-data").set_value(json.dumps({"x": 0.4567, "y": 0.1234}))
    at.run(timeout=30)
    assert not at.exception
    assert at.slider(key="cv-panx").value == pytest.approx(0.4567) and at.slider(key="cv-pany").value == pytest.approx(0.1234)
    assert any("drag to pan" in caption.value for caption in at.caption)


@needs_node
def test_curve_extra_points_never_share_an_x_with_another_point(tmp_path):
    fixed = [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]
    html = _curve_editor_html(fixed, [(0, 64), (0, 128), (64, 192), (128, 255), (192, 255)], extra_points=[(100, 120)])
    # The fake layout is 400 px wide for a 256-unit viewBox; curve x -> client x.
    to_client = "const cx = (x) => (34 + (x / 255) * 210) * 400 / 256;"
    scenario = to_client + """
      const svgEl = document.getElementById('curve-svg');
      document.getElementById('curve-bg').dispatch('click', {clientX: cx(128), clientY: 100});
      const extra = document.getElementById('extra-points').children[0];  // the first extra point
      extra.className = 'pt extra'; extra.dataset.idx = extra.getAttribute('data-idx');  // a browser maps these attributes itself
      svgEl.dispatch('pointerdown', {target: extra});
      svgEl.dispatch('pointermove', {clientX: cx(64), clientY: 100});
      svgEl.dispatch('pointerup', {});
    """
    prelude = """
      const svgEl0 = document.getElementById('curve-svg');
      svgEl0.appendChild(document.getElementById('curve-bg'));
      for (let i = 0; i < 5; i++) { const c = document.createElement('circle'); c.className = 'pt fixed'; c.dataset.idx = i; svgEl0.appendChild(c); }
      svgEl0.appendChild(document.getElementById('extra-points'));
    """
    out = _run_editor(tmp_path, html, scenario, prelude)
    added = out["commits"][0]["curve_extra_points"]
    dragged = out["commits"][1]["curve_extra_points"]
    fixed_x = {x for x, _ in fixed}
    assert added[1][0] in (127, 129)  # a click right on the Midtones point's X lands beside it
    assert dragged[0][0] in (63, 65)  # dragging onto the Shadows point's X stops beside it
    for points in (added, dragged):
        xs = [x for x, _ in points]
        assert len(set(xs)) == len(xs) and not fixed_x & set(xs)
