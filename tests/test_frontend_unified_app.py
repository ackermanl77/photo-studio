"""Regression tests for app/frontend_unified_app.py's tab composition.

Every tab in the unified workspace runs on every Streamlit rerun and feeds the same
accumulated `options` dict to app/backend.py's /process pipeline (see the module docstring /
"Recently Fixed" notes in docs/PROJECT_STATUS.md). Nothing exercised the composed script
directly before this file, which is exactly how Layers and Masks were once silently dropped
from the final image without any test catching it. These tests drive the real app through
Streamlit's AppTest harness instead of the individual processing functions in isolation.

The network call to the processing backend is stubbed (see `_stub_backend` below) so these
stay fast, offline unit tests; the stub still runs the real app.backend.process_image()
pipeline in-process, just without the HTTP/subprocess round trip, so a regression in how the
frontend assembles the request would still be caught.
"""

import base64
import io
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from app import backend_client
from app.backend import process_image
from app.layer_workspace import LayerWorkspace, WorkspaceLayer
from app.major_features import PhotoCatalog
from app.presets import PRESETS
from tests.legacy_project import PhotoProject
from app.project_bundle import ProjectBundle

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "frontend_unified_app.py"
UI_DIR = APP_PATH.parent / "ui"
TOOL_TABS = ["☼ Basic", "🎨 Curves & Color", "🔬 Detail", "⛶ Geometry", "◉ Local", "◈ Layers", "✨ Effects", "⬇ Export", "✦ Presets & Library"]


def ui_source() -> str:
    """The UI's source text: the entry script plus every app/ui module (F14 split it up)."""
    return "\n".join(path.read_text(encoding="utf-8") for path in [APP_PATH, *sorted(UI_DIR.rglob("*.py"))])


def open_tab(at, tool_tab: str, sub_tab: str | None = None) -> None:
    """Make a tab the visible one for the next run (hidden tabs skip display-only work, F1/F9)."""
    at.session_state["tool-tab"] = tool_tab
    if sub_tab is not None:
        at.session_state["local-tab" if tool_tab == "◉ Local" else "library-tab"] = sub_tab

# Every test here reruns the whole Streamlit script -- `pytest -m "not slow"` skips them for a quick
# check of the processing modules alone (T7 in the improvement report).
pytestmark = pytest.mark.slow


def image_bytes(color=(90, 120, 160), size=(40, 30), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def painted_mask_base64(size=(20, 20)):
    """Stands in for what the on-canvas paint editor's JS would produce: a solid-white PNG whose
    ALPHA channel is the mask (right half painted, left half not) -- see
    app.frontend_unified_app._decode_painted_mask, which reads exactly that alpha channel."""
    mask = Image.new("RGBA", size, (255, 255, 255, 0))
    for x in range(size[0] // 2, size[0]):
        for y in range(size[1]):
            mask.putpixel((x, y), (255, 255, 255, 255))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _FakeFileStorage:
    """Minimal stand-in for Flask's FileStorage -- just enough of its interface
    (`.filename`, `.read()`, `.stream`) for app.backend.process_image() to accept it."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.stream = io.BytesIO(content)

    def read(self) -> bytes:
        self.stream.seek(0)
        return self.stream.read()


def _fake_process_via_backend(image_bytes_, filename, options, local_mask_bytes=None, layers_bytes=None, removal_mask_bytes=None, stage_handle=None, local_masks_bytes=None, lane=None, timeout=None):
    request_options = dict(options)
    request_options["output_format"] = "PNG"
    image_file = _FakeFileStorage(filename, image_bytes_)
    local_mask = _FakeFileStorage("mask.png", local_mask_bytes) if local_mask_bytes else None
    layers_file = _FakeFileStorage("layers.json", layers_bytes) if layers_bytes else None
    removal_mask = _FakeFileStorage("removal_mask.png", removal_mask_bytes) if removal_mask_bytes else None
    try:
        local_masks = _FakeFileStorage("local_masks.json", local_masks_bytes) if local_masks_bytes else None
        return process_image(image_file, request_options, removal_mask=removal_mask, local_mask=local_mask, layers_file=layers_file, local_masks_file=local_masks)
    except (KeyError, ValueError, OSError, RuntimeError, json.JSONDecodeError, cv2.error) as error:
        # Mirrors app/backend.py's /process route, which catches exactly this tuple and returns
        # a 400 JSON error -- process_via_backend() (the real HTTP client) then turns a non-OK
        # response into a RuntimeError. This stub calls process_image() directly, bypassing
        # Flask's request/response cycle entirely, so without this it would let a raw exception
        # (e.g. a corrupted layer image) escape here uncaught -- a failure mode a real HTTP round
        # trip already can't produce, since the Flask route always catches it first.
        raise RuntimeError(f"Processing backend error: {error}") from error


@pytest.fixture(autouse=True)
def _stub_backend(monkeypatch):
    """Skip the real HTTP call / on-demand subprocess: app.frontend_unified_app imports
    `ensure_backend_running`/`process_via_backend` fresh from app.backend_client on every
    Streamlit rerun (`from ... import ...`), so patching them on the shared app.backend_client
    module object is picked up by every rerun without needing to reach into the app module."""
    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)


def run_with_uploaded_image(**upload_kwargs):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("photo.jpg", image_bytes(**upload_kwargs), "image/jpeg"))
    at.run(timeout=30)
    return at


def canvas_hash(at):
    """The media URL Streamlit assigns the main preview image -- distinct content gets a
    distinct URL, so comparing it before/after an edit is a cheap way to assert the edit
    actually reached the composed image without decoding pixels."""
    matches = [img for img in at.image if img.captions and img.captions[0] == "Current Edited Image"]
    assert matches, "expected the main canvas image to be rendered"
    return matches[0].value[0]


def canvas_dims(at):
    markdown = next(m.value for m in at.markdown if "Live View" in m.value)
    match = re.search(r"\((\d+)×(\d+)\)", markdown)
    assert match, f"could not find dimensions in: {markdown!r}"
    return int(match.group(1)), int(match.group(2))


def test_no_image_shows_upload_prompt_and_stops():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    assert any("Upload an image" in info.value for info in at.info)
    assert len(at.tabs) == 0


def test_empty_state_offers_a_sample_photo_button_next_to_the_uploader():
    """A prior UI review found the empty state was a single st.info() line plus a bare
    file_uploader, with no guided first-import option (unlike Lightroom's/Capture One's sample
    onboarding screens). "Try a sample photo" should be visible right there."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    assert at.button(key="sample-photo-btn") is not None


def test_sample_photo_button_loads_a_valid_image_through_the_normal_upload_path():
    """Clicking "Try a sample photo" should open the full editing workspace -- the same
    restored_source_bytes/restored_source_name fallback a reopened .photoproject already uses,
    which every downstream step (PhotoProcessor.from_bytes, the tab tree, canvas_col) already
    treats identically to a real upload."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    at.button(key="sample-photo-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert len(at.tabs) == len(TOOL_TABS) + 5 + 4  # + the Local and Presets & Library sub-tabs, the curve channels
    assert canvas_hash(at)


def test_sample_photo_button_is_hidden_once_a_real_image_is_loaded():
    """Once a photo is loaded (real upload or a reopened project), the button should no longer
    be offered -- clicking it after the fact would silently swap out the user's own photo."""
    at = run_with_uploaded_image()

    assert not at.exception
    assert not any(b.key == "sample-photo-btn" for b in at.button)


def test_new_tooltips_render_without_breaking_the_app():
    """Spot-checks a representative sample of the help= tooltips added across Tone, Detail,
    Masks, and Layer Styles -- confirms they're present and that adding them didn't disturb the
    widgets' own values/behavior."""
    at = run_with_uploaded_image()

    assert at.slider(key="t-ev").help
    assert at.slider(key="d-text").help
    assert at.slider(key="m-tev").help
    assert at.checkbox(key="d-sharp").help

    at.radio(key="l-new-kind").set_value("Text")
    at.run(timeout=30)
    at.button(key="l-add-text-btn").click()
    at.run(timeout=30)
    layer_id = at.session_state["image_layers"][0]["id"]
    at.checkbox(key=f"l-ds-en-{layer_id}").set_value(True)
    at.run(timeout=30)

    assert not at.exception
    assert at.checkbox(key=f"l-ds-en-{layer_id}").help
    assert at.slider(key=f"l-ds-ox-{layer_id}").help


def test_two_stage_cache_skips_detail_stage_when_only_a_later_option_changes(monkeypatch):
    """The whole point of the two-stage cache split (_cached_process_through_detail /
    _cached_process_from_repair_onward, see docs/PROJECT_STATUS.md's Known Issues): a Mask-only
    change should not re-invoke the expensive "through Detail" stage -- st.cache_data should serve
    it from cache instead, since none of its own inputs changed -- while the cheaper "Repair
    onward" stage does re-run. Wraps the existing _fake_process_via_backend stub with a call
    counter rather than replacing it, so this still exercises the real in-process pipeline."""
    calls = []

    def counting_stub(image_bytes_, filename, options, *args, **kwargs):
        calls.append(dict(options))
        return _fake_process_via_backend(image_bytes_, filename, options, *args, **kwargs)

    monkeypatch.setattr(backend_client, "process_via_backend", counting_stub)

    at = run_with_uploaded_image(color=(80, 80, 80))
    assert sum(1 for c in calls if c.get("stop_after") == "detail") == 1

    at.selectbox(key="m-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.button(key="m-add-btn").click()
    at.run(timeout=30)
    at.slider(key="m-tev").set_value(1.0)
    at.run(timeout=30)

    assert not at.exception
    assert sum(1 for c in calls if c.get("stop_after") == "detail") == 1  # still just the initial call -- cached
    assert sum(1 for c in calls if c.get("resume_from_detail")) >= 2  # re-ran for each mask-related change


def test_uploading_a_corrupted_image_shows_an_error_instead_of_crashing():
    """Regression test for a real bug found in review: a truncated/corrupted image upload used
    to raise an uncaught OSError (PIL's "image file is truncated" / UnidentifiedImageError) all
    the way up, showing the end user a raw Python traceback -- including local file paths --
    instead of a plain error message. Only RuntimeError was caught here before; reproduced live
    by uploading a real PNG cut off halfway through."""
    valid_png = image_bytes()
    truncated = valid_png[: len(valid_png) // 2]

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("broken.png", truncated, "image/png"))
    at.run(timeout=30)

    assert not at.exception
    assert any("corrupted" in message.value.lower() for message in at.error)
    assert len(at.tabs) == 0


def test_uploading_image_renders_every_tab_without_errors():
    at = run_with_uploaded_image()

    assert not at.exception
    # Grouped by workflow (F9): nine tool tabs, with sub-tabs inside Local and Presets & Library.
    labels = [t.label for t in at.tabs]
    assert [label for label in labels if label in TOOL_TABS] == TOOL_TABS == [
        "☼ Basic", "🎨 Curves & Color", "🔬 Detail", "⛶ Geometry", "◉ Local", "◈ Layers",
        "✨ Effects", "⬇ Export", "✦ Presets & Library",
    ]
    assert [label for label in labels if label not in TOOL_TABS] == [
        "RGB", "Red", "Green", "Blue",  # the tone curve's channels (F13)
        "◉ Masks", "🩹 Repair", "◌ Portrait", "✦ Presets", "☰ Library & Batch",
    ]
    assert canvas_dims(at) == (40, 30)


def test_ai_upscale_control_shows_an_install_caption_when_unavailable(monkeypatch):
    """Most environments only install requirements.txt (never requirements-gpu.txt, per
    CLAUDE.md's Commands section and CI's own install list), so app.ai_upscale.ai_upscale_installed()
    is normally False -- but a dev machine that DOES have requirements-gpu.txt installed (e.g. to
    verify the feature for real) must not make this test flaky, so the "not installed" case is
    forced via the same monkeypatch pattern the sibling "enabled" test below uses. Confirms the
    Detail tab degrades to a plain caption instead of rendering a checkbox/selectbox that would be
    broken if the feature weren't actually installed, and that the rest of the app still renders
    without error."""
    import app.ai_upscale as ai_upscale_module

    monkeypatch.setattr(ai_upscale_module, "ai_upscale_installed", lambda: False)

    at = run_with_uploaded_image()

    assert not at.exception
    assert any("requirements-gpu.txt" in caption.value for caption in at.caption)
    assert not [checkbox for checkbox in at.checkbox if checkbox.key == "d-ai-upscale-en"]


def test_ai_upscale_is_an_export_action_run_only_when_save_is_clicked(monkeypatch):
    """AI Upscale moved from the Detail tab to the Export settings: choosing a factor must never
    run Real-ESRGAN during a rerun (it used to re-run on every Stage 2 change), and clicking Save
    must hand the backend the selected factor. The Save button's data is a callable Streamlit
    runs on click; AppTest can't click a download, so st.download_button is wrapped to capture
    the callable and invoke it directly. PhotoProcessor.apply_ai_upscale is faked (no GPU/model)."""
    import streamlit as st

    import app.ai_upscale as ai_upscale_module
    from app.image_processor import PhotoProcessor

    monkeypatch.setattr(ai_upscale_module, "ai_upscale_installed", lambda: True)
    monkeypatch.setattr(ai_upscale_module, "ai_upscale_available", lambda: True)
    calls = []

    def fake_apply_ai_upscale(self, factor=2):
        calls.append(factor)
        self.image = self.image.resize((self.image.width * factor, self.image.height * factor))
        return self

    monkeypatch.setattr(PhotoProcessor, "apply_ai_upscale", fake_apply_ai_upscale)
    exports = []

    def fake_export_via_backend(image_bytes_, filename, options, export_settings, local_mask_bytes=None, layers_bytes=None, removal_mask_bytes=None):
        from app.export import encode_export

        result = process_image(_FakeFileStorage(filename, image_bytes_), options)
        exports.append(options)
        return encode_export(result, export_settings)[0]

    monkeypatch.setattr(backend_client, "export_via_backend", fake_export_via_backend)
    captured = {}
    original_download_button = st.download_button

    def capturing_download_button(label, data, *args, **kwargs):
        if kwargs.get("key") == "exp-download-btn":
            captured["data"] = data
        return original_download_button(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capturing_download_button)

    at = run_with_uploaded_image()
    at.selectbox(key="exp-ai-upscale").set_value(4)
    at.run(timeout=30)

    assert not at.exception
    assert not calls  # choosing the factor ran nothing
    assert not [checkbox for checkbox in at.checkbox if checkbox.key == "d-ai-upscale-en"]

    content = captured["data"]()
    assert calls == [4]
    assert exports[-1]["ai_upscale"] is True and exports[-1]["ai_upscale_factor"] == 4
    assert Image.open(io.BytesIO(content)).size == (160, 120)


def test_layer_composite_is_reflected_in_final_image():
    """Regression test for the bug where the Layers tab composited into a local variable
    that every later tab (and the final export) ignored -- see docs/PROJECT_STATUS.md's
    "Recently Fixed" entry. A layer that changes the composite must change what's shown."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.get_by_key("l-new-file").set_value(("overlay.png", image_bytes((255, 0, 0), (40, 30)), "image/png"))
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_layers_tab_supports_stacking_more_than_one_layer():
    """The Layers tab used to accept exactly one overlay via a single file_uploader; the
    LayerWorkspace compositing engine underneath always supported an arbitrary stack, so this
    checks the panel now actually exposes that -- two uploads must both land in the stack, in
    the order they were added, each with its own default opacity/blend mode."""
    at = run_with_uploaded_image(color=(10, 10, 10))

    at.get_by_key("l-new-file").set_value(("first.png", image_bytes((255, 0, 0), (40, 30)), "image/png"))
    at.run(timeout=30)
    at.get_by_key("l-new-file").set_value(("second.png", image_bytes((0, 255, 0), (40, 30)), "image/png"))
    at.run(timeout=30)

    assert not at.exception
    layers = at.session_state["image_layers"]
    assert [layer["name"] for layer in layers] == ["first", "second"]
    assert all(layer["opacity"] == 1.0 and layer["blend_mode"] == "normal" and layer["visible"] for layer in layers)


def test_removing_a_layer_takes_it_out_of_the_stack_and_the_final_image():
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.get_by_key("l-new-file").set_value(("overlay.png", image_bytes((255, 0, 0), (40, 30)), "image/png"))
    at.run(timeout=30)
    with_layer_hash = canvas_hash(at)
    layer_id = at.session_state["image_layers"][0]["id"]

    at.button(key=f"l-rm-{layer_id}").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["image_layers"] == []
    assert canvas_hash(at) != with_layer_hash


def test_reordering_layers_changes_which_one_is_on_top():
    """Moving the bottom layer above the top one must change the composite -- if a fully-opaque
    layer ends up on top, it should now be what's visible instead of the layer that used to be
    above it."""
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.get_by_key("l-new-file").set_value(("red.png", image_bytes((255, 0, 0), (40, 30)), "image/png"))
    at.run(timeout=30)
    at.get_by_key("l-new-file").set_value(("blue.png", image_bytes((0, 0, 255), (40, 30)), "image/png"))
    at.run(timeout=30)
    bottom_layer_id = at.session_state["image_layers"][0]["id"]  # "red", currently underneath "blue"

    at.button(key=f"l-up-{bottom_layer_id}").click()
    at.run(timeout=30)

    assert not at.exception
    assert [layer["name"] for layer in at.session_state["image_layers"]] == ["blue", "red"]


def test_changing_a_layers_blend_mode_changes_the_final_image():
    at = run_with_uploaded_image(color=(100, 100, 100))
    at.get_by_key("l-new-file").set_value(("overlay.png", image_bytes((200, 200, 200), (40, 30)), "image/png"))
    at.run(timeout=30)
    normal_hash = canvas_hash(at)
    layer_id = at.session_state["image_layers"][0]["id"]

    at.selectbox(key=f"l-blend-{layer_id}").set_value("multiply")
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["image_layers"][0]["blend_mode"] == "multiply"
    assert canvas_hash(at) != normal_hash


def test_enabling_a_layer_style_changes_the_final_image_and_persists_its_settings():
    # A Text layer, not an uploaded Image layer: Layer Styles only produce a visible difference
    # where a layer has genuinely transparent surrounding pixels for a shadow/glow/stroke to
    # occupy -- an opaque, canvas-filling uploaded rectangle has no "outside its own shape" area
    # within the canvas for any of these effects to appear in, the same real limitation Photoshop's
    # own layer styles have on a fully opaque background-filling layer.
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Text")
    at.run(timeout=30)
    at.button(key="l-add-text-btn").click()
    at.run(timeout=30)
    without_style_hash = canvas_hash(at)
    layer_id = at.session_state["image_layers"][0]["id"]

    at.checkbox(key=f"l-st-en-{layer_id}").set_value(True)
    at.run(timeout=30)
    at.color_picker(key=f"l-st-col-{layer_id}").set_value("#0000FF")
    at.slider(key=f"l-st-w-{layer_id}").set_value(10)
    at.run(timeout=30)

    assert not at.exception
    styles = at.session_state["image_layers"][0]["styles"]
    assert styles["stroke_enabled"] is True
    assert styles["stroke_color"] == "#0000FF"
    assert styles["stroke_width"] == 10
    assert canvas_hash(at) != without_style_hash


def test_disabling_a_layer_style_does_not_affect_other_styles_on_the_same_layer():
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Text")
    at.run(timeout=30)
    at.button(key="l-add-text-btn").click()
    at.run(timeout=30)
    layer_id = at.session_state["image_layers"][0]["id"]

    at.checkbox(key=f"l-ds-en-{layer_id}").set_value(True)
    at.checkbox(key=f"l-co-en-{layer_id}").set_value(True)
    at.run(timeout=30)

    assert not at.exception
    styles = at.session_state["image_layers"][0]["styles"]
    assert styles["drop_shadow_enabled"] is True
    assert styles["color_overlay_enabled"] is True
    assert styles["stroke_enabled"] is False
    assert styles["outer_glow_enabled"] is False


def test_adding_a_text_layer_renders_text_onto_the_final_image():
    at = run_with_uploaded_image(color=(10, 10, 10))
    baseline = canvas_hash(at)

    at.radio(key="l-new-kind").set_value("Text")
    at.run(timeout=30)
    at.button(key="l-add-text-btn").click()
    at.run(timeout=30)

    assert not at.exception
    layers = at.session_state["image_layers"]
    assert len(layers) == 1
    assert layers[0]["kind"] == "text"
    assert canvas_hash(at) != baseline



def test_a_text_layer_and_the_watermark_offer_a_font_choice():
    from app import fonts

    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Text")
    at.run(timeout=30)
    at.button(key="l-add-text-btn").click()
    at.run(timeout=30)
    layer_id = at.session_state["image_layers"][0]["id"]
    font_box = at.selectbox(key=f"l-font-{layer_id}")
    assert font_box.value == ""  # the automatic default
    assert at.selectbox(key="wm-font-sel").value == ""
    names = list(fonts.available_fonts())
    if names:
        font_box.set_value(names[0])
        at.run(timeout=30)
        assert not at.exception
        assert at.session_state["image_layers"][0]["font"] == names[0]

def test_editing_a_text_layers_content_color_and_position_changes_the_final_image():
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Text")
    at.run(timeout=30)
    at.button(key="l-add-text-btn").click()
    at.run(timeout=30)
    default_text_hash = canvas_hash(at)
    layer_id = at.session_state["image_layers"][0]["id"]

    at.text_area(key=f"l-text-{layer_id}").set_value("Poster Title")
    at.color_picker(key=f"l-fcolor-{layer_id}").set_value("#FF0000")
    at.slider(key=f"l-fx-{layer_id}").set_value(20.0)
    at.run(timeout=30)

    assert not at.exception
    updated_layer = at.session_state["image_layers"][0]
    assert updated_layer["text"] == "Poster Title"
    assert updated_layer["color"] == "#FF0000"
    assert updated_layer["x_pct"] == 20.0
    assert canvas_hash(at) != default_text_hash


def test_a_text_layer_can_be_reordered_and_removed_like_an_image_layer():
    """Text layers share the generic opacity/blend/visible/up/down/duplicate/remove controls with
    Image layers -- this exercises that shared code path for the "text" kind specifically, since
    the earlier reorder/remove tests only ever used Image layers."""
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Text")
    at.run(timeout=30)
    at.button(key="l-add-text-btn").click()
    at.run(timeout=30)
    layer_id = at.session_state["image_layers"][0]["id"]

    at.button(key=f"l-rm-{layer_id}").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["image_layers"] == []


def test_adding_a_shape_layer_renders_it_onto_the_final_image():
    at = run_with_uploaded_image(color=(10, 10, 10))
    baseline = canvas_hash(at)

    at.radio(key="l-new-kind").set_value("Shape")
    at.run(timeout=30)
    at.selectbox(key="l-new-shape-type").set_value("Ellipse")
    at.button(key="l-add-shape-btn").click()
    at.run(timeout=30)

    assert not at.exception
    layers = at.session_state["image_layers"]
    assert len(layers) == 1
    assert layers[0]["kind"] == "shape"
    assert layers[0]["shape_type"] == "Ellipse"
    assert canvas_hash(at) != baseline


def test_a_layer_mask_from_a_masks_selection_limits_the_layer():
    """F13: a layer's "Layer mask" picks one of the Masks tab's selections; the composite changes
    with it, its Invert toggle, and an edit of that selection (the stack signature carries it)."""
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Shape")
    at.run(timeout=30)
    at.button(key="l-add-shape-btn").click()
    at.run(timeout=30)
    at.selectbox(key="m-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.slider(key="m-new-rr").set_value(0.1)  # smaller than the centred 30 % shape
    at.button(key="m-add-btn").click()
    at.run(timeout=30)
    unmasked = canvas_hash(at)

    layer_id = at.session_state["image_layers"][0]["id"]
    selection = at.session_state["mask_layers"][0]
    at.selectbox(key=f"l-mask-{layer_id}").set_value(selection["id"])
    at.run(timeout=30)
    masked = canvas_hash(at)
    assert not at.exception and masked != unmasked
    assert at.session_state["image_layers"][0]["mask_selection"] == selection["id"]

    at.checkbox(key=f"l-mask-inv-{layer_id}").check()
    at.run(timeout=30)
    assert canvas_hash(at) not in (masked, unmasked)

    at.checkbox(key=f"l-mask-inv-{layer_id}").uncheck()
    at.slider(key=f"m-sf-{selection['id']}").set_value(20)
    at.run(timeout=30)
    assert canvas_hash(at) not in (masked, unmasked)  # the selection's edit reached the layer


def test_a_layers_expander_splits_into_content_styles_and_settings_sub_tabs():
    """Regression test for splitting each layer's expander into sub-tabs (Content/Styles/Settings)
    to declutter what had become a long scroll once a layer could be four kinds each with several
    controls, plus Layer Styles' own four effects. Nesting st.tabs() inside st.expander() is legal
    in Streamlit (unlike nesting a second st.expander()), and -- like every other tab in this file --
    every sub-tab's body still executes on every rerun regardless of which is visually selected, so
    a widget from the LAST sub-tab (Settings' "Visible" checkbox) must still be directly reachable
    and functional without ever "clicking into" that tab first."""
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Text")
    at.run(timeout=30)
    at.button(key="l-add-text-btn").click()
    at.run(timeout=30)
    layer_id = at.session_state["image_layers"][0]["id"]

    sub_tab_labels = {tab.label for tab in at.tabs if tab.label in {"🎨 Content", "✨ Styles", "⚙️ Settings"}}
    assert sub_tab_labels == {"🎨 Content", "✨ Styles", "⚙️ Settings"}

    at.checkbox(key=f"l-vis-{layer_id}").set_value(False)
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["image_layers"][0]["visible"] is False


def test_editing_a_shape_layers_size_position_and_fill_changes_the_final_image():
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Shape")
    at.run(timeout=30)
    at.button(key="l-add-shape-btn").click()  # default shape type: Rectangle
    at.run(timeout=30)
    default_hash = canvas_hash(at)
    layer_id = at.session_state["image_layers"][0]["id"]

    at.color_picker(key=f"l-shp-fcolor-{layer_id}").set_value("#00FF00")
    at.slider(key=f"l-shp-w-{layer_id}").set_value(60.0)
    at.slider(key=f"l-shp-rot-{layer_id}").set_value(45.0)
    at.run(timeout=30)

    assert not at.exception
    updated = at.session_state["image_layers"][0]
    assert updated["fill_color"] == "#00FF00"
    assert updated["width_pct"] == 60.0
    assert updated["rotation_deg"] == 45.0
    assert canvas_hash(at) != default_hash


def test_a_shape_layer_can_be_reordered_and_removed_like_an_image_layer():
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Shape")
    at.run(timeout=30)
    at.button(key="l-add-shape-btn").click()
    at.run(timeout=30)
    layer_id = at.session_state["image_layers"][0]["id"]

    at.button(key=f"l-rm-{layer_id}").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["image_layers"] == []


def _triangle_path_points():
    return [{"x_pct": 20, "y_pct": 20, "handle_dx_pct": 0, "handle_dy_pct": 0},
            {"x_pct": 80, "y_pct": 20, "handle_dx_pct": 0, "handle_dy_pct": 0},
            {"x_pct": 50, "y_pct": 80, "handle_dx_pct": 0, "handle_dy_pct": 0}]


def test_adding_a_pen_path_layer_via_the_hidden_commit_field_renders_onto_the_final_image():
    """The Pen tool's actual click/drag interaction lives inside a components.html() iframe, which
    streamlit.testing.v1.AppTest cannot exercise (see CLAUDE.md's Testing conventions) -- this
    drives the same hidden text_area the real JS commits into on each click, exactly like the
    existing crop-editor/wand-picker tests do for their own on-canvas editors."""
    at = run_with_uploaded_image(color=(10, 10, 10))
    baseline = canvas_hash(at)

    at.radio(key="l-new-kind").set_value("Pen")
    at.run(timeout=30)
    at.text_area(key="l-path-draft-data").set_value(json.dumps(_triangle_path_points()))
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["l-path-draft-points"] == _triangle_path_points()

    at.button(key="l-add-path-btn").click()
    at.run(timeout=30)

    assert not at.exception
    layers = at.session_state["image_layers"]
    assert len(layers) == 1
    assert layers[0]["kind"] == "path"
    assert layers[0]["points"] == _triangle_path_points()
    assert at.session_state["l-path-draft-points"] == []  # draft cleared after adding
    assert canvas_hash(at) != baseline


def test_adding_a_closed_pen_path_resets_closed_path_for_the_next_draft():
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Pen")
    at.run(timeout=30)
    at.text_area(key="l-path-draft-data").set_value(json.dumps(_triangle_path_points()))
    at.checkbox(key="l-path-draft-closed").set_value(True)
    at.run(timeout=30)
    at.button(key="l-add-path-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["image_layers"][0]["closed"] is True
    assert at.checkbox(key="l-path-draft-closed").value is False


def test_closing_a_pen_path_and_enabling_fill_changes_the_final_image():
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Pen")
    at.run(timeout=30)
    at.text_area(key="l-path-draft-data").set_value(json.dumps(_triangle_path_points()))
    at.run(timeout=30)
    at.button(key="l-add-path-btn").click()
    at.run(timeout=30)
    open_stroke_hash = canvas_hash(at)
    layer_id = at.session_state["image_layers"][0]["id"]

    at.checkbox(key=f"l-path-{layer_id}-closed").set_value(True)
    at.run(timeout=30)
    at.checkbox(key=f"l-path-fen-{layer_id}").set_value(True)
    at.run(timeout=30)
    at.color_picker(key=f"l-path-fcolor-{layer_id}").set_value("#00FF00")
    at.run(timeout=30)

    assert not at.exception
    updated = at.session_state["image_layers"][0]
    assert updated["closed"] is True
    assert updated["fill_enabled"] is True
    assert updated["fill_color"] == "#00FF00"
    assert canvas_hash(at) != open_stroke_hash


def test_editing_an_existing_pen_path_layers_points_changes_the_final_image():
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Pen")
    at.run(timeout=30)
    at.text_area(key="l-path-draft-data").set_value(json.dumps(_triangle_path_points()))
    at.run(timeout=30)
    at.button(key="l-add-path-btn").click()
    at.run(timeout=30)
    original_hash = canvas_hash(at)
    layer_id = at.session_state["image_layers"][0]["id"]

    moved_points = [{"x_pct": 10, "y_pct": 10, "handle_dx_pct": 0, "handle_dy_pct": 0},
                     {"x_pct": 90, "y_pct": 10, "handle_dx_pct": 0, "handle_dy_pct": 0},
                     {"x_pct": 50, "y_pct": 90, "handle_dx_pct": 0, "handle_dy_pct": 0}]
    at.text_area(key=f"l-path-{layer_id}-data").set_value(json.dumps(moved_points))
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["image_layers"][0]["points"] == moved_points
    assert canvas_hash(at) != original_hash


def test_a_pen_path_layer_can_be_reordered_and_removed_like_an_image_layer():
    at = run_with_uploaded_image(color=(10, 10, 10))
    at.radio(key="l-new-kind").set_value("Pen")
    at.run(timeout=30)
    at.text_area(key="l-path-draft-data").set_value(json.dumps(_triangle_path_points()))
    at.run(timeout=30)
    at.button(key="l-add-path-btn").click()
    at.run(timeout=30)
    layer_id = at.session_state["image_layers"][0]["id"]

    at.button(key=f"l-rm-{layer_id}").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["image_layers"] == []


def test_mask_stack_target_adjustment_is_reflected_in_final_image():
    """Regression test for the bug where only one mask type could be active at a time --
    exercises the same MaskStack path (add a layer, then adjust Target Exposure) that the
    unified UI now uses."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.selectbox(key="m-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.button(key="m-add-btn").click()
    at.run(timeout=30)
    assert at.session_state["mask_layers"], "expected the mask layer to be added"

    at.slider(key="m-tev").set_value(1.3)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_repair_selection_removes_content_and_changes_the_final_image():
    """Regression coverage for the Repair tab (object removal) -- PhotoProcessor.remove_objects()
    was previously only reachable via the CLI or the raw backend API, never the unified UI. Unlike
    Masks, no separate "target adjustment" slider is needed -- adding an enabled selection alone
    triggers removal."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.selectbox(key="rm-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.button(key="rm-add-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["removal_mask_layers"], "expected the removal selection to be added"
    assert canvas_hash(at) != baseline


def test_painted_mask_selection_changes_the_final_image_via_masks_tab():
    """The on-canvas paint editor itself is JS an AppTest harness (no real DOM) can't drive -- this
    instead verifies the Python-side wiring its "Add to selections" button ultimately depends on:
    setting the hidden textarea it writes into (see _paint_editor_html / the "Painted" branch in
    _render_selection_stack) is exactly what that JS does, just without a real canvas involved."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.selectbox(key="m-new-kind").set_value("Painted")
    at.run(timeout=30)
    at.text_area(key="m-paint-data").set_value(painted_mask_base64())
    at.run(timeout=30)
    at.run(timeout=30)  # a second rerun with the same committed value must not add a duplicate layer

    assert not at.exception
    layers = at.session_state["mask_layers"]
    assert len(layers) == 1
    assert layers[0]["kind"] == "Painted"

    at.slider(key="m-tev").set_value(1.3)
    at.run(timeout=30)
    assert canvas_hash(at) != baseline


def test_painted_mask_selection_removes_content_via_repair_tab():
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.selectbox(key="rm-new-kind").set_value("Painted")
    at.run(timeout=30)
    at.text_area(key="rm-paint-data").set_value(painted_mask_base64())
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["removal_mask_layers"][0]["kind"] == "Painted"
    assert canvas_hash(at) != baseline


def _full_painted_mask_base64(size=(20, 20)):
    """A distinctly different painted region than painted_mask_base64()'s right-half default --
    the whole canvas opaque -- for asserting an edit genuinely changed the stored mask rather than
    round-tripping the same bytes."""
    mask = Image.new("RGBA", size, (255, 255, 255, 255))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_editing_an_existing_painted_mask_updates_it_in_place():
    """Regression coverage for the "Edit Painting" toggle (_render_selection_stack's per-layer
    re-edit section): committing through it must update the same layer's painted mask in place, not
    append a second layer -- the same "genuine new commit only" guard as adding a fresh Painted
    selection, applied to the edit path instead."""
    at = run_with_uploaded_image(color=(80, 80, 80))

    at.selectbox(key="m-new-kind").set_value("Painted")
    at.run(timeout=30)
    at.text_area(key="m-paint-data").set_value(painted_mask_base64())
    at.run(timeout=30)

    layer_id = at.session_state["mask_layers"][0]["id"]
    # Painted masks are stored once per content in the session table and referenced by hash (P5).
    original_ref = at.session_state["mask_layers"][0]["params"]["mask_ref"]
    assert at.session_state["_painted_masks"][original_ref] == painted_mask_base64()

    at.checkbox(key=f"m-editpaint-toggle-{layer_id}").set_value(True)
    at.run(timeout=30)
    edited_mask = _full_painted_mask_base64()
    at.text_area(key=f"m-ep-{layer_id}-paint-data").set_value(edited_mask)
    at.run(timeout=30)
    at.run(timeout=30)  # a second rerun with the same committed edit must not add a duplicate layer

    assert not at.exception
    layers = at.session_state["mask_layers"]
    assert len(layers) == 1
    assert layers[0]["id"] == layer_id
    assert layers[0]["params"]["mask_ref"] != original_ref
    assert at.session_state["_painted_masks"][layers[0]["params"]["mask_ref"]] == edited_mask


def test_masks_tab_state_survives_adding_a_repair_selection():
    """Regression test for a real bug found live: Repair renders before Masks in the tab order
    (see the pipeline-order comment above `options = {...}`), and every tab's code runs on every
    rerun regardless of which is visible. Before `_request_rerun()` existed, committing a Painted
    removal selection in Repair called st.rerun() immediately -- aborting the script before the
    Masks tab's own widgets (further down) were ever instantiated that pass. Streamlit garbage-
    collects a keyed widget's session_state when the widget isn't instantiated during a run (the
    same mechanism documented in PROJECT_STATUS.md for the HSL Color Mixer's dropdown-hidden-band
    bug), so Target Saturation and the "Add a selection" dropdown silently reset to their defaults
    on the very next run -- even though `st.session_state.mask_layers` itself (not a widget key)
    was untouched and still showed the mask as enabled. Confirmed live: the masked Target
    Saturation effect visibly disappeared from the rendered image after adding a Repair selection,
    and reappeared only once an unrelated widget interaction forced another full rerun."""
    at = run_with_uploaded_image(color=(80, 80, 80))

    at.selectbox(key="m-new-kind").set_value("Painted")
    at.run(timeout=30)
    at.text_area(key="m-paint-data").set_value(painted_mask_base64())
    at.run(timeout=30)
    at.slider(key="m-tsaturation").set_value(72)
    at.run(timeout=30)
    baseline = canvas_hash(at)

    at.selectbox(key="rm-new-kind").set_value("Painted")
    at.run(timeout=30)
    at.text_area(key="rm-paint-data").set_value(painted_mask_base64())
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="m-tsaturation").value == 72
    assert at.selectbox(key="m-new-kind").value == "Painted"
    assert canvas_hash(at) != baseline


def test_repair_softness_slider_disabled_until_a_selection_exists():
    at = run_with_uploaded_image()

    assert at.slider(key="rm-radius").disabled

    at.selectbox(key="rm-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.button(key="rm-add-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert not at.slider(key="rm-radius").disabled


def test_crop_tab_changes_output_dimensions():
    at = run_with_uploaded_image(size=(40, 30))
    assert canvas_dims(at) == (40, 30)

    at.selectbox(key="crop-aspect").set_value("1:1 Square")
    at.run(timeout=30)

    assert not at.exception
    assert canvas_dims(at) == (30, 30)


def test_selecting_a_grid_overlay_changes_the_canvas_image():
    """Grid/guides are view-only (see app.professional_tools.draw_guides_and_grid's docstring) --
    this confirms the toggle actually reaches the rendered canvas, the same regression shape as the
    "Show clipping" overlay checkbox, without asserting on exact pixel positions (covered directly
    in tests/test_professional_tools.py)."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.selectbox(key="grid-type").set_value("Rule of Thirds")
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_adding_and_removing_a_guide_changes_the_canvas_image_and_its_own_state():
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.selectbox(key="guide-new-orientation").set_value("Vertical")
    at.slider(key="guide-new-position").set_value(25.0)
    at.button(key="guide-add-btn").click()
    at.run(timeout=30)

    assert not at.exception
    guides = at.session_state["guides"]
    assert len(guides) == 1
    assert guides[0]["orientation"] == "vertical"
    assert guides[0]["position_pct"] == 25.0
    with_guide_hash = canvas_hash(at)
    assert with_guide_hash != baseline

    at.button(key=f"guide-rm-{guides[0]['id']}").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["guides"] == []
    assert canvas_hash(at) == baseline


def test_export_multiple_sizes_prepares_a_downloadable_zip():
    at = run_with_uploaded_image(size=(400, 300))

    at.multiselect(key="exp-multi-profiles").set_value(["Web", "Social"])
    at.run(timeout=30)
    at.button(key="exp-multi-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any(db.label == "⬇️ Download ZIP" for db in at.download_button)


def test_export_multiple_sizes_button_disabled_with_no_sizes_selected():
    at = run_with_uploaded_image(size=(400, 300))

    at.multiselect(key="exp-multi-profiles").set_value([])
    at.run(timeout=30)

    assert at.button(key="exp-multi-btn").disabled


def test_mask_target_sliders_disabled_until_an_enabled_mask_layer_exists():
    at = run_with_uploaded_image()

    assert at.slider(key="m-tev").disabled
    assert at.slider(key="m-tcontrast").disabled
    assert at.slider(key="m-tsaturation").disabled

    at.selectbox(key="m-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.button(key="m-add-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert not at.slider(key="m-tev").disabled
    assert not at.slider(key="m-tcontrast").disabled
    assert not at.slider(key="m-tsaturation").disabled


def test_removing_a_mask_layer_does_not_corrupt_the_remaining_layers():
    """Regression test for a real bug found in review: with three layers A/B/C, each layer's
    widgets used to be keyed by its list *index* (e.g. "m-sf-1"). Removing the middle layer (B)
    shifts C from index 2 to index 1 on the next run -- so C's widgets got created under B's old
    key, which still held B's stale session_state value, silently overwriting C's own soften
    value with B's. Fixed by keying each layer's widgets by a stable per-layer id instead of its
    position in the list."""
    at = run_with_uploaded_image(color=(80, 80, 80))

    for _ in range(3):
        at.selectbox(key="m-new-kind").set_value("Radial")
        at.run(timeout=30)
        at.button(key="m-add-btn").click()
        at.run(timeout=30)

    layer_a, layer_b, layer_c = at.session_state["mask_layers"]
    at.slider(key=f"m-sf-{layer_a['id']}").set_value(5)
    at.run(timeout=30)
    at.slider(key=f"m-sf-{layer_b['id']}").set_value(15)
    at.run(timeout=30)
    at.slider(key=f"m-sf-{layer_c['id']}").set_value(25)
    at.run(timeout=30)

    at.button(key=f"m-rm-{layer_b['id']}").click()
    at.run(timeout=30)
    at.run(timeout=30)  # let the deferred rerun settle

    assert not at.exception
    remaining_ids = {layer["id"] for layer in at.session_state["mask_layers"]}
    assert remaining_ids == {layer_a["id"], layer_c["id"]}
    assert at.slider(key=f"m-sf-{layer_a['id']}").value == 5
    assert at.slider(key=f"m-sf-{layer_c['id']}").value == 25
    softens = {layer["id"]: layer["soften"] for layer in at.session_state["mask_layers"]}
    assert softens[layer_a["id"]] == 5
    assert softens[layer_c["id"]] == 25


def test_adding_a_mask_selection_does_not_reset_canvas_view_settings():
    """Regression test for a real bug found in review: the deferred-rerun consume check for
    _request_rerun() used to sit before canvas_col rendered, so any _request_rerun() call from a
    tab body (like adding a mask selection here) triggered st.rerun() before canvas_col's own
    widgets -- zoom, split view, pan, export format/quality -- were instantiated that pass,
    causing Streamlit to garbage-collect and reset them. Fixed by moving the consume check to the
    very end of the script, after canvas_col."""
    at = run_with_uploaded_image()
    at.selectbox(key="cv-zoom").set_value(2.0)
    at.checkbox(key="cv-split").set_value(True)
    at.run(timeout=30)
    assert at.selectbox(key="cv-zoom").value == 2.0
    assert at.checkbox(key="cv-split").value is True

    at.selectbox(key="m-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.button(key="m-add-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["mask_layers"], "expected the mask layer to be added"
    assert at.selectbox(key="cv-zoom").value == 2.0
    assert at.checkbox(key="cv-split").value is True


def test_the_history_list_jumps_to_an_earlier_step_and_keeps_the_later_ones():
    at = run_with_uploaded_image()
    at.slider(key="t-ev").set_value(0.5)
    at.run(timeout=30)
    at.slider(key="t-contrast").set_value(20)
    at.run(timeout=30)
    labels = [button.label for button in at.button if button.key and button.key.startswith("hist-step-")]
    assert labels[:2] == ["▶ 2. Contrast 20", "1. Exposure (EV) 0.5"]

    at.button(key="hist-step-1").click()
    at.run(timeout=30)
    assert not at.exception
    assert at.slider(key="t-ev").value == 0.5 and at.slider(key="t-contrast").value == 0
    assert at.button(key="hist-step-1").disabled and not at.button(key="hist-step-2").disabled  # step 2 kept (Redo)


def test_undo_button_does_not_reset_canvas_view_settings():
    """Same bug class as test_adding_a_mask_selection_does_not_reset_canvas_view_settings, but for
    canvas_col's own Undo button, which used to call st.rerun() directly -- aborting the script
    before the zoom/split-view/pan/export widgets defined right after it in canvas_col were
    instantiated that pass."""
    at = run_with_uploaded_image()
    at.slider(key="t-ev").set_value(0.6)
    at.run(timeout=30)
    at.selectbox(key="cv-zoom").set_value(2.0)
    at.run(timeout=30)
    assert at.slider(key="t-ev").value == 0.6

    at.button(key="hist-undo-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-ev").value == 0.0
    assert at.selectbox(key="cv-zoom").value == 2.0


def test_preset_thumbnails_render_for_every_preset_option():
    at = run_with_uploaded_image()

    captions = [c.value for c in at.caption]
    assert "None" in captions
    for preset_name in PRESETS:
        assert preset_name in captions


def test_reset_button_restores_defaults_without_touching_the_layer():
    """Reset restores the same recipe fields Undo/Redo operate on (see _build_recipe) -- it must
    not silently drop an uploaded layer, since that's a separate, deliberate user action."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    at.get_by_key("l-new-file").set_value(("overlay.png", image_bytes((255, 0, 0), (40, 30)), "image/png"))
    at.run(timeout=30)
    with_layer_hash = canvas_hash(at)

    at.slider(key="t-ev").set_value(1.0)
    at.slider(key="d-text").set_value(60)
    at.run(timeout=30)
    assert at.slider(key="t-ev").value == 1.0

    at.button(key="hist-reset-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-ev").value == 0.0
    assert at.slider(key="d-text").value == 0
    # The layer itself survives the reset -- same composited image as right after uploading it.
    assert canvas_hash(at) == with_layer_hash


def test_auto_enhance_adapts_to_the_photo():
    """F10: Auto Enhance measures the photo (app.image_processor.auto_enhance_recipe) instead of
    writing the same constants for every image -- a dark photo is brightened, a bright one darkened."""
    dark = run_with_uploaded_image(color=(30, 35, 40))
    dark.button(key="auto-enhance-btn").click()
    dark.run(timeout=30)
    assert not dark.exception
    assert dark.slider(key="t-ev").value > 0.5
    assert any("Auto Enhance" in toast.value for toast in dark.toast)

    bright = run_with_uploaded_image(color=(230, 225, 220))
    bright.button(key="auto-enhance-btn").click()
    bright.run(timeout=30)
    assert bright.slider(key="t-ev").value < 0


def test_import_presets_adds_new_ones_and_shows_a_toast(monkeypatch, tmp_path):
    """Redirects Path.home() so PresetStore (which always resolves to ~/.photo-studio/presets.json
    with no override) reads/writes a throwaway location instead of the real one."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    payload = json.dumps({"version": 1, "presets": {"Imported Look": {"exposure": 1.2}}}).encode("utf-8")

    at = run_with_uploaded_image()
    at.get_by_key("preset-import-upload").set_value(("presets.json", payload, "application/json"))
    at.run(timeout=30)
    at.button(key="preset-import-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("Imported 1 preset" in t.value for t in at.toast)
    saved = json.loads((tmp_path / ".photo-studio" / "presets.json").read_text(encoding="utf-8"))
    assert "Imported Look" in saved


def test_import_presets_skips_names_that_already_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    presets_path = tmp_path / ".photo-studio" / "presets.json"
    presets_path.parent.mkdir(parents=True)
    presets_path.write_text(json.dumps({"Existing Look": {"exposure": 1.0}}), encoding="utf-8")
    payload = json.dumps({"version": 1, "presets": {"Existing Look": {"exposure": 1.5}}}).encode("utf-8")

    at = run_with_uploaded_image()
    at.get_by_key("preset-import-upload").set_value(("presets.json", payload, "application/json"))
    at.run(timeout=30)
    at.button(key="preset-import-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("already exist" in t.value for t in at.toast)
    # Untouched -- import_presets() defaults to overwrite=False.
    saved = json.loads(presets_path.read_text(encoding="utf-8"))
    assert saved["Existing Look"]["exposure"] == 1.0


def test_save_preset_without_a_name_shows_a_warning_toast():
    """Doesn't click Save with an actual preset name -- that would write to the real
    ~/.photo-studio/presets.json on whatever machine runs this test. The empty-name path is
    safe: PresetStore.save() is never called, only the toast fires."""
    at = run_with_uploaded_image()

    at.button(key="p-save-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("name" in t.value.lower() for t in at.toast)


def test_save_preset_captures_the_full_recipe_not_just_five_fields(monkeypatch, tmp_path):
    """Regression test for a real gap found in review: "Save Current Look as Preset" used to
    hand-pick only 5 fields (exposure/contrast/saturation/temperature/tint), silently dropping
    everything else -- a preset saved after dialing in a lens correction or a mask lost both.
    Now it saves _build_recipe(), the same full snapshot Undo/Redo and Save Project already use."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    at = run_with_uploaded_image()

    at.slider(key="lens-distortion").set_value(40)
    at.run(timeout=30)
    at.text_input(key="p-name-in").set_value("Full Look")
    at.run(timeout=30)
    at.button(key="p-save-btn").click()
    at.run(timeout=30)

    assert not at.exception
    saved = json.loads((tmp_path / ".photo-studio" / "presets.json").read_text(encoding="utf-8"))
    assert saved["Full Look"]["lens_distortion"] == 40
    assert "hsl_red_hue" in saved["Full Look"]  # a field the old 5-field save never touched
    assert "mask_layers" in saved["Full Look"]


def test_load_preset_button_applies_its_saved_values(monkeypatch, tmp_path):
    """Regression test for a real gap found in review: PresetStore.get() existed and was
    already tested in isolation, but nothing in the UI ever called it -- a saved preset could be
    exported and imported, but never actually reapplied to a photo."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    presets_path = tmp_path / ".photo-studio" / "presets.json"
    presets_path.parent.mkdir(parents=True)
    presets_path.write_text(json.dumps({"Warm": {"exposure": 1.3, "temperature": 40}}), encoding="utf-8")

    at = run_with_uploaded_image()
    assert at.slider(key="t-ev").value == 0.0

    at.selectbox(key="p-load-sel").set_value("Warm")
    at.run(timeout=30)
    at.button(key="p-load-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-ev").value == 0.85
    assert at.slider(key="c-temp").value == 40


def test_delete_preset_button_removes_it_from_the_store(monkeypatch, tmp_path):
    """Same gap as the load-preset test above, for PresetStore.delete()."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    presets_path = tmp_path / ".photo-studio" / "presets.json"
    presets_path.parent.mkdir(parents=True)
    presets_path.write_text(json.dumps({"Keep": {"exposure": 1.0}, "Drop": {"exposure": 1.0}}), encoding="utf-8")

    at = run_with_uploaded_image()
    at.selectbox(key="p-load-sel").set_value("Drop")
    at.run(timeout=30)
    at.button(key="p-delete-btn").click()
    at.run(timeout=30)

    assert not at.exception
    saved = json.loads(presets_path.read_text(encoding="utf-8"))
    assert "Drop" not in saved
    assert "Keep" in saved
    assert any("Deleted preset 'Drop'" in t.value for t in at.toast)


def test_batch_processor_preset_dropdown_lists_saved_custom_presets(monkeypatch, tmp_path):
    """The Batch Processor's Preset dropdown ("sync settings" across a shoot) offers built-in
    named looks and every saved custom preset side by side, the custom ones suffixed " (custom)"
    so Run Batch can tell which lookup (app.presets.PRESETS vs. PresetStore.get()) to use."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    presets_path = tmp_path / ".photo-studio" / "presets.json"
    presets_path.parent.mkdir(parents=True)
    presets_path.write_text(json.dumps({"My Look": {"exposure": 1.2}}), encoding="utf-8")

    at = run_with_uploaded_image()
    at.radio(key="more-mode").set_value("Batch Processor")
    at.run(timeout=30)

    assert not at.exception
    options = at.selectbox(key="b-preset-sel").options
    assert "None" in options
    assert "My Look (custom)" in options
    assert any(name in options for name in PRESETS)


def _capture_download(monkeypatch, key):
    """Records the `data` a st.download_button with `key` is given on each run -- a callable for
    the lazy buttons, which AppTest itself never invokes."""
    import streamlit as st

    captured = {}
    original = st.download_button

    def capture(label, data, *args, **kwargs):
        if kwargs.get("key") == key:
            captured["data"] = data
            captured["disabled"] = kwargs.get("disabled", False)
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture)
    return captured


def test_save_project_is_one_lazy_download_button(monkeypatch):
    """Save Project used to be two steps (package, then a separate download button that vanished
    on the next rerun). It's one download_button now, whose data callable builds the archive only
    when clicked."""
    captured = _capture_download(monkeypatch, "proj-dl-btn")
    at = run_with_uploaded_image()

    assert not at.exception
    assert callable(captured["data"]) and not captured["disabled"]
    restored = ProjectBundle.load(captured["data"]())
    assert restored.source_name == "photo.jpg"


def test_split_view_renders_the_before_after_comparison_without_error():
    """Regression guard for _before_after_html()/components.html() -- AppTest can't inspect
    inside the rendered iframe, but it does catch any exception building or embedding it
    (e.g. a before/after size mismatch after a crop not being reconciled)."""
    at = run_with_uploaded_image()

    at.checkbox(key="cv-split").set_value(True)
    at.run(timeout=30)

    assert not at.exception
    # Split View takes priority over "Original" -- it's disabled while Split View is checked.
    assert at.checkbox(key="cv-orig").disabled


def test_split_view_still_works_after_a_crop_changes_the_edited_size():
    """The before/after comparison must reconcile before/after size itself when a crop makes
    edited_image smaller than source -- otherwise embedding two different-sized previews would
    raise instead of silently misaligning."""
    at = run_with_uploaded_image(size=(40, 30))

    at.selectbox(key="crop-aspect").set_value("1:1 Square")
    at.run(timeout=30)
    at.checkbox(key="cv-split").set_value(True)
    at.run(timeout=30)

    assert not at.exception


def test_free_form_crop_changes_output_dimensions_to_the_selected_box():
    """The on-canvas draggable crop rectangle is JS an AppTest harness can't drive -- this instead
    verifies the Python-side wiring its drag-release commit ultimately depends on: setting the
    hidden field (_crop_editor_html's fractional {x0,y0,x1,y1} JSON) is exactly what a real drag
    sends, just without a browser involved. 0.1..0.6 of a 40px width is 20px; 0.1..0.7 of a 30px
    height is 18px."""
    at = run_with_uploaded_image(size=(40, 30))
    assert canvas_dims(at) == (40, 30)

    at.selectbox(key="crop-aspect").set_value("Free-form")
    at.run(timeout=30)
    at.text_area(key="crop-editor-data").set_value(json.dumps({"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.7}))
    at.run(timeout=30)
    at.run(timeout=30)  # a second rerun with the same committed box must not re-trigger anything odd

    assert not at.exception
    assert canvas_dims(at) == (20, 18)


def test_free_form_crop_box_survives_undo_redo():
    """Regression coverage for the fractional-storage fix -- the old pixel-range-slider version's
    Known Issue was that Free-form's exact box reset to the full frame after an Undo/Redo or a
    project reload, because it was never added to RECIPE_WIDGET_KEYS/RECIPE_DEFAULTS at all."""
    at = run_with_uploaded_image(size=(40, 30))
    at.selectbox(key="crop-aspect").set_value("Free-form")
    at.run(timeout=30)
    at.text_area(key="crop-editor-data").set_value(json.dumps({"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.7}))
    at.run(timeout=30)
    assert canvas_dims(at) == (20, 18)

    at.button(key="hist-undo-btn").click()
    at.run(timeout=30)
    assert not at.exception
    assert canvas_dims(at) == (40, 30)  # back to the full, uncropped frame

    at.button(key="hist-redo-btn").click()
    at.run(timeout=30)
    assert not at.exception
    assert canvas_dims(at) == (20, 18)  # the exact box restored, not reset to full-frame


def test_undo_restores_previous_control_value():
    at = run_with_uploaded_image()

    at.slider(key="t-ev").set_value(0.6)
    at.run(timeout=30)
    assert at.slider(key="t-ev").value == 0.6

    at.button(key="hist-undo-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-ev").value == 0.0


def test_saved_project_keeps_image_layers_as_png_entries_and_text_layers_editable(monkeypatch):
    """B10: layers are saved parametrically -- an Image layer's pixels as their own PNG entry in
    the archive, a Text layer as its text/font/position (it used to be flattened to pixels and
    reopen as an uneditable Image layer)."""
    import zipfile

    captured = _capture_download(monkeypatch, "proj-dl-btn")
    at = run_with_uploaded_image()
    at.get_by_key("l-new-file").set_value(("overlay.png", image_bytes((255, 0, 0), (40, 30)), "image/png"))
    at.run(timeout=30)
    at.radio(key="l-new-kind").set_value("Text")
    at.run(timeout=30)
    at.button(key="l-add-text-btn").click()
    at.run(timeout=30)
    text_id = at.session_state["image_layers"][1]["id"]
    at.text_area(key=f"l-text-{text_id}").set_value("Hello")
    at.run(timeout=30)
    assert not at.exception

    content = captured["data"]()
    names = zipfile.ZipFile(io.BytesIO(content)).namelist()
    assert "layers/stack.json" in names and any(name.startswith("layers/") and name.endswith(".png") for name in names)
    restored = ProjectBundle.load(content)
    kinds = [layer["kind"] for layer in restored.layer_stack]
    assert kinds == ["image", "text"] and restored.layer_stack[1]["text"] == "Hello"

    # ...and reopening it brings the Text layer back as a Text layer.
    at2 = run_with_uploaded_image()
    at2.get_by_key("proj-upload").set_value(("p.photoproject", content, "application/zip"))
    at2.run(timeout=30)
    at2.button(key="proj-load-btn").click()
    at2.run(timeout=30)
    assert not at2.exception
    layers = at2.session_state["image_layers"]
    assert [layer["kind"] for layer in layers] == ["image", "text"]
    assert layers[1]["text"] == "Hello" and layers[0]["image_b64"]


def test_adding_a_corrupted_layer_image_shows_an_error_instead_of_crashing():
    """Layer images are decoded and validated as soon as they're added in the Layers tab
    (_render_layer_stack), not deferred until Save Project -- a corrupted upload must show a
    friendly error right there and never make it into the layer stack, the same "don't leak a raw
    traceback for user-controlled bytes" convention as the main image uploader."""
    valid_png = image_bytes((255, 0, 0), (40, 30))
    truncated = valid_png[: len(valid_png) // 2]

    at = run_with_uploaded_image()
    at.get_by_key("l-new-file").set_value(("overlay.png", truncated, "image/png"))
    at.run(timeout=30)

    assert not at.exception
    assert any("layer image" in message.value.lower() for message in at.error)
    assert at.session_state["image_layers"] == []


def test_load_project_restores_recipe_and_layers_from_a_project_bundle():
    """Regression test for the migration off PhotoProject to ProjectBundle: the overlay layer
    lives in ProjectBundle's structured `layer_workspace` field instead of being base64-embedded in
    the recipe, so the Load handler has to translate it back into the recipe["layers"] shape
    _restore_recipe_to_state() expects (a list, even for this old-format file's single layer)."""
    source = Image.new("RGB", (40, 30), (80, 80, 80))
    overlay = Image.new("RGBA", (40, 30), (255, 0, 0, 255))
    workspace = LayerWorkspace(source).add(WorkspaceLayer("Overlay", overlay, 0.7, "multiply"))
    source_buffer = io.BytesIO()
    source.save(source_buffer, format="PNG")

    bundle_buffer = io.BytesIO()
    ProjectBundle("saved.png", source_buffer.getvalue(), {"exposure": 1.3}, workspace.to_json()).save(bundle_buffer)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.get_by_key("proj-upload").set_value(("saved.photoproject", bundle_buffer.getvalue(), "application/zip"))
    at.run(timeout=30)
    at.button(key="proj-load-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["restored_source_name"] == "saved.png"
    assert at.slider(key="t-ev").value == 0.85
    restored_layers = at.session_state["image_layers"]
    assert len(restored_layers) == 1
    assert restored_layers[0]["opacity"] == 0.7
    assert restored_layers[0]["blend_mode"] == "multiply"
    restored_overlay = Image.open(io.BytesIO(base64.b64decode(restored_layers[0]["image_b64"])))
    assert restored_overlay.size == (40, 30)


def test_load_project_still_reads_the_legacy_photoproject_format():
    """ProjectBundle.load() must stay able to open files saved by the older, simpler PhotoProject
    format -- users may still have .photoproject files from before this migration."""
    source = Image.new("RGB", (20, 16), (30, 60, 90))
    source_buffer = io.BytesIO()
    source.save(source_buffer, format="PNG")

    legacy_buffer = io.BytesIO()
    PhotoProject("legacy.png", source_buffer.getvalue(), {"exposure": 1.4}).save(legacy_buffer)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.get_by_key("proj-upload").set_value(("legacy.photoproject", legacy_buffer.getvalue(), "application/zip"))
    at.run(timeout=30)
    at.button(key="proj-load-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["restored_source_name"] == "legacy.png"
    assert at.slider(key="t-ev").value == 1.05


def test_load_project_with_a_corrupted_file_shows_an_error_instead_of_crashing():
    """A hand-edited or corrupted .photoproject must not crash the app -- every other user-input
    path in this file (RAW load, backend calls, curve JSON) already degrades to a friendly
    st.error() instead of an unhandled traceback, and Load Project previously had no such guard."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.get_by_key("proj-upload").set_value(("broken.photoproject", b"not actually a zip file", "application/zip"))
    at.run(timeout=30)
    at.button(key="proj-load-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("Could not open this project file" in message.value for message in at.error)
    assert "restored_source_name" not in at.session_state


def test_hsl_color_mixer_defaults_to_no_adjustment():
    at = run_with_uploaded_image()

    for band in ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"):
        assert at.slider(key=f"hsl-{band}-h").value == 0
        assert at.slider(key=f"hsl-{band}-s").value == 0
        assert at.slider(key=f"hsl-{band}-l").value == 0


def test_hsl_color_mixer_changes_the_final_image_and_persists_across_bands():
    """Every band's sliders render unconditionally every rerun (not behind a "currently selected
    band" picker) specifically so Streamlit doesn't garbage-collect an untouched band's
    session_state value -- see the comment above this UI in frontend_unified_app.py for why an
    earlier version of this using a selectbox lost values that way. Uses a saturated reddish base
    color (not neutral gray), since a Red-band saturation cut has nothing to do on a pixel that's
    already fully desaturated."""
    at = run_with_uploaded_image(color=(200, 60, 60))
    baseline = canvas_hash(at)

    at.slider(key="hsl-red-s").set_value(-80)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline
    red_only_hash = canvas_hash(at)

    at.slider(key="hsl-blue-h").set_value(40)
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="hsl-red-s").value == -80  # Red's value survived Blue's change
    # Blue's hue shift correctly has no further effect here -- the base color (200, 60, 60) is
    # pure red, ~120 degrees from Blue's hue center, well outside the mixer's falloff radius; see
    # test_professional_color.py for coverage of a band actually changing a matching-hue pixel.
    assert canvas_hash(at) == red_only_hash


def test_hsl_color_mixer_is_included_in_reset():
    at = run_with_uploaded_image()
    at.slider(key="hsl-red-s").set_value(-80)
    at.run(timeout=30)
    assert at.slider(key="hsl-red-s").value == -80

    at.button(key="hist-reset-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="hsl-red-s").value == 0


def test_gamut_warning_shows_a_no_profile_message_for_a_plain_test_image():
    """The shared image_bytes() test helper doesn't embed an ICC profile (matching most synthetic
    test fixtures), so the "nothing to proof" message is the expected default state."""
    at = run_with_uploaded_image()
    assert any("no embedded color profile" in c.value for c in at.caption)


def test_gamut_warning_checkbox_renders_the_overlay_for_a_profiled_image():
    from app.professional_color import srgb_icc_bytes

    buf = io.BytesIO()
    Image.new("RGB", (20, 20), (180, 90, 40)).save(buf, format="PNG", icc_profile=srgb_icc_bytes())
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("profiled.png", buf.getvalue(), "image/png"))
    at.run(timeout=30)

    at.checkbox(key="gamut-warning-toggle").set_value(True)
    at.run(timeout=30)

    assert not at.exception
    assert any(img.captions and img.captions[0] == "Magenta = clipped when converted to this app's sRGB working space" for img in at.image)


def test_vibrance_defaults_to_no_adjustment():
    at = run_with_uploaded_image()
    assert at.slider(key="c-vib").value == 0


def test_vibrance_slider_changes_the_final_image():
    at = run_with_uploaded_image(color=(200, 160, 160))
    baseline = canvas_hash(at)

    at.slider(key="c-vib").set_value(90)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_vibrance_is_included_in_reset():
    at = run_with_uploaded_image()
    at.slider(key="c-vib").set_value(60)
    at.run(timeout=30)
    assert at.slider(key="c-vib").value == 60

    at.button(key="hist-reset-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="c-vib").value == 0


def test_white_balance_picker_commits_via_the_hidden_click_field():
    """Same JS-can't-be-driven-by-AppTest limitation as the Magic Wand picker (see
    test_magic_wand_selection_commits_via_the_hidden_click_field) -- this instead verifies the
    Python-side wiring the picker's click handler ultimately depends on: setting the hidden field
    is exactly what a real click sends, just without a browser involved."""
    at = run_with_uploaded_image(color=(180, 140, 90), size=(20, 20))  # a warm cast to correct

    at.text_area(key="wb-wand-data").set_value(json.dumps({"x": 0.5, "y": 0.5}))
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="c-temp").value < 0  # cooled to counteract the warm cast


def test_color_grading_defaults_to_no_adjustment():
    at = run_with_uploaded_image()
    assert at.slider(key="cg-sh-int").value == 0
    assert at.slider(key="cg-mid-int").value == 0
    assert at.slider(key="cg-hi-int").value == 0
    assert at.slider(key="cg-balance").value == 0


def test_color_grading_shadows_changes_the_final_image():
    at = run_with_uploaded_image(color=(20, 20, 20))
    baseline = canvas_hash(at)

    at.slider(key="cg-sh-hue").set_value(220)
    at.run(timeout=30)
    at.slider(key="cg-sh-int").set_value(100)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_color_grading_is_included_in_reset():
    at = run_with_uploaded_image()
    at.slider(key="cg-hi-int").set_value(60)
    at.run(timeout=30)
    assert at.slider(key="cg-hi-int").value == 60

    at.button(key="hist-reset-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="cg-hi-int").value == 0


def test_dehaze_defaults_to_no_adjustment():
    at = run_with_uploaded_image()
    assert at.slider(key="d-haze").value == 0


def test_dehaze_slider_changes_the_final_image():
    """Dehaze recovers lost local contrast, so it's a near-identity transform on a perfectly flat
    test image (nothing to recover) -- unlike most other sliders in this file, this needs an
    upload with actual internal contrast, the same synthetic "patch buried in uniform haze" shape
    used in tests/test_backend.py and tests/test_dehaze.py."""
    hazy = np.full((40, 40, 3), 190, dtype=np.uint8)
    hazy[10:30, 10:30] = 90
    buf = io.BytesIO()
    Image.fromarray(hazy, "RGB").save(buf, format="PNG")

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("hazy.png", buf.getvalue(), "image/png"))
    at.run(timeout=30)
    baseline = canvas_hash(at)

    at.slider(key="d-haze").set_value(80)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_dehaze_is_included_in_reset():
    at = run_with_uploaded_image()
    at.slider(key="d-haze").set_value(50)
    at.run(timeout=30)
    assert at.slider(key="d-haze").value == 50

    at.button(key="hist-reset-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="d-haze").value == 0


def test_color_noise_slider_is_disabled_until_denoise_is_nonzero():
    at = run_with_uploaded_image()
    assert at.slider(key="d-den-color").value == 0
    assert at.slider(key="d-den-color").disabled is True

    at.slider(key="d-den").set_value(4)
    at.run(timeout=30)

    assert at.slider(key="d-den-color").disabled is False


def test_color_noise_slider_changes_the_final_image_independently_of_denoise():
    rng = np.random.default_rng(3)
    noisy = np.clip(np.full((40, 40, 3), 128, dtype=np.float32) + rng.normal(0, 20, (40, 40, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(noisy, "RGB").save(buf, format="PNG")

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("noisy.png", buf.getvalue(), "image/png"))
    at.run(timeout=30)
    at.slider(key="d-den").set_value(4)
    at.run(timeout=30)
    baseline = canvas_hash(at)

    at.slider(key="d-den-color").set_value(9)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_post_crop_vignette_and_grain_default_to_no_adjustment():
    at = run_with_uploaded_image()
    assert at.slider(key="fx-vignette-amt").value == 0
    assert at.slider(key="fx-vignette-mid").value == 50
    assert at.slider(key="fx-grain-amt").value == 0


def test_post_crop_vignette_changes_the_final_image():
    at = run_with_uploaded_image(color=(200, 200, 200), size=(40, 40))
    baseline = canvas_hash(at)

    at.slider(key="fx-vignette-amt").set_value(-80)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_film_grain_changes_the_final_image():
    at = run_with_uploaded_image(color=(120, 120, 120))
    baseline = canvas_hash(at)

    at.slider(key="fx-grain-amt").set_value(80)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_effects_are_included_in_reset():
    at = run_with_uploaded_image()
    at.slider(key="fx-vignette-amt").set_value(-60)
    at.slider(key="fx-grain-amt").set_value(50)
    at.run(timeout=30)
    assert at.slider(key="fx-vignette-amt").value == -60

    at.button(key="hist-reset-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="fx-vignette-amt").value == 0
    assert at.slider(key="fx-grain-amt").value == 0


def test_clipping_warning_checkbox_renders_the_overlay():
    at = run_with_uploaded_image()
    at.checkbox(key="clip-warning-toggle").set_value(True)
    at.run(timeout=30)

    assert not at.exception
    assert any(img.captions and "clipped" in img.captions[0] for img in at.image)


def test_clipping_margin_is_disabled_until_the_warning_is_on():
    at = run_with_uploaded_image()
    assert at.slider(key="clip-warning-margin").disabled
    at.checkbox(key="clip-warning-toggle").set_value(True)
    at.run(timeout=30)
    at.slider(key="clip-warning-margin").set_value(10)
    at.run(timeout=30)

    assert not at.exception
    assert not at.slider(key="clip-warning-margin").disabled
    assert at.slider(key="clip-warning-margin").value == 10


def test_magic_wand_selection_commits_via_the_hidden_click_field():
    """The on-canvas click-to-sample picker is JS an AppTest harness can't drive (no real DOM) --
    this instead verifies the Python-side wiring its commit ultimately depends on: setting the
    hidden text field is exactly what the picker's click handler does, just without a real click
    involved (the same pattern test_painted_mask_selection_changes_the_final_image_via_masks_tab
    uses for the paint editor)."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.selectbox(key="m-new-kind").set_value("Magic Wand")
    at.run(timeout=30)
    at.text_area(key="m-wand-data").set_value(json.dumps({"x": 0.5, "y": 0.5}))
    at.run(timeout=30)
    at.run(timeout=30)  # a second rerun with the same committed click must not add a duplicate layer

    assert not at.exception
    layers = at.session_state["mask_layers"]
    assert len(layers) == 1
    assert layers[0]["kind"] == "Magic Wand"
    assert layers[0]["params"]["click_x"] == 0.5
    assert layers[0]["params"]["click_y"] == 0.5

    at.slider(key="m-tev").set_value(1.3)
    at.run(timeout=30)
    assert canvas_hash(at) != baseline


def test_load_preset_section_shows_a_thumbnail_preview(monkeypatch, tmp_path):
    """Regression test for a real UI gap found live: unlike the 4 built-in looks (each already
    shown via _preset_thumbnails()), a saved custom preset had no visual preview anywhere -- you
    had to load it blind to see what it did. Covers _custom_preset_thumbnail() wired into the
    Presets tab's "Load a Preset" section."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    presets_path = tmp_path / ".photo-studio" / "presets.json"
    presets_path.parent.mkdir(parents=True)
    presets_path.write_text(json.dumps({"Warm": {"exposure": 1.3, "temperature": 40}}), encoding="utf-8")

    at = run_with_uploaded_image()
    open_tab(at, "✦ Presets & Library", "✦ Presets")  # thumbnails only render in the visible tab (F1)
    at.selectbox(key="p-load-sel").set_value("Warm")
    at.run(timeout=30)

    assert not at.exception
    assert any(img.captions and img.captions[0] == "Warm" for img in at.image)


def test_batch_processor_shows_a_thumbnail_preview_for_a_selected_custom_preset(monkeypatch, tmp_path):
    """Same gap as the Presets tab's "Load a Preset" preview, for the Batch Processor's preset
    dropdown -- only custom presets lacked a preview (the 4 built-in looks are self-explanatory by
    name), so the thumbnail should appear only once a " (custom)" option is actually selected."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    presets_path = tmp_path / ".photo-studio" / "presets.json"
    presets_path.parent.mkdir(parents=True)
    presets_path.write_text(json.dumps({"My Look": {"exposure": 1.2}}), encoding="utf-8")

    at = run_with_uploaded_image()
    at.radio(key="more-mode").set_value("Batch Processor")
    at.run(timeout=30)
    images_before = len(at.image)

    at.selectbox(key="b-preset-sel").set_value("My Look (custom)")
    at.run(timeout=30)

    assert not at.exception
    assert len(at.image) == images_before + 1


def test_run_batch_shows_progress_and_a_done_summary(monkeypatch, tmp_path):
    """Regression test for a real gap found live: process_files() already accepted a `progress`
    callback and was tested for it in tests/test_batch_release.py, but the UI never passed one in
    -- Run Batch gave no feedback at all until the whole batch finished. Confirms the progress
    status reaches a final "Done" summary and the results table + download button render."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    at = run_with_uploaded_image()
    at.radio(key="more-mode").set_value("Batch Processor")
    at.run(timeout=30)

    at.get_by_key("b-up").set_value([("one.jpg", image_bytes(), "image/jpeg"), ("two.jpg", image_bytes(), "image/jpeg")])
    at.run(timeout=30)
    at.button(key="b-run-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("Done — 2/2 succeeded" in c.value for c in at.caption)
    assert at.dataframe
    assert any(db.label == "Download ZIP" for db in at.download_button)


def test_scan_folder_shows_progress_and_a_done_summary(monkeypatch, tmp_path):
    """Same gap as Run Batch above, for the Library Catalog's "Scan Folder" button:
    app.library.index_folder() already accepted a `progress` callback (see
    tests/test_library.py), but the UI never passed one in -- scanning a large folder gave no
    feedback until it finished entirely."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(photos_dir / "one.jpg")
    Image.new("RGB", (8, 8), (40, 50, 60)).save(photos_dir / "two.jpg")

    at = run_with_uploaded_image()
    # "Library Catalog" is the radio's default (first) option -- no need to select it explicitly,
    # unlike the Batch Processor tests above which switch away from that default.
    at.text_input(key="lib-f-in").set_value(str(photos_dir))
    at.run(timeout=30)
    at.button(key="lib-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("Done — indexed 2 photo(s)" in c.value for c in at.caption)
    assert any("Indexed 2 photo(s)" in t.value for t in at.toast)


def test_library_search_shows_a_no_results_message(monkeypatch, tmp_path):
    """Regression test for a real UI gap found live: an empty search or a Min Stars filter that
    matched nothing just rendered a blank table with no explanation -- easy to mistake for the
    scan itself having found nothing. Seeds the catalog directly via PhotoCatalog.upsert() (what a
    real Scan Folder run would have already done) rather than scanning a real folder."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    lib_db = tmp_path / ".photo-studio" / "catalog.sqlite"
    lib_db.parent.mkdir(parents=True)
    PhotoCatalog(lib_db).upsert("photo-a.jpg")

    at = run_with_uploaded_image()
    at.session_state["records"] = [{"path": "photo-a.jpg", "name": "photo-a.jpg", "width": 1, "height": 1, "format": "JPEG", "bytes": 1}]
    at.run(timeout=30)

    assert not any("No photos match" in c.value for c in at.caption)

    at.text_input(key="lib-q-in").set_value("no-such-photo")
    at.run(timeout=30)

    assert not at.exception
    assert any("No photos match this search / star filter." in c.value for c in at.caption)


def test_library_rate_and_flag_panel_shows_its_keyboard_shortcuts(monkeypatch, tmp_path):
    """Same gap as the search caption above: the Library Catalog's 1-5/0/X/arrow shortcuts (see
    _LIBRARY_KEYBOARD_SHORTCUTS_HTML) existed only in the docs, with nothing on screen telling a
    user they're there."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    at = run_with_uploaded_image()
    at.session_state["records"] = [{"path": "photo-a.jpg", "name": "photo-a.jpg", "width": 1, "height": 1, "format": "JPEG", "bytes": 1}]
    at.run(timeout=30)

    assert not at.exception
    assert any("1-5 rate" in c.value and "Prev/Next" in c.value for c in at.caption)


def test_magic_wand_selection_removes_content_via_repair_tab():
    """Unlike the flat-color fixture most tests here reuse, this needs an actual color patch to
    click on: a click on a perfectly uniform image would select the *entire* frame (nothing is
    outside the color tolerance), leaving cv2.inpaint no unselected pixels to repair from, which
    is a legitimate degenerate case but not what this test means to exercise."""
    base = Image.new("RGB", (40, 30), (80, 80, 80))
    ImageDraw.Draw(base).rectangle((15, 10, 25, 20), fill=(220, 30, 30))
    buf = io.BytesIO()
    base.save(buf, format="PNG")

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("patch.png", buf.getvalue(), "image/png"))
    at.run(timeout=30)
    baseline = canvas_hash(at)

    at.selectbox(key="rm-new-kind").set_value("Magic Wand")
    at.run(timeout=30)
    at.text_area(key="rm-wand-data").set_value(json.dumps({"x": 0.5, "y": 0.5}))  # center of the patch
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["removal_mask_layers"][0]["kind"] == "Magic Wand"
    assert canvas_hash(at) != baseline


def test_lens_corrections_default_to_no_adjustment():
    at = run_with_uploaded_image()

    assert at.selectbox(key="lens-profile").value == "None"
    assert at.slider(key="lens-distortion").value == 0
    assert at.slider(key="lens-vignette-amt").value == 0
    assert at.slider(key="lens-vignette-mid").value == 50
    assert at.slider(key="lens-ca").value == 0


def test_lens_correction_slider_changes_the_final_image():
    at = run_with_uploaded_image(color=(200, 200, 200))
    baseline = canvas_hash(at)

    at.slider(key="lens-vignette-amt").set_value(-80)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_lens_profile_apply_button_fills_in_the_sliders():
    at = run_with_uploaded_image()

    at.selectbox(key="lens-profile").set_value("Fisheye")
    at.run(timeout=30)
    at.button(key="lens-apply-profile-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="lens-distortion").value == -60
    assert at.slider(key="lens-vignette-amt").value == -35
    assert at.slider(key="lens-vignette-mid").value == 40
    assert at.slider(key="lens-ca").value == 20


def test_lens_corrections_are_included_in_reset():
    at = run_with_uploaded_image()
    at.slider(key="lens-distortion").set_value(40)
    at.run(timeout=30)
    assert at.slider(key="lens-distortion").value == 40

    at.button(key="hist-reset-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="lens-distortion").value == 0


def test_lens_perspective_defaults_to_no_adjustment():
    at = run_with_uploaded_image()
    assert at.slider(key="lens-persp-v").value == 0
    assert at.slider(key="lens-persp-h").value == 0


def test_lens_perspective_slider_changes_the_final_image():
    """Unlike distortion (which visibly changes even a flat-color image, since cv2.undistort's
    default border fill is black), correct_perspective() uses BORDER_REPLICATE -- a perfectly flat
    image warps back to the exact same flat image, nothing to detect. Needs actual spatial
    variation, the same reason test_dehaze_slider_changes_the_final_image needs a patterned
    fixture instead of the flat-color image_bytes() convenience most tests here use."""
    grid = Image.new("RGB", (60, 60), (30, 30, 30))
    ImageDraw.Draw(grid).line([(30, 0), (30, 60)], fill=(220, 220, 220), width=2)
    buf = io.BytesIO()
    grid.save(buf, format="PNG")

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("grid.png", buf.getvalue(), "image/png"))
    at.run(timeout=30)
    baseline = canvas_hash(at)

    at.slider(key="lens-persp-v").set_value(60)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_lens_perspective_is_included_in_reset():
    at = run_with_uploaded_image()
    at.slider(key="lens-persp-h").set_value(50)
    at.run(timeout=30)
    assert at.slider(key="lens-persp-h").value == 50

    at.button(key="hist-reset-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="lens-persp-h").value == 0


# The two regression tests that used to live here (test_crop_preview_reflects_lens_distortion_
# correction, test_crop_preview_reflects_lens_perspective_correction) asserted on a Crop-tab
# st.image's rendered bytes to confirm the free-form crop preview was built from mask_source (the
# lens-corrected geometry) rather than the raw, uncorrected source -- otherwise a crop box framed
# against the wrong-looking preview could land on the wrong region once the backend actually warps
# the image. The draggable on-canvas crop editor (_crop_editor_html, replacing the old plain X/Y
# range sliders + st.image preview pair) still deliberately passes it
# preview_image(mask_source, (480, 480)) for exactly the same reason, but that background image
# now lives inside a components.html() iframe's srcdoc, which streamlit.testing.v1.AppTest has no
# API to inspect (no real DOM, and there's no safe way to import app.frontend_unified_app directly
# to monkeypatch around it -- see its own module docstring: it's a Streamlit script with top-level
# st.* calls meant to run under AppTest/`streamlit run`, not to be imported as a plain library).
# This is now covered by live manual verification only, the same documented limitation already
# true of the Library Catalog's keyboard shortcuts (see "Known Issues" in docs/PROJECT_STATUS.md).


def _canvas_iframe_images(at) -> list[str]:
    """The base64 PNGs inside the components.html() iframes on the page (what an on-canvas view
    draws: AppTest has no DOM, but the iframe's srcdoc is in its proto)."""
    return [image for frame in at.get("iframe") for image in re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", frame.proto.srcdoc)]


def test_mask_overlay_preview_reflects_lens_distortion_correction():
    """Same bug class as test_crop_preview_reflects_lens_distortion_correction, but for the
    Masks tab's "Preview combined mask overlay" view, which used to build the overlay from the raw
    `source` regardless of the Lens tab's sliders. Ticking it now draws it over the photo."""
    def run_in_local_tab():
        open_tab(at, "◉ Local")  # the tab must be open for its views to be offered over the photo
        at.run(timeout=30)

    at = run_with_uploaded_image(color=(80, 80, 80))
    at.selectbox(key="m-new-kind").set_value("Radial")
    run_in_local_tab()
    at.button(key="m-add-btn").click()
    run_in_local_tab()
    at.checkbox(key="m-preview-overlay").set_value(True)
    run_in_local_tab()
    assert at.session_state["canvas-mode"] == "🟥 Mask preview"
    baseline = _canvas_iframe_images(at)
    assert baseline

    at.slider(key="lens-distortion").set_value(80)
    run_in_local_tab()

    assert not at.exception
    assert at.session_state["canvas-mode"] == "🟥 Mask preview"
    assert _canvas_iframe_images(at) != baseline


def test_rgb_curve_defaults_to_identity_and_is_off():
    at = run_with_uploaded_image()

    assert at.checkbox(key="c-curve-en").value is False
    assert at.slider(key="c-p0").value == 0
    assert at.slider(key="c-p1").value == 64
    assert at.slider(key="c-p2").value == 128
    assert at.slider(key="c-p3").value == 192
    assert at.slider(key="c-p4").value == 255


def test_rgb_curve_changes_the_final_image_once_enabled():
    """The draggable curve editor (a components.html() canvas) doesn't touch these sliders
    directly -- it writes into a hidden text_area instead (see the next test) -- but this still
    verifies the Python-side wiring the recipe ultimately depends on: enabling the curve and
    moving a point reaches the final image, regardless of how that value got set."""
    at = run_with_uploaded_image()
    before = canvas_hash(at)

    at.checkbox(key="c-curve-en").set_value(True)
    at.slider(key="c-p2").set_value(180)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != before


def test_rgb_curve_drag_commit_via_hidden_field_changes_the_final_image():
    """Regression test for a real bug: the curve editor originally wrote a dragged point straight
    into the real <input type="range"> via a native-value-setter + dispatch('input'/'change')
    trick. Confirmed live (comparing the actual rendered image, not just the slider's displayed
    number) that this does NOT reliably reach Streamlit's backend for range sliders in this
    Streamlit version -- the number visibly updates as a client-side echo while the value the
    script actually sees next run can still be the old one. Fixed by writing the whole curve as
    JSON into a hidden st.text_area instead and routing it through the same
    st.session_state["_pending_restore"] + st.rerun() mechanism Undo/Redo/Reset already use --
    this test drives exactly that hidden field, the same thing the editor's JS does on drag
    release, and confirms it reaches the final image."""
    at = run_with_uploaded_image()
    before = canvas_hash(at)

    at.text_area(key="curve-commit-data").set_value(
        json.dumps({"curve_black": 0, "curve_shadows": 64, "curve_mid": 180, "curve_highlights": 192, "curve_white": 255, "curve_enabled": True})
    )
    at.run(timeout=30)

    assert not at.exception
    assert at.checkbox(key="c-curve-en").value is True
    assert at.slider(key="c-p2").value == 180
    assert canvas_hash(at) != before


def test_rgb_curve_extra_point_commit_changes_the_final_image():
    """Same hidden-field simulation as the drag-commit test above, now covering an added "extra"
    point (click-to-add on the curve, not one of the 5 fixed Blacks/Shadows/Midtones/Highlights/
    Whites sliders) -- the feature this app's Known Issues used to call a UI-only limitation of
    PhotoProcessor.apply_curves(), which already accepted an arbitrary point list. color=(50,50,50)
    deliberately: the extra point added below sits strictly between the identity curve's 0 and 64
    fixed points, so a fixture whose value falls outside that 0-64 span (e.g. the default flat
    image_bytes() color) would show no change at all -- the reshaping is local to that segment."""
    at = run_with_uploaded_image(color=(50, 50, 50))
    before = canvas_hash(at)

    at.text_area(key="curve-commit-data").set_value(
        json.dumps({
            "curve_black": 0, "curve_shadows": 64, "curve_mid": 128, "curve_highlights": 192, "curve_white": 255,
            "curve_enabled": True, "curve_extra_points": [[30, 220]],
        })
    )
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["curve_extra_points"] == [[30, 220]]
    assert canvas_hash(at) != before


def test_rgb_curve_extra_points_are_included_in_reset():
    at = run_with_uploaded_image()
    at.text_area(key="curve-commit-data").set_value(
        json.dumps({
            "curve_black": 0, "curve_shadows": 64, "curve_mid": 128, "curve_highlights": 192, "curve_white": 255,
            "curve_enabled": True, "curve_extra_points": [[30, 220]],
        })
    )
    at.run(timeout=30)
    assert at.session_state["curve_extra_points"] == [[30, 220]]

    at.button(key="hist-reset-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["curve_extra_points"] == []


def test_library_rating_buttons_write_to_the_catalog(monkeypatch, tmp_path):
    """Seeds session_state.records directly (the same shape index_folder() produces) instead of
    scanning a real folder, and redirects Path.home() so PhotoCatalog resolves to a throwaway
    catalog.sqlite instead of the real ~/.photo-studio/ on whatever machine runs the suite."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    at = run_with_uploaded_image()
    at.radio(key="more-mode").set_value("Library Catalog")
    at.session_state["records"] = [
        {"path": "photo-a.jpg", "name": "photo-a.jpg", "width": 10, "height": 10, "format": "JPEG", "bytes": 100},
        {"path": "photo-b.jpg", "name": "photo-b.jpg", "width": 10, "height": 10, "format": "JPEG", "bytes": 100},
    ]
    at.run(timeout=30)

    at.button(key="lib-star-4").click()
    at.run(timeout=30)

    assert not at.exception
    catalog = PhotoCatalog(tmp_path / ".photo-studio" / "catalog.sqlite")
    assert catalog.get("photo-a.jpg")["rating"] == 4
    # Rating the first (current) record must not touch the second.
    assert catalog.get("photo-b.jpg")["rating"] == 0


def test_library_reject_toggles_a_tag_without_changing_rating(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    at = run_with_uploaded_image()
    at.radio(key="more-mode").set_value("Library Catalog")
    at.session_state["records"] = [{"path": "photo-a.jpg", "name": "photo-a.jpg", "width": 1, "height": 1, "format": "JPEG", "bytes": 1}]
    at.run(timeout=30)

    at.button(key="lib-star-3").click()
    at.run(timeout=30)
    at.button(key="lib-reject-btn").click()
    at.run(timeout=30)

    catalog = PhotoCatalog(tmp_path / ".photo-studio" / "catalog.sqlite")
    record = catalog.get("photo-a.jpg")
    assert "rejected" in record["tags"]
    assert record["rating"] == 3

    at.button(key="lib-reject-btn").click()
    at.run(timeout=30)

    assert "rejected" not in catalog.get("photo-a.jpg")["tags"]


def test_library_next_and_prev_cycle_through_records(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    at = run_with_uploaded_image()
    at.radio(key="more-mode").set_value("Library Catalog")
    at.session_state["records"] = [
        {"path": "a.jpg", "name": "a.jpg", "width": 1, "height": 1, "format": "JPEG", "bytes": 1},
        {"path": "b.jpg", "name": "b.jpg", "width": 1, "height": 1, "format": "JPEG", "bytes": 1},
    ]
    at.run(timeout=30)
    assert any("1 / 2" in c.value for c in at.caption)

    at.button(key="lib-next-btn").click()
    at.run(timeout=30)
    assert any("2 / 2" in c.value for c in at.caption)

    at.button(key="lib-prev-btn").click()
    at.run(timeout=30)
    assert any("1 / 2" in c.value for c in at.caption)


def test_library_next_button_does_not_reset_canvas_view_settings(monkeypatch, tmp_path):
    """Regression test for a real bug found in review: Library Catalog's Prev/Next/rating/Clear/
    Reject buttons used to call st.rerun() directly instead of _request_rerun() -- since this tab
    renders before canvas_col, that aborted the script before canvas_col's zoom/split-view/pan/
    export widgets were instantiated that pass, resetting them."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    at = run_with_uploaded_image()
    at.selectbox(key="cv-zoom").set_value(2.0)
    at.run(timeout=30)
    assert at.selectbox(key="cv-zoom").value == 2.0

    at.radio(key="more-mode").set_value("Library Catalog")
    at.session_state["records"] = [
        {"path": "a.jpg", "name": "a.jpg", "width": 1, "height": 1, "format": "JPEG", "bytes": 1},
        {"path": "b.jpg", "name": "b.jpg", "width": 1, "height": 1, "format": "JPEG", "bytes": 1},
    ]
    at.run(timeout=30)

    at.button(key="lib-next-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("2 / 2" in c.value for c in at.caption)
    assert at.selectbox(key="cv-zoom").value == 2.0


# --- Named Snapshots -------------------------------------------------------------------------

def test_save_snapshot_without_a_name_shows_a_warning_toast():
    """Same safe-to-click empty-name guard as "Save Current Look as Preset" -- nothing is
    written to history.named_snapshots."""
    at = run_with_uploaded_image()

    at.button(key="snap-save-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("name" in t.value.lower() for t in at.toast)
    assert at.session_state["edit_history"].snapshot_names == []


def test_save_and_restore_named_snapshot_recovers_the_saved_control_value():
    at = run_with_uploaded_image()
    at.slider(key="t-ev").set_value(0.8)
    at.run(timeout=30)

    at.text_input(key="snap-name-in").set_value("My Look")
    at.run(timeout=30)
    at.button(key="snap-save-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("My Look" in t.value for t in at.toast)
    assert "My Look" in at.session_state["edit_history"].snapshot_names

    at.slider(key="t-ev").set_value(0.0)
    at.run(timeout=30)
    assert at.slider(key="t-ev").value == 0.0

    at.button(key="snap-restore-My Look").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-ev").value == 0.8


def test_restoring_a_snapshot_does_not_remove_it_and_undo_still_reaches_the_step_before_the_restore():
    """A snapshot is a bookmark, not a one-time jump (see app/edit_history.py's
    restore_snapshot() docstring): restoring it must not delete it from the saved list, and must
    not disturb the separate linear Undo/Redo step list beyond acting like any other new edit --
    Undo right after a restore goes back to the value just before the restore, the same as it
    would after applying a preset."""
    at = run_with_uploaded_image()
    at.slider(key="t-ev").set_value(0.8)
    at.run(timeout=30)
    at.text_input(key="snap-name-in").set_value("Bookmark")
    at.run(timeout=30)
    at.button(key="snap-save-btn").click()
    at.run(timeout=30)

    at.slider(key="t-ev").set_value(0.0)
    at.run(timeout=30)

    at.button(key="snap-restore-Bookmark").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-ev").value == 0.8
    assert "Bookmark" in at.session_state["edit_history"].snapshot_names  # still there, not consumed

    at.button(key="hist-undo-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-ev").value == 0.0


def test_delete_snapshot_removes_it_from_the_saved_list():
    at = run_with_uploaded_image()
    at.text_input(key="snap-name-in").set_value("Temp")
    at.run(timeout=30)
    at.button(key="snap-save-btn").click()
    at.run(timeout=30)
    assert "Temp" in at.session_state["edit_history"].snapshot_names

    at.button(key="snap-delete-Temp").click()
    at.run(timeout=30)

    assert not at.exception
    assert "Temp" not in at.session_state["edit_history"].snapshot_names


def test_named_snapshots_live_in_the_canvas_header_popover_not_the_presets_tab():
    """A prior UI review found Snapshots buried at the bottom of the Presets tab (10th of 11
    tabs), below preset import/export -- Undo/Redo/Reset were already moved to the canvas header
    to fix this exact "buried in Presets" problem (see docs/PROJECT_STATUS.md's history), but
    Snapshots wasn't, until now. This asserts the "**Named Snapshots**" heading and its
    save/restore/delete widgets render exactly once, and that the Presets tab now only points at
    the canvas header instead of duplicating the whole panel."""
    at = run_with_uploaded_image()

    named_snapshot_headings = [m.value for m in at.markdown if m.value == "**Named Snapshots**"]
    assert len(named_snapshot_headings) == 1

    # The save control (text input + button) still exists and still works -- it just isn't
    # inside tab_presets's own body anymore, it's reachable via the canvas header popover.
    assert at.text_input(key="snap-name-in") is not None
    assert at.button(key="snap-save-btn") is not None

    # The Presets tab keeps a short pointer instead of duplicating the whole panel.
    pointer_captions = [c.value for c in at.caption if "Named Snapshots are always available" in c.value]
    assert pointer_captions, "expected the Presets tab to point at the relocated Snapshots panel"


# --- Waveform monitor --------------------------------------------------------------------------

def test_waveform_monitor_is_hidden_until_toggled_on_and_reflects_the_selected_channel_mode():
    at = run_with_uploaded_image()
    assert not any(img.captions and img.captions[0].startswith("Waveform") for img in at.image)

    at.checkbox(key="waveform-toggle").set_value(True)
    at.run(timeout=30)

    assert not at.exception
    assert any(img.captions and img.captions[0] == "Waveform (Luma)" for img in at.image)

    at.radio(key="waveform-mode").set_value("RGB")
    at.run(timeout=30)

    assert not at.exception
    assert any(img.captions and img.captions[0] == "Waveform (RGB)" for img in at.image)


def test_waveform_mode_choice_survives_toggling_the_panel_off_and_back_on():
    """The channel-mode radio is always instantiated (even while the panel is hidden) precisely
    so toggling the panel off and back on doesn't silently reset it to its default -- the
    conditional-widget-loses-state class of bug CLAUDE.md documents for this codebase."""
    at = run_with_uploaded_image()
    at.checkbox(key="waveform-toggle").set_value(True)
    at.run(timeout=30)
    at.radio(key="waveform-mode").set_value("RGB")
    at.run(timeout=30)
    at.checkbox(key="waveform-toggle").set_value(False)
    at.run(timeout=30)
    at.checkbox(key="waveform-toggle").set_value(True)
    at.run(timeout=30)

    assert not at.exception
    assert at.radio(key="waveform-mode").value == "RGB"
    assert any(img.captions and img.captions[0] == "Waveform (RGB)" for img in at.image)


def test_waveform_monitor_has_a_visible_caption_not_just_a_hover_tooltip():
    """The waveform toggle's help= tooltip already explained it well, but a prior UI review
    found it easy to miss since nothing was visible without hovering. A short st.caption right
    above the checkbox row should now say so up front."""
    at = run_with_uploaded_image()

    assert any("Waveform monitor" in c.value for c in at.caption)
    assert at.checkbox(key="waveform-toggle").help


# --- Export watermarking ------------------------------------------------------------------------

def test_watermark_controls_render_and_setting_text_does_not_break_the_single_format_export():
    """AppTest can't read a download_button's actual file bytes back out (it only exposes
    whether it was clicked), so this -- like the Split View regression guard above -- is a
    "building the export doesn't raise" guard; app/professional_tools.py's own unit tests
    (test_apply_watermark_*, test_export_profile_applies_watermark_when_text_given) cover the
    actual pixel-level watermarking behavior."""
    at = run_with_uploaded_image()

    at.text_input(key="wm-text-in").set_value("© Test Photographer")
    at.run(timeout=30)
    at.selectbox(key="wm-pos-sel").set_value("Center")
    at.run(timeout=30)
    at.slider(key="wm-opacity-sl").set_value(80)
    at.run(timeout=30)

    assert not at.exception
    assert any(db.label == "💾 Save JPEG" for db in at.download_button)


def test_export_multiple_sizes_zip_builds_without_error_when_a_watermark_is_set():
    at = run_with_uploaded_image()
    at.text_input(key="wm-text-in").set_value("Preview")
    at.run(timeout=30)

    at.multiselect(key="exp-multi-profiles").set_value(["Web"])
    at.run(timeout=30)
    at.button(key="exp-multi-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any(db.label == "⬇️ Download ZIP" for db in at.download_button)


def test_blank_watermark_text_leaves_export_controls_working_as_before():
    """Blank text (the default) must not change the single-format export's behavior at all --
    the watermark feature is fully opt-in."""
    at = run_with_uploaded_image()

    assert at.text_input(key="wm-text-in").value == ""
    assert not at.exception
    assert any(db.label == "💾 Save JPEG" for db in at.download_button)


def test_load_project_replaces_an_already_open_photo_and_resets_unmentioned_controls():
    """Regression: the main uploader had no key and always won over the project's bundled source,
    so loading a .photoproject while a photo was open applied its recipe to the *open* photo; and
    the recipe wasn't merged over RECIPE_DEFAULTS, so any control the project didn't mention kept
    the previous photo's value."""
    at = run_with_uploaded_image(size=(40, 30))
    at.slider(key="t-contrast").set_value(40)
    at.run(timeout=30)
    assert canvas_dims(at) == (40, 30)

    source_buffer = io.BytesIO()
    Image.new("RGB", (20, 10), (10, 200, 10)).save(source_buffer, format="PNG")
    bundle_buffer = io.BytesIO()
    ProjectBundle("project.png", source_buffer.getvalue(), {"exposure": 1.3}).save(bundle_buffer)

    at.get_by_key("proj-upload").set_value(("p.photoproject", bundle_buffer.getvalue(), "application/zip"))
    at.run(timeout=30)
    at.button(key="proj-load-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert canvas_dims(at) == (20, 10)
    assert not at.file_uploader[0].value
    assert at.slider(key="t-ev").value == 0.85
    assert at.slider(key="t-contrast").value == 0


def test_every_streamlit_data_cache_is_bounded():
    """Regression: the st.cache_data caches had no max_entries, so every slider position kept a
    full-resolution image alive for the life of the server process."""
    source = ui_source()
    decorators = re.findall(r"@st\.cache_data\((.*?)\)\n", source)
    assert decorators
    assert all("max_entries" in args for args in decorators)


def test_repair_starts_on_the_repair_brush_and_shows_the_result_on_the_photo():
    """Repair is mostly painting over something: its selection type starts on the paint editor,
    named a repair brush, and adding a painting returns the canvas to the photo with it removed."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    assert at.selectbox(key="rm-new-kind").value == "Painted"
    baseline = canvas_hash(at)
    open_tab(at, "◉ Local", "🩹 Repair")
    at.run(timeout=30)
    assert "to-canvas-rm-paint" in [button.key for button in at.button]

    at.session_state["canvas-mode"] = "🩹 Repair brush"
    open_tab(at, "◉ Local", "🩹 Repair")
    at.run(timeout=30)
    assert at.session_state["canvas-mode"] == "🩹 Repair brush"
    assert any("🩹 Remove" in frame.proto.srcdoc for frame in at.get("iframe"))

    at.text_area(key="rm-paint-data").set_value(painted_mask_base64())
    open_tab(at, "◉ Local", "🩹 Repair")
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["removal_mask_layers"][0]["kind"] == "Painted"
    assert at.session_state["canvas-mode"] == "🖼 Photo"
    assert canvas_hash(at) != baseline


def test_find_a_tool_opens_its_tab_and_scrolls_to_it():
    at = run_with_uploaded_image()
    at.selectbox(key="find-tool").set_value("Remove Objects")
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["tool-tab"] == "◉ Local" and at.session_state["local-tab"] == "🩹 Repair"
    assert at.selectbox(key="find-tool").value is None  # ready for the next search
    assert any(".st-key-" in frame.proto.srcdoc and '"rm-new-kind"' in frame.proto.srcdoc for frame in at.get("iframe"))

    at.selectbox(key="find-tool").set_value("Batch Processor")
    at.run(timeout=30)
    assert at.session_state["tool-tab"] == "✦ Presets & Library" and at.radio(key="more-mode").value == "Batch Processor"


def _upload_with_profile(icc: bytes | None):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 60, 40)).save(buf, format="PNG", **({"icc_profile": icc} if icc else {}))
    at.file_uploader[0].set_value(("photo.png", buf.getvalue(), "image/png"))
    at.run(timeout=30)
    return at


def test_a_wide_gamut_profile_is_named_above_the_photo_with_a_way_to_proof_it():
    """Photoshop stops at "Embedded Profile Mismatch"; here the conversion is automatic, and said."""
    from app.color_profiles import display_p3_icc_bytes

    at = _upload_with_profile(display_p3_icc_bytes())
    assert any("Display P3" in caption.value and "sRGB" in caption.value for caption in at.caption)
    at.button(key="profile-proof-btn").click()
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["tool-tab"] == "🎨 Curves & Color" and at.checkbox(key="gamut-warning-toggle").value is True

    plain = _upload_with_profile(None)
    assert "profile-proof-btn" not in [button.key for button in plain.button]
