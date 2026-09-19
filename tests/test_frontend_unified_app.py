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
from app.project import PhotoProject
from app.project_bundle import ProjectBundle

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "frontend_unified_app.py"


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


def _fake_process_via_backend(image_bytes_, filename, options, local_mask_bytes=None, layer_bytes=None, removal_mask_bytes=None):
    request_options = dict(options)
    request_options["output_format"] = "PNG"
    image_file = _FakeFileStorage(filename, image_bytes_)
    local_mask = _FakeFileStorage("mask.png", local_mask_bytes) if local_mask_bytes else None
    layer_file = _FakeFileStorage("layer.png", layer_bytes) if layer_bytes else None
    removal_mask = _FakeFileStorage("removal_mask.png", removal_mask_bytes) if removal_mask_bytes else None
    try:
        return process_image(image_file, request_options, removal_mask=removal_mask, local_mask=local_mask, layer_file=layer_file)
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
    at.slider(key="m-texp").set_value(1.3)
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
    assert [t.label for t in at.tabs] == [
        "☼ Tone", "◎ Lens", "⛶ Crop", "\U0001f3a8 Color", "\U0001f52c Detail", "🩹 Repair", "◈ Layers",
        "◉ Masks", "◌ Portrait", "✦ Presets", "☰ Batch & Library",
    ]
    assert canvas_dims(at) == (40, 30)


def test_layer_composite_is_reflected_in_final_image():
    """Regression test for the bug where the Layers tab composited into a local variable
    that every later tab (and the final export) ignored -- see docs/PROJECT_STATUS.md's
    "Recently Fixed" entry. A layer that changes the composite must change what's shown."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.get_by_key("l-file").set_value(("overlay.png", image_bytes((255, 0, 0), (40, 30)), "image/png"))
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


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

    at.slider(key="m-texp").set_value(1.5)
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

    at.slider(key="m-texp").set_value(1.5)
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
    re-edit section): committing through it must update the same layer's mask_b64 in place, not
    append a second layer -- the same "genuine new commit only" guard as adding a fresh Painted
    selection, applied to the edit path instead."""
    at = run_with_uploaded_image(color=(80, 80, 80))

    at.selectbox(key="m-new-kind").set_value("Painted")
    at.run(timeout=30)
    at.text_area(key="m-paint-data").set_value(painted_mask_base64())
    at.run(timeout=30)

    layer_id = at.session_state["mask_layers"][0]["id"]
    original_mask_b64 = at.session_state["mask_layers"][0]["params"]["mask_b64"]

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
    assert layers[0]["params"]["mask_b64"] == edited_mask
    assert layers[0]["params"]["mask_b64"] != original_mask_b64


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
    at.slider(key="m-tsat").set_value(1.72)
    at.run(timeout=30)
    baseline = canvas_hash(at)

    at.selectbox(key="rm-new-kind").set_value("Painted")
    at.run(timeout=30)
    at.text_area(key="rm-paint-data").set_value(painted_mask_base64())
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="m-tsat").value == 1.72
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


def test_mask_target_sliders_disabled_until_an_enabled_mask_layer_exists():
    at = run_with_uploaded_image()

    assert at.slider(key="m-texp").disabled
    assert at.slider(key="m-tcont").disabled
    assert at.slider(key="m-tsat").disabled

    at.selectbox(key="m-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.button(key="m-add-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert not at.slider(key="m-texp").disabled
    assert not at.slider(key="m-tcont").disabled
    assert not at.slider(key="m-tsat").disabled


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


def test_undo_button_does_not_reset_canvas_view_settings():
    """Same bug class as test_adding_a_mask_selection_does_not_reset_canvas_view_settings, but for
    canvas_col's own Undo button, which used to call st.rerun() directly -- aborting the script
    before the zoom/split-view/pan/export widgets defined right after it in canvas_col were
    instantiated that pass."""
    at = run_with_uploaded_image()
    at.slider(key="t-exp").set_value(1.2)
    at.run(timeout=30)
    at.selectbox(key="cv-zoom").set_value(2.0)
    at.run(timeout=30)
    assert at.slider(key="t-exp").value == 1.2

    at.button(key="hist-undo-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-exp").value == 1.0
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
    at.get_by_key("l-file").set_value(("overlay.png", image_bytes((255, 0, 0), (40, 30)), "image/png"))
    at.run(timeout=30)
    with_layer_hash = canvas_hash(at)

    at.slider(key="t-exp").set_value(1.4)
    at.slider(key="d-text").set_value(60)
    at.run(timeout=30)
    assert at.slider(key="t-exp").value == 1.4

    at.button(key="hist-reset-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-exp").value == 1.0
    assert at.slider(key="d-text").value == 0
    # The layer itself survives the reset -- same composited image as right after uploading it.
    assert canvas_hash(at) == with_layer_hash


def test_auto_enhance_button_applies_photoprocessor_auto_enhance_values():
    """Mirrors PhotoProcessor.auto_enhance()'s fixed constants (app/image_processor.py) --
    a regression here would mean the button and the CLI's --auto-enhance flag have drifted
    apart."""
    at = run_with_uploaded_image()

    at.button(key="auto-enhance-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-exp").value == 1.12
    assert at.slider(key="t-cont").value == 1.15
    assert at.slider(key="c-sat").value == 1.08
    assert at.slider(key="d-den").value == 3
    assert at.checkbox(key="d-sharp").value is True


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
    assert at.slider(key="t-exp").value == 1.0

    at.selectbox(key="p-load-sel").set_value("Warm")
    at.run(timeout=30)
    at.button(key="p-load-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-exp").value == 1.3
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
    at.radio[0].set_value("Batch Processor")
    at.run(timeout=30)

    assert not at.exception
    options = at.selectbox(key="b-preset-sel").options
    assert "None" in options
    assert "My Look (custom)" in options
    assert any(name in options for name in PRESETS)


def test_save_project_shows_a_toast_pointing_at_the_download_button():
    """Safe to click for real: ProjectBundle.save() here writes to an in-memory io.BytesIO,
    not disk."""
    at = run_with_uploaded_image()

    at.button(key="proj-save-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("Download" in t.value for t in at.toast)


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

    at.slider(key="t-exp").set_value(1.2)
    at.run(timeout=30)
    assert at.slider(key="t-exp").value == 1.2

    at.button(key="hist-undo-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-exp").value == 1.0


def test_save_project_button_builds_a_bundle_with_the_layer_without_erroring():
    """Regression guard for the Save Project button's ProjectBundle wiring (layer_bytes/opacity/
    blend -> LayerWorkspace -> ProjectBundle) -- a mismatch with LayerWorkspace's API would raise
    here rather than only being caught by decoding the downloaded bytes, which AppTest can't do."""
    at = run_with_uploaded_image()
    at.get_by_key("l-file").set_value(("overlay.png", image_bytes((255, 0, 0), (40, 30)), "image/png"))
    at.run(timeout=30)

    at.button(key="proj-save-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any(db.label == "⬇️ Download .photoproject" for db in at.download_button)


def test_save_project_with_a_corrupted_layer_shows_an_error_instead_of_crashing():
    """Same bug class as test_uploading_a_corrupted_image_shows_an_error_instead_of_crashing,
    for the second unguarded Image.open() found in review: Save Project opens the layer bytes
    locally to build a LayerWorkspace, which previously had no error handling of its own. In
    practice a corrupted layer is already caught earlier too, by the same backend round-trip
    every rerun already goes through (compare test_uploading_a_corrupted_image_... above) -- this
    confirms clicking Save Project on top of that still doesn't crash, and gets its own clearer,
    layer-specific message rather than relying only on the generic backend error."""
    valid_png = image_bytes((255, 0, 0), (40, 30))
    truncated = valid_png[: len(valid_png) // 2]

    at = run_with_uploaded_image()
    at.get_by_key("l-file").set_value(("overlay.png", truncated, "image/png"))
    at.run(timeout=30)

    at.button(key="proj-save-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("layer image" in message.value.lower() for message in at.error)
    assert not any(db.label == "⬇️ Download .photoproject" for db in at.download_button)


def test_load_project_restores_recipe_and_layer_from_a_project_bundle():
    """Regression test for the migration off PhotoProject to ProjectBundle: the overlay layer now
    lives in ProjectBundle's structured `layer_workspace` field instead of being base64-embedded in
    the recipe, so the Load handler has to translate it back into the recipe["layer"] shape
    _restore_recipe_to_state() expects."""
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
    assert at.slider(key="t-exp").value == 1.3
    assert at.slider(key="l-opac").value == 0.7
    assert at.selectbox(key="l-blend").value == "multiply"
    restored_overlay = Image.open(io.BytesIO(at.session_state["restored_layer_bytes"]))
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
    assert at.slider(key="t-exp").value == 1.4


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

    at.slider(key="m-texp").set_value(1.5)
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
    at.radio[0].set_value("Batch Processor")
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
    at.radio[0].set_value("Batch Processor")
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


def test_mask_overlay_preview_reflects_lens_distortion_correction():
    """Same bug class as test_crop_preview_reflects_lens_distortion_correction, but for the
    Masks tab's "Preview combined mask overlay" debug view, which used to build the overlay
    from the raw `source` regardless of the Lens tab's sliders."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    at.selectbox(key="m-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.button(key="m-add-btn").click()
    at.run(timeout=30)
    at.checkbox(key="m-preview-overlay").set_value(True)
    at.run(timeout=30)
    baseline = [img.value[0] for img in at.image if img.captions and img.captions[0] == "Red = selected by the combined mask"][0]

    at.slider(key="lens-distortion").set_value(80)
    at.run(timeout=30)

    assert not at.exception
    overlay_after = [img.value[0] for img in at.image if img.captions and img.captions[0] == "Red = selected by the combined mask"][0]
    assert overlay_after != baseline


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
    at.radio[0].set_value("Library Catalog")
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
    at.radio[0].set_value("Library Catalog")
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
    at.radio[0].set_value("Library Catalog")
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

    at.radio[0].set_value("Library Catalog")
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
