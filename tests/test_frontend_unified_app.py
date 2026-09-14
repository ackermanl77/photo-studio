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

import io
import re
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from app import backend_client
from app.backend import process_image
from app.layer_workspace import LayerWorkspace, WorkspaceLayer
from app.presets import PRESETS
from app.project import PhotoProject
from app.project_bundle import ProjectBundle

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "frontend_unified_app.py"


def image_bytes(color=(90, 120, 160), size=(40, 30), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


class _FakeFileStorage:
    """Minimal stand-in for Flask's FileStorage -- just enough of its interface
    (`.filename`, `.read()`, `.stream`) for app.backend.process_image() to accept it."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.stream = io.BytesIO(content)

    def read(self) -> bytes:
        self.stream.seek(0)
        return self.stream.read()


def _fake_process_via_backend(image_bytes_, filename, options, local_mask_bytes=None, layer_bytes=None):
    request_options = dict(options)
    request_options["output_format"] = "PNG"
    image_file = _FakeFileStorage(filename, image_bytes_)
    local_mask = _FakeFileStorage("mask.png", local_mask_bytes) if local_mask_bytes else None
    layer_file = _FakeFileStorage("layer.png", layer_bytes) if layer_bytes else None
    return process_image(image_file, request_options, removal_mask=None, local_mask=local_mask, layer_file=layer_file)


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


def test_uploading_image_renders_every_tab_without_errors():
    at = run_with_uploaded_image()

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "☼ Tone", "⛶ Crop", "\U0001f3a8 Color", "\U0001f52c Detail", "◈ Layers",
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


def test_save_preset_without_a_name_shows_a_warning_toast():
    """Doesn't click Save with an actual preset name -- that would write to the real
    ~/.photo-studio/presets.json on whatever machine runs this test. The empty-name path is
    safe: PresetStore.save() is never called, only the toast fires."""
    at = run_with_uploaded_image()

    at.button(key="p-save-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert any("name" in t.value.lower() for t in at.toast)


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
    at = run_with_uploaded_image(size=(40, 30))
    assert canvas_dims(at) == (40, 30)

    at.selectbox(key="crop-aspect").set_value("Free-form")
    at.run(timeout=30)
    at.slider(key="crop-x").set_value((5, 25))
    at.slider(key="crop-y").set_value((2, 20))
    at.run(timeout=30)

    assert not at.exception
    assert canvas_dims(at) == (20, 18)


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
