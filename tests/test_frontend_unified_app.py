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

import pytest
from PIL import Image
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
    return process_image(image_file, request_options, removal_mask=removal_mask, local_mask=local_mask, layer_file=layer_file)


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
        "☼ Tone", "⛶ Crop", "\U0001f3a8 Color", "\U0001f52c Detail", "🩹 Repair", "◈ Layers",
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
