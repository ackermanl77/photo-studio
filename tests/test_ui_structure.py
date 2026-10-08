"""The UI's structure after the clean-up items: one control registry (F15), the app/ui package
split (F14), the workflow tab grouping with "modified" marks (F9), fragments (F1) and painted
masks stored once (P5)."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from app.controls import CONTROLS, CONTROLS_BY_KEY, RECIPE_WIDGET_KEYS, Control, controls_to_options, section_keys
from app.edit_schema import DETAIL_STAGE_OPTION_KEYS, OPTION_SPECS, RECIPE_CONTROLS, OptionSpec, _control_option_spec, recipe_defaults
from app.painted_masks import inline_painted_masks, intern_painted_mask, intern_recipe_painted_masks, painted_mask_ref, resolve_painted_mask
from app.recipe_options import recipe_to_options
from tests.test_frontend_unified_app import APP_PATH, UI_DIR, _capture_download, painted_mask_base64, run_with_uploaded_image
from tests.test_frontend_unified_app import _stub_backend  # noqa: F401  (autouse: the in-process backend stand-in)

# Saved state, presets, projects and the AppTest suite address widgets by these keys: they must
# never change (the HSL keys, hsl-<band>-<h|s|l>, are checked by pattern below).
PINNED_WIDGET_KEYS = {
    "cg_balance": "cg-balance", "cg_highlights_hue": "cg-hi-hue",
    "cg_highlights_intensity": "cg-hi-int", "cg_midtones_hue": "cg-mid-hue",
    "cg_midtones_intensity": "cg-mid-int", "cg_shadows_hue": "cg-sh-hue",
    "cg_shadows_intensity": "cg-sh-int", "clarity": "d-clar", "constrain_crop": "geo-constrain",
    "contrast_amount": "t-contrast", "crop_aspect": "crop-aspect", "crop_custom_h": "crop-custom-h",
    "crop_custom_w": "crop-custom-w", "crop_free_x0": "crop-free-x0", "crop_free_x1": "crop-free-x1",
    "crop_free_y0": "crop-free-y0", "crop_free_y1": "crop-free-y1", "crop_swap": "crop-swap",
    "curve_black": "c-p0", "curve_enabled": "c-curve-en", "curve_highlights": "c-p3",
    "curve_mid": "c-p2", "curve_shadows": "c-p1", "curve_white": "c-p4", "dehaze": "d-haze",
    "denoise": "d-den", "denoise_color": "d-den-color", "effects_grain_amount": "fx-grain-amt",
    "effects_vignette_amount": "fx-vignette-amt", "effects_vignette_midpoint": "fx-vignette-mid",
    "exposure_ev": "t-ev", "flip_horizontal": "geo-flip-h", "flip_vertical": "geo-flip-v",
    "highlights": "t-high", "lens_chromatic_aberration": "lens-ca",
    "lens_distortion": "lens-distortion", "lens_perspective_h": "lens-persp-h",
    "lens_perspective_v": "lens-persp-v", "lens_profile": "lens-profile",
    "lens_vignette_amount": "lens-vignette-amt", "lens_vignette_midpoint": "lens-vignette-mid",
    "mask_target_contrast_amount": "m-tcontrast", "mask_target_exposure_ev": "m-tev",
    "mask_target_saturation_amount": "m-tsaturation", "portrait_smoothing": "port-sm", "portrait_faces_only": "port-faces",
    "preset": "tone-preset", "removal_radius": "rm-radius", "rotate": "geo-rotate",
    "saturation_amount": "c-saturation", "shadows": "t-shad", "sharpen": "d-sharp",
    "straighten": "geo-straighten", "temperature": "c-temp", "texture": "d-text", "tint": "c-tint",
    "vibrance": "c-vib", "whites": "t-whites", "blacks": "t-blacks", "black_white": "bw-enabled", "removal_method": "rm-method",
    **{f"curve_{c}_{point}": f"c-{c}-p{i}" for c in "rgb" for i, point in enumerate(("black", "shadows", "mid", "highlights", "white"))},
    **{f"bw_{band}": f"bw-{band}" for band in ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")},
}


# --- F15: one control registry ------------------------------------------------------------------

def test_widget_keys_are_unchanged():
    assert {key: value for key, value in RECIPE_WIDGET_KEYS.items() if not key.startswith("hsl_")} == PINNED_WIDGET_KEYS
    hsl = {key: value for key, value in RECIPE_WIDGET_KEYS.items() if key.startswith("hsl_")}
    assert len(hsl) == 24 and all(value == f"hsl-{key.split('_')[1]}-{key.split('_')[2][0]}" for key, value in hsl.items())
    assert len(set(RECIPE_WIDGET_KEYS.values())) == len(RECIPE_WIDGET_KEYS)


def test_schema_tables_are_derived_from_the_registry():
    assert set(RECIPE_CONTROLS) == set(CONTROLS_BY_KEY)
    for control in CONTROLS:
        recipe_control = RECIPE_CONTROLS[control.key]
        assert (recipe_control.kind, recipe_control.default, recipe_control.minimum, recipe_control.maximum) == (control.kind, control.default, control.minimum, control.maximum)
        if control.option:
            spec = OPTION_SPECS[control.option]
            assert spec.stage == control.stage, control.key
            if spec.kind in ("number", "int"):
                # The widget's whole range, carried through its transform, is valid for the API.
                assert spec.minimum <= control.to_option(control.minimum) and control.to_option(control.maximum) <= spec.maximum, control.key
    assert set(DETAIL_STAGE_OPTION_KEYS) >= {control.option for control in CONTROLS if control.option and control.stage == "detail"}


def test_a_new_control_is_declared_once():
    """A Control is all it takes: its option spec, recipe entry and options mapping follow."""
    control = Control("glow", "fx-glow", "Glow", "int", 0, -50, 50, section="effects", option="glow", stage="detail", transform="amount")
    assert _control_option_spec(control) == OptionSpec("number", 0.5, 1.5, stage="detail")
    assert control.widget_type == "slider" and control.to_option(50) == 1.5


def test_live_options_and_saved_recipes_share_one_mapping():
    recipe = recipe_defaults() | {"contrast_amount": 40, "saturation_amount": -30, "mask_target_contrast_amount": 10, "rotate": 90, "exposure_ev": 0.5}
    live = controls_to_options(recipe)
    saved = recipe_to_options(recipe)
    assert live["contrast"] == saved["contrast"] == pytest.approx(1.4)
    assert live["saturation"] == saved["saturation"] == pytest.approx(0.7)
    assert live["mask_contrast"] == pytest.approx(1.1) and live["exposure_ev"] == 0.5
    assert live["rotate"] == 90 and "rotate" not in saved  # framing is photo-specific
    assert set(section_keys("presence")) == {"texture", "clarity", "dehaze"}


# --- F14: the split -----------------------------------------------------------------------------

def test_the_entry_script_is_the_only_streamlit_page_and_stays_small():
    sources = {path: path.read_text(encoding="utf-8") for path in [APP_PATH, *UI_DIR.rglob("*.py")]}
    pages = [path.name for path, text in sources.items() if "st.set_page_config(" in text]
    assert pages == ["frontend_unified_app.py"]
    assert len(sources[APP_PATH].splitlines()) < 300
    assert (UI_DIR / "static" / "theme.css").is_file() and (UI_DIR / "static" / "library_shortcuts.html").is_file()


def test_colours_are_defined_once_as_theme_variables():
    import re

    from app import ui_editors
    from app.ui.theme import THEME, editor_style, page_style

    css = page_style()
    assert all(f"--ps-{name}:" in css for name in THEME)
    stylesheet = (UI_DIR / "static" / "theme.css").read_text(encoding="utf-8")
    assert not re.findall(r"#[0-9a-fA-F]{3,6}\b|rgba?\(", stylesheet)  # only var(--ps-...)
    assert editor_style().startswith("<style>:root {--ps-")
    editors = open(ui_editors.__file__, encoding="utf-8").read()
    assert not re.findall(r"#[0-9a-fA-F]{6}", editors) and "var(--ps-accent)" in editors
    from PIL import Image

    markup, _height = ui_editors._wand_picker_html(Image.new("RGB", (8, 8)), "t")
    assert markup.startswith(editor_style())


# --- F1: fragments ------------------------------------------------------------------------------

def _fragment_script():
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    from app.ui.state import consume_rerun_request, request_rerun, run_fragment

    def scopes():
        st.checkbox("scope", key="scope")

    def view():
        if st.button("go", key="go"):
            request_rerun()

    run_fragment(scopes)
    run_fragment(view)
    storage = getattr(get_script_run_ctx().fragment_storage, "_fragments", {})
    st.session_state["fragments"] = len(storage)
    st.session_state["runs"] = st.session_state.get("runs", 0) + 1
    consume_rerun_request()


@pytest.mark.slow
def test_each_canvas_panel_is_its_own_fragment_and_reruns_are_deferred():
    at = AppTest.from_function(_fragment_script)
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["fragments"] == 2  # distinct ids, though both live in one container
    at.button(key="go").click()
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["runs"] == 3  # the click's run, then the deferred full rerun


@pytest.mark.slow
def test_panning_and_scopes_still_work_inside_their_fragments():
    at = run_with_uploaded_image(size=(40, 30))
    at.selectbox(key="cv-zoom").set_value(2.0)
    at.run(timeout=30)
    at.slider(key="cv-panx").set_value(0.8)
    at.run(timeout=30)
    at.checkbox(key="waveform-toggle").check()
    at.run(timeout=30)
    assert not at.exception
    assert at.slider(key="cv-panx").value == 0.8
    assert any(img.captions and img.captions[0].startswith("Waveform") for img in at.image)


# --- F9: workflow tabs and "modified" marks -----------------------------------------------------

def _markdowns(at):
    return [markdown.value for markdown in at.markdown]


@pytest.mark.slow
def test_section_headers_show_a_modified_mark():
    at = run_with_uploaded_image()
    assert "**Tone**" in _markdowns(at) and "**Presence**" in _markdowns(at)
    at.slider(key="t-ev").set_value(0.5)
    at.slider(key="d-clar").set_value(30)
    at.run(timeout=30)
    assert not at.exception
    marked = [value for value in _markdowns(at) if "●" in value]
    assert any(value.startswith("**Tone**") for value in marked)
    assert any(value.startswith("**Presence**") for value in marked)
    assert "**Color**" in _markdowns(at)  # untouched sections stay unmarked


@pytest.mark.slow
def test_keyed_section_expanders_carry_the_mark():
    at = run_with_uploaded_image()
    labels = {expander.label for expander in at.expander}
    assert "📈 RGB Tone Curve" in labels
    at.checkbox(key="c-curve-en").check()
    at.slider(key="cg-sh-int").set_value(20)
    at.run(timeout=30)
    labels = {expander.label for expander in at.expander}
    assert "📈 RGB Tone Curve ●" in labels and "🎨 Color Grading (Split Toning) ●" in labels


@pytest.mark.slow
def test_presence_has_its_own_reset_and_detail_keeps_its_own():
    at = run_with_uploaded_image()
    at.slider(key="d-clar").set_value(40)
    at.slider(key="d-den").set_value(4)
    at.run(timeout=30)
    at.button(key="reset-presence-btn").click()
    at.run(timeout=30)
    at.run(timeout=30)
    assert not at.exception
    assert at.slider(key="d-clar").value == 0 and at.slider(key="d-den").value == 4


@pytest.mark.slow
def test_custom_ratio_survives_switching_the_aspect_ratio():
    """The custom ratio fields always render now (disabled unless "Custom"), so their values
    aren't dropped while another ratio is picked."""
    at = run_with_uploaded_image()
    at.selectbox(key="crop-aspect").set_value("Custom")
    at.run(timeout=30)
    at.number_input(key="crop-custom-w").set_value(7)
    at.run(timeout=30)
    at.selectbox(key="crop-aspect").set_value("1:1 Square")
    at.run(timeout=30)
    at.selectbox(key="crop-aspect").set_value("Custom")
    at.run(timeout=30)
    assert not at.exception
    assert at.number_input(key="crop-custom-w").value == 7


# --- P5: painted masks stored once --------------------------------------------------------------

def test_painted_masks_are_interned_and_inlined_again_for_files():
    table = {}
    mask = painted_mask_base64()
    ref = intern_painted_mask(mask, table)
    assert ref == painted_mask_ref(mask) == intern_painted_mask(mask, table) and len(table) == 1
    recipe = {"mask_layers": [{"kind": "Painted", "params": {"mask_b64": mask}}], "removal_mask_layers": [{"kind": "Painted", "params": {"mask_b64": mask}}]}
    interned = intern_recipe_painted_masks(recipe, table)
    assert len(table) == 1 and mask not in json.dumps(interned)
    assert interned["mask_layers"][0]["params"] == {"mask_ref": ref}
    assert inline_painted_masks(interned, table) == recipe
    assert resolve_painted_mask({"mask_ref": "sha256:missing"}, table)  # an empty mask, not a crash


@pytest.mark.slow
def test_painted_masks_are_stored_once_per_session_and_inlined_in_projects(monkeypatch):
    captured = _capture_download(monkeypatch, "proj-dl-btn")
    at = run_with_uploaded_image(color=(80, 80, 80))
    mask = painted_mask_base64()
    for prefix in ("m", "rm"):
        at.selectbox(key=f"{prefix}-new-kind").set_value("Painted")
        at.run(timeout=30)
        at.text_area(key=f"{prefix}-paint-data").set_value(mask)
        at.run(timeout=30)
    at.slider(key="t-ev").set_value(0.3)
    at.run(timeout=30)
    assert not at.exception
    assert len(at.session_state["_painted_masks"]) == 1  # the same painting in Masks and Repair
    history = at.session_state["edit_history"]
    assert len(history.snapshots) >= 3 and mask not in json.dumps(history.snapshots)

    from app.project_bundle import ProjectBundle

    project = ProjectBundle.load(captured["data"]())
    assert project.recipe["mask_layers"][0]["params"]["mask_b64"] == mask  # files stay self-contained
    assert project.recipe["removal_mask_layers"][0]["params"]["mask_b64"] == mask
