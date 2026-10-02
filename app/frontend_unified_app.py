from __future__ import annotations

import base64
import functools
import html
import io
import json
import os
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

import cv2

from app.advanced_masks import COMBINE_MODES, MaskLayer, MaskStack, color_mask, gradient_mask, luminance_mask, magic_wand_mask, radial_mask, soften_mask
from app.ai_upscale import ai_upscale_installed
from app.backend import process_image
from app.backend_client import SupersededError, bind_edit_session, ensure_backend_running, export_via_backend, process_via_backend
from app.batch_release import FileLike, export_multiple, stream_to_zip
from app.config import MAX_UPLOAD_BYTES, user_data_dir, user_data_file
from app.custom_presets import PresetStore
from app.edit_history import EditHistory
from app.edit_schema import DETAIL_STAGE_OPTION_KEYS, migrate_legacy_keys, recipe_defaults, sanitize_recipe
from app.export import DEFAULT_NAME_TEMPLATE, EXPORT_FORMATS, METADATA_POLICIES, METADATA_POLICY_LABELS, OUTPUT_SHARPENING, ExportPresetStore, cached_export, encode_export, render_export_name
from app.image_processor import PhotoProcessor, auto_enhance_recipe
from app.pipeline import check_upscale_output, expected_output_size
from app.layer_styles import apply_layer_styles
from app.layer_workspace import LayerWorkspace, WorkspaceLayer, layers_to_json, render_image_layer, render_text_layer
from app.ui_editors import _before_after_html, _crop_editor_html, _curve_editor_html, _paint_editor_html, _pen_editor_html, _wand_picker_html
from app.vector_shapes import SHAPE_TYPES, render_path_layer, render_shape_layer
from app.library import index_folder, thumbnail
from app.major_features import BLEND_MODE_LABELS, PhotoCatalog
from app.presets import PRESETS, apply_preset
from app.preset_exchange import export_presets, import_presets
from app.portrait_tools import detect_faces, face_detection_backend
from app.professional_color import HSL_BAND_ORDER, ICC_TRANSFORM_ERRORS, clipping_warning_overlay, estimate_white_balance, gamut_warning_overlay
from app.geometry import ROTATIONS, apply_geometry, estimate_straighten_angle, geometry_output_size, orient
from app.lens_corrections import LENS_PROFILES
from app.professional_tools import EXPORT_PROFILES, GRID_PRESETS, OVERLAY_STYLES, WATERMARK_POSITIONS, apply_mask_overlay, apply_watermark, compute_histogram, compute_waveform, draw_guides_and_grid, draw_viewport_box, export_profile, grid_line_fractions, preview_image, zoom_preview, zoom_viewport
from app.project_bundle import ProjectBundle
from app.raw_import import SUPPORTED_EXTENSIONS, uploader_types
from app.reliability import image_digest
from app.recipe_options import ASPECT_RATIOS, merge_curve_points, recipe_to_options, without_per_photo_only_keys

# A small bundled placeholder image (generated via PIL -- gradient background plus a few flat
# shapes, not a real photo) so a first-time visitor can explore every tool without needing to
# supply their own file first. "Try a sample photo" below loads it through the exact same
# restored_source_bytes/restored_source_name fallback path a reopened .photoproject uses.
SAMPLE_PHOTO_PATH = Path(__file__).resolve().parent / "assets" / "sample_photo.jpg"
SAMPLE_PHOTO_NAME = "sample_photo.jpg"
# What a saved project may expand to and still reopen (app.reliability.validate_project_archive).
PROJECT_MAX_BYTES = 250 * 1024 * 1024

# Maps a JSON-serializable recipe key to the session_state key of the widget that reads/writes it.
# Shared by Undo/Redo (EditHistory snapshots) and Save/Load Project (ProjectBundle.recipe) so both
# features capture and restore exactly the same set of controls.
RECIPE_WIDGET_KEYS = {
    "exposure_ev": "t-ev", "contrast_amount": "t-contrast", "highlights": "t-high", "shadows": "t-shad", "preset": "tone-preset",
    "temperature": "c-temp", "tint": "c-tint", "saturation_amount": "c-saturation", "vibrance": "c-vib",
    "curve_enabled": "c-curve-en", "curve_black": "c-p0", "curve_shadows": "c-p1", "curve_mid": "c-p2",
    "curve_highlights": "c-p3", "curve_white": "c-p4",
    "texture": "d-text", "clarity": "d-clar", "denoise": "d-den", "denoise_color": "d-den-color",
    "sharpen": "d-sharp", "dehaze": "d-haze",
    "effects_vignette_amount": "fx-vignette-amt", "effects_vignette_midpoint": "fx-vignette-mid",
    "effects_grain_amount": "fx-grain-amt",
    "portrait_smoothing": "port-sm",
    "mask_target_exposure_ev": "m-tev", "mask_target_contrast_amount": "m-tcontrast", "mask_target_saturation_amount": "m-tsaturation",
    "crop_aspect": "crop-aspect", "removal_radius": "rm-radius",
    "crop_custom_w": "crop-custom-w", "crop_custom_h": "crop-custom-h", "crop_swap": "crop-swap",
    "rotate": "geo-rotate", "flip_horizontal": "geo-flip-h", "flip_vertical": "geo-flip-v",
    "straighten": "geo-straighten", "constrain_crop": "geo-constrain",
    "crop_free_x0": "crop-free-x0", "crop_free_y0": "crop-free-y0",
    "crop_free_x1": "crop-free-x1", "crop_free_y1": "crop-free-y1",
    "lens_profile": "lens-profile", "lens_distortion": "lens-distortion",
    "lens_vignette_amount": "lens-vignette-amt", "lens_vignette_midpoint": "lens-vignette-mid",
    "lens_chromatic_aberration": "lens-ca",
    "lens_perspective_v": "lens-persp-v", "lens_perspective_h": "lens-persp-h",
    "cg_shadows_hue": "cg-sh-hue", "cg_shadows_intensity": "cg-sh-int",
    "cg_midtones_hue": "cg-mid-hue", "cg_midtones_intensity": "cg-mid-int",
    "cg_highlights_hue": "cg-hi-hue", "cg_highlights_intensity": "cg-hi-int",
    "cg_balance": "cg-balance",
}
# Defaults (and every control's valid range/type) come from app.edit_schema.RECIPE_CONTROLS -- the
# same table sanitize_recipe() clamps restored values against, so the defaults Reset/Load Project
# fall back to can't drift from what the widgets accept. AI Upscale is no longer a recipe control:
# it's an export-time action (see the Export panel), not part of a look.
RECIPE_DEFAULTS = recipe_defaults()


def _hsl_widget_key(band: str, channel: str) -> str:
    """channel is one of "h"/"s"/"l" -- e.g. _hsl_widget_key("Orange", "s") -> "hsl-orange-s"."""
    return f"hsl-{band.lower()}-{channel}"


# Every band/channel combination participates in Undo/Redo/Save-Project exactly like every other
# control, via the same RECIPE_WIDGET_KEYS/RECIPE_DEFAULTS mechanism -- generated here instead of
# 24 hand-written entries since it's a direct product of HSL_BAND_ORDER x {hue, sat, lum}.
for _band in HSL_BAND_ORDER:
    for _channel, _suffix in (("hue", "h"), ("sat", "s"), ("lum", "l")):
        RECIPE_WIDGET_KEYS[f"hsl_{_band.lower()}_{_channel}"] = _hsl_widget_key(_band, _suffix)


def _build_recipe() -> dict:
    """Capture every editable control's current value into one JSON-serializable dict: used both
    for Undo/Redo snapshots and for the recipe saved inside a .photoproject file. The layer stack's
    image data is deliberately NOT included here (see the Layers tab and the Project File section)
    -- it is saved with the project file, but kept out of per-step Undo/Redo history so that
    history snapshots stay small.
    """
    recipe = {key: st.session_state.get(widget_key, RECIPE_DEFAULTS[key]) for key, widget_key in RECIPE_WIDGET_KEYS.items()}
    recipe["mask_layers"] = st.session_state.get("mask_layers", [])
    recipe["removal_mask_layers"] = st.session_state.get("removal_mask_layers", [])
    recipe["curve_extra_points"] = st.session_state.get("curve_extra_points", [])
    return recipe


def _restore_recipe_to_state(recipe: dict) -> None:
    """Write a saved recipe (from Undo/Redo or a loaded project) into the session_state keys the
    corresponding widgets read their initial value from.

    Streamlit forbids writing to a widget's session_state key after that widget has already been
    instantiated in the current script run. So this must only be called before ANY tab widget below
    is created -- in practice, right at the top of the very next run after a st.rerun() triggered by
    Undo, Redo, or loading a project (see the "_pending_restore" handling right after the page
    style, before any tab is built).

    Every restore path (Undo/Redo, Reset, snapshots, presets, Load Project, Auto Enhance, the
    on-canvas editors' commits) funnels through here, so the recipe is sanitized here once
    (app.edit_schema.sanitize_recipe): a hand-edited, older, or newer preset/project with a value
    outside a slider's range used to raise StreamlitValueAboveMaxError and take the page down;
    now it's clamped, a wrong type falls back to the default, and unknown keys are dropped.
    """
    recipe = sanitize_recipe(recipe)
    for key, widget_key in RECIPE_WIDGET_KEYS.items():
        if key in recipe:
            st.session_state[widget_key] = recipe[key]
    if "mask_layers" in recipe:
        st.session_state.mask_layers = recipe["mask_layers"]
    if "removal_mask_layers" in recipe:
        st.session_state.removal_mask_layers = recipe["removal_mask_layers"]
    if "curve_extra_points" in recipe:
        st.session_state.curve_extra_points = recipe["curve_extra_points"]
    if "layers" in recipe:
        st.session_state.image_layers = recipe["layers"]


def _decode_painted_mask(mask_b64: str) -> Image.Image:
    """The on-canvas paint editor (_paint_editor_html) always draws in solid white at varying
    alpha -- alpha 0 = unpainted, alpha 255 = fully selected -- so the alpha channel of the PNG
    it exports *is* the grayscale mask. Deliberately returned at whatever (smaller-than-source)
    resolution the canvas was drawn at: MaskLayer.normalized() already resizes every mask kind to
    the source's size before use, so upscaling here too would be redundant work."""
    raw = base64.b64decode(mask_b64)
    return Image.open(io.BytesIO(raw)).convert("RGBA").split()[3]


def _build_mask_image(source: Image.Image, kind: str, params: dict) -> Image.Image:
    if kind == "Luminance":
        return luminance_mask(source, params["low"], params["high"])
    if kind == "Color":
        return color_mask(source, params["color"], params["tolerance"])
    if kind == "Radial":
        # Center/radius/feather as fractions of the frame (they used to be ignored: always a
        # centered circle).
        return radial_mask(
            source.size, (float(params.get("center_x", 0.5)), float(params.get("center_y", 0.5))),
            float(params.get("radius", 0.5)), float(params.get("feather", 0.25)),
        )
    if kind == "Painted":
        return _decode_painted_mask(params["mask_b64"])
    if kind == "Magic Wand":
        return magic_wand_mask(source, params["click_x"], params["click_y"], params["tolerance"], params.get("contiguous", True))
    return gradient_mask(source.size, "horizontal" if "Horizontal" in kind else "vertical")


# Whether the tool tab being rendered is the one on screen (F1 in the improvement report). Every
# tab's body still runs on every rerun -- every widget must be instantiated each pass or Streamlit
# drops its state -- but display-only work (editor iframes with base64 backgrounds, preview
# thumbnails, face detection) is skipped for the tabs that aren't visible. Set before each tab
# body from st.tabs(..., on_change="rerun")'s `.open`.
_TAB_VISIBLE = True


def _tab_open(tab) -> bool:
    is_open = getattr(tab, "open", None)
    return True if is_open is None else bool(is_open)


def _show_component(build) -> None:
    """components.html() for an on-canvas editor, only when its tab is visible -- `build()` returns
    (markup, height) and isn't even called otherwise (it base64-encodes a preview image). The
    editor's hidden commit text_area is rendered by the caller regardless, so its state survives."""
    if _TAB_VISIBLE:
        markup, height = build()
        components.html(markup, height=height)


def _render_selection_stack(state_key: str, key_prefix: str, source: Image.Image, include_opacity: bool = True) -> str:
    """Shared "add a named selection (Luminance/Color/gradient/Radial/Painted/Magic Wand), then
    list/edit/remove existing ones" UI used by both the Masks tab (local tone/color adjustments)
    and the Repair tab (object removal) -- same selection tools, a different session_state list
    and widget-key prefix per caller so the two tabs' widgets don't collide. Returns the list's
    _selections_signature(); the caller turns it into the combined mask via _combined_mask_png()
    and decides what to do with it (masked adjustments vs. inpainting).
    """
    if state_key not in st.session_state:
        st.session_state[state_key] = []
    layers = st.session_state[state_key]
    # Widget keys below are per-layer, keyed by this id rather than by list index -- a layer's
    # position shifts whenever an earlier one is removed, and an index-based key would then point
    # a later layer's checkbox/slider at the *previous occupant's* still-cached session_state
    # value instead of its own (a real bug found in review: removing layer B silently corrupted
    # layer C's enabled/soften values). setdefault() backfills an id for layers restored from a
    # project/recipe saved before this field existed.
    for layer in layers:
        layer.setdefault("id", uuid.uuid4().hex)

    with st.expander("➕ Add a selection", expanded=not layers):
        new_kind = st.selectbox(
            "Selection Type",
            ["Luminance", "Color", "Horizontal gradient", "Vertical gradient", "Radial", "Painted", "Magic Wand"],
            key=f"{key_prefix}-new-kind",
        )
        if new_kind == "Magic Wand":
            new_tolerance = st.slider("Tolerance", 1, 150, 40, key=f"{key_prefix}-wand-tol")
            new_contiguous = st.checkbox("Contiguous (connected region only)", value=True, key=f"{key_prefix}-wand-contig")
            st.caption("Click a point on the photo below to select similar colors around it.")
            _show_component(lambda: _wand_picker_html(preview_image(source, (480, 480)), key_prefix))
            with st.container(key=f"{key_prefix}-wand-hidden"):
                pending_wand = st.text_area("Magic wand click", key=f"{key_prefix}-wand-data", label_visibility="collapsed")
            # Same "only a genuine new commit adds a layer" guard as the Painted branch below --
            # clicking the exact same spot twice in a row is correctly a no-op, not a duplicate.
            if _consume_new_commit(pending_wand, f"{key_prefix}-wand-committed"):
                try:
                    click = json.loads(pending_wand)
                    click_x, click_y = float(click["x"]), float(click["y"])
                except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                    pass
                else:
                    layers.append(
                        {
                            "id": uuid.uuid4().hex,
                            "name": f"Magic Wand {len(layers) + 1}",
                            "kind": "Magic Wand",
                            "params": {"click_x": click_x, "click_y": click_y, "tolerance": new_tolerance, "contiguous": new_contiguous},
                            "soften": 0,
                            "invert": False,
                            "opacity": 1.0,
                            "enabled": True,
                        }
                    )
                    _request_rerun()
        elif new_kind == "Painted":
            st.caption('Paint directly on the photo below — brush, radial, or gradient, plus an eraser. Click "Add to selections" inside the canvas when done.')
            _show_component(lambda: _paint_editor_html(preview_image(source, (480, 480)), key_prefix))
            with st.container(key=f"{key_prefix}-paint-hidden"):
                pending_paint = st.text_area("Painted mask data", key=f"{key_prefix}-paint-data", label_visibility="collapsed")
            # Streamlit forbids writing to a widget's session_state key after it's already been
            # instantiated this run (the same rule Undo/Redo/_pending_restore work around
            # elsewhere), so this can't just clear the field after reading it. Instead, remember
            # the last string actually turned into a layer and compare against that -- since a
            # fresh components.html() iframe always starts with a blank canvas, only a genuine new
            # commit (the user drew something and clicked "Add to selections" again) produces
            # different base64 data; clicking it twice without drawing anything new is correctly
            # a no-op rather than a duplicate layer.
            if _consume_new_commit(pending_paint, f"{key_prefix}-paint-committed"):
                layers.append(
                    {
                        "id": uuid.uuid4().hex,
                        "name": f"Painted {len(layers) + 1}",
                        "kind": "Painted",
                        "params": {"mask_b64": pending_paint},
                        "soften": 0,
                        "invert": False,
                        "opacity": 1.0,
                        "enabled": True,
                    }
                )
                _request_rerun()
        else:
            new_params: dict = {}
            if new_kind == "Luminance":
                new_params["low"] = st.slider("Min Luma", 0, 255, 0, key=f"{key_prefix}-new-lmin")
                new_params["high"] = st.slider("Max Luma", 0, 255, 255, key=f"{key_prefix}-new-lmax")
            elif new_kind == "Color":
                new_color_hex = st.color_picker("Target Color", "#806450", key=f"{key_prefix}-new-color")
                new_params["color"] = tuple(int(new_color_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
                new_params["tolerance"] = st.slider("Color Tolerance", 0, 255, 50, key=f"{key_prefix}-new-ctol")
            elif new_kind == "Radial":
                radial_c1, radial_c2 = st.columns(2)
                new_params["center_x"] = radial_c1.slider("Center X", 0.0, 1.0, 0.5, 0.01, key=f"{key_prefix}-new-rcx")
                new_params["center_y"] = radial_c2.slider("Center Y", 0.0, 1.0, 0.5, 0.01, key=f"{key_prefix}-new-rcy")
                new_params["radius"] = radial_c1.slider("Radius", 0.05, 1.5, 0.5, 0.01, key=f"{key_prefix}-new-rr")
                new_params["feather"] = radial_c2.slider("Feather", 0.01, 1.0, 0.25, 0.01, key=f"{key_prefix}-new-rf")
            new_soften = st.slider("Soften edges (px)", 0, 40, 0, key=f"{key_prefix}-new-soften")
            new_invert = st.checkbox("Invert selection", key=f"{key_prefix}-new-invert")
            new_name = st.text_input("Layer name", value=f"{new_kind} {len(layers) + 1}", key=f"{key_prefix}-new-name")
            if st.button("Add selection", key=f"{key_prefix}-add-btn"):
                layers.append(
                    {
                        "id": uuid.uuid4().hex,
                        "name": new_name.strip() or new_kind,
                        "kind": new_kind,
                        "params": new_params,
                        "soften": new_soften,
                        "invert": new_invert,
                        "opacity": 1.0,
                        "enabled": True,
                    }
                )
                _request_rerun()

    for idx, layer in enumerate(layers):
        layer_id = layer["id"]
        status_icon = "✅" if layer["enabled"] else "⬜"
        with st.expander(f"{status_icon} {layer['name']} — {layer['kind']}", expanded=False):
            layer["name"] = st.text_input("Name", layer["name"], key=f"{key_prefix}-name-{layer_id}")
            layer["enabled"] = st.checkbox("Enabled", layer["enabled"], key=f"{key_prefix}-en-{layer_id}")
            layer["invert"] = st.checkbox("Invert", layer["invert"], key=f"{key_prefix}-inv-{layer_id}")
            if include_opacity:
                layer["opacity"] = st.slider("Opacity", 0.0, 1.0, layer["opacity"], 0.05, key=f"{key_prefix}-op-{layer_id}")
            layer["soften"] = st.slider("Soften edges (px)", 0, 40, layer.get("soften", 0), key=f"{key_prefix}-sf-{layer_id}")
            combine_choices = list(COMBINE_MODES)
            layer["combine"] = st.selectbox(
                "Combine with the selections above", combine_choices, index=combine_choices.index(layer.get("combine", "add")),
                format_func=lambda mode: {"add": "Add (union)", "subtract": "Subtract", "intersect": "Intersect"}[mode],
                key=f"{key_prefix}-comb-{layer_id}", disabled=idx == 0 or layer.get("adjust", False),
            )
            if layer["kind"] == "Radial":
                params = layer["params"]
                edit_c1, edit_c2 = st.columns(2)
                params["center_x"] = edit_c1.slider("Center X", 0.0, 1.0, float(params.get("center_x", 0.5)), 0.01, key=f"{key_prefix}-rcx-{layer_id}")
                params["center_y"] = edit_c2.slider("Center Y", 0.0, 1.0, float(params.get("center_y", 0.5)), 0.01, key=f"{key_prefix}-rcy-{layer_id}")
                params["radius"] = edit_c1.slider("Radius", 0.05, 1.5, float(params.get("radius", 0.5)), 0.01, key=f"{key_prefix}-rr-{layer_id}")
                params["feather"] = edit_c2.slider("Feather", 0.01, 1.0, float(params.get("feather", 0.25)), 0.01, key=f"{key_prefix}-rf-{layer_id}")
            if include_opacity:
                # Per-selection adjustment set (B8): this selection is applied on its own with these
                # values instead of joining the combined mask that the Target sliders below drive --
                # e.g. brighten a face and darken the sky in the same edit.
                layer["adjust"] = st.checkbox("Own adjustments", layer.get("adjust", False), key=f"{key_prefix}-own-{layer_id}")
                if layer["adjust"]:
                    adjustments = layer.setdefault("adjustments", {})
                    own_c1, own_c2 = st.columns(2)
                    adjustments["exposure_ev"] = own_c1.slider("Exposure (EV)", -5.0, 5.0, float(adjustments.get("exposure_ev", 0.0)), 0.05, key=f"{key_prefix}-own-ev-{layer_id}")
                    adjustments["contrast"] = own_c2.slider("Contrast", -100, 100, int(adjustments.get("contrast", 0)), key=f"{key_prefix}-own-ct-{layer_id}")
                    adjustments["saturation"] = own_c1.slider("Saturation", -100, 100, int(adjustments.get("saturation", 0)), key=f"{key_prefix}-own-sat-{layer_id}")
                    adjustments["clarity"] = own_c2.slider("Clarity", -100, 100, int(adjustments.get("clarity", 0)), key=f"{key_prefix}-own-cl-{layer_id}")
                    adjustments["temperature"] = own_c1.slider("Temperature", -100, 100, int(adjustments.get("temperature", 0)), key=f"{key_prefix}-own-tmp-{layer_id}")
                    adjustments["tint"] = own_c2.slider("Tint", -100, 100, int(adjustments.get("tint", 0)), key=f"{key_prefix}-own-tnt-{layer_id}")
            if layer["kind"] == "Painted":
                # The only selection kind with a spatial shape that can't be re-derived from a
                # handful of numeric params (Luminance/Color/gradient/Radial/Magic Wand all can,
                # from their existing name/enabled/invert/opacity/soften controls above) -- a
                # second pass at the canvas, not just parameter tweaks, needs its own toggle.
                if st.checkbox("🎨 Edit Painting", key=f"{key_prefix}-editpaint-toggle-{layer_id}"):
                    edit_prefix = f"{key_prefix}-ep-{layer_id}"
                    _show_component(lambda: _paint_editor_html(
                        preview_image(source, (480, 480)), edit_prefix,
                        preload_mask_b64=layer["params"].get("mask_b64"), commit_label="💾 Update Painting",
                    ))
                    with st.container(key=f"{edit_prefix}-paint-hidden"):
                        pending_edit = st.text_area("Edited painted mask data", key=f"{edit_prefix}-paint-data", label_visibility="collapsed")
                    # Same "only a genuine new commit updates anything" guard as adding a fresh
                    # Painted selection -- clicking "Update Painting" twice without drawing
                    # anything new between clicks is correctly a no-op.
                    if _consume_new_commit(pending_edit, f"{edit_prefix}-paint-committed"):
                        layer["params"]["mask_b64"] = pending_edit
                        _request_rerun()
            if st.button("🗑️ Remove", key=f"{key_prefix}-rm-{layer_id}"):
                layers.pop(idx)
                _request_rerun()
                # Stop rendering this list for the rest of this pass rather than continuing
                # enumerate() over a list we just mutated mid-iteration (deferring the rerun --
                # see _request_rerun() -- means this loop no longer aborts here the way an
                # immediate st.rerun() used to). The deferred rerun re-renders the remaining
                # layers correctly on the very next pass -- true now that each layer's widget
                # keys are id-based rather than index-based (see the setdefault() above): a
                # later layer shifting into a removed layer's old index no longer inherits that
                # index's cached session_state.
                break

    # The pixels themselves are built (and cached) by _combined_mask_png() from this signature,
    # not here -- rebuilding every selection at full resolution on every rerun was one of the
    # biggest per-rerun costs (see _selection_mask()). Selections with their own adjustments are
    # rendered separately (see _own_adjustment_layers), not into this combined mask.
    return _selections_signature([layer for layer in layers if not layer.get("adjust")])


def _hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), 255)


_STYLE_EFFECT_PREFIXES = ("drop_shadow", "outer_glow", "stroke", "color_overlay")
_STYLE_COLOR_KEYS = ("drop_shadow_color", "outer_glow_color", "stroke_color", "color_overlay_color")


def _resolved_layer_styles(styles: dict) -> dict:
    """`styles` stores its colors as `st.color_picker` hex strings (e.g. "#FF0000") -- convert to
    the plain RGB int tuples app.layer_styles.apply_layer_styles() expects, leaving every other
    field (enabled flags, blur/offset/width/opacity numbers) untouched."""
    resolved = dict(styles)
    for key in _STYLE_COLOR_KEYS:
        if key in resolved:
            resolved[key] = _hex_to_rgba(resolved[key])[:3]
    return resolved


def _scaled_styles(styles: dict, scale: float) -> dict:
    """Layer Style sizes are pixels of the full-resolution photo; a proxy render scales them."""
    if scale == 1.0:
        return styles
    scaled = dict(styles)
    for key in ("drop_shadow_offset_x", "drop_shadow_offset_y", "drop_shadow_blur", "outer_glow_blur", "stroke_width"):
        if key in scaled:
            scaled[key] = scaled[key] * scale
    return scaled


def _layer_dict_to_image(layer: dict, canvas_size: tuple[int, int], scale: float = 1.0) -> Image.Image:
    """Rasterize one entry of st.session_state.image_layers into an actual RGBA image -- an
    "Image" layer decodes its stored upload (resized to the canvas's own size so Layer Styles below
    compute their blur/offset/width in the same real pixel scale for either kind); a "Text" layer
    renders fresh from its parametric fields every time (cheap: a single PIL draw call), the same
    "stay parametric, only rasterize on demand" approach app.advanced_masks.MaskLayer's selections
    already use, so editing the text after adding it doesn't require re-uploading anything. Shared
    by _render_layer_stack() (the backend wire payload) and the Save Project handler so the two
    never diverge on how a layer's kind maps to pixels.

    Layer Styles (Drop Shadow/Outer Glow/Stroke/Color Overlay, see app.layer_styles) are applied
    here too, after the base content is rasterized and before it becomes a WorkspaceLayer -- so the
    normal opacity/blend-mode compositing on the resulting WorkspaceLayer runs on top of whatever
    the styles produce, unmodified.
    """
    if layer.get("kind") == "text":
        # font_size_pct: % of the canvas's short edge (F5) -- the same look on a 2 MP or 45 MP
        # photo and on a preview proxy. Older layers stored pixels (font_size).
        if "font_size_pct" in layer:
            font_size = float(layer["font_size_pct"]) / 100.0 * min(canvas_size)
        else:
            font_size = float(layer.get("font_size", 48)) * scale
        content = render_text_layer(
            canvas_size, layer.get("text", ""), font_size, _hex_to_rgba(layer.get("color", "#FFFFFF")),
            layer.get("align", "left"), layer.get("x_pct", 50.0), layer.get("y_pct", 50.0),
        )
    elif layer.get("kind") == "shape":
        # fill_color/stroke_color are stored as st.color_picker hex strings, same UI convention as
        # a Text layer's "color" -- render_shape_layer() wants plain RGB int tuples.
        shape_params = {
            **layer,
            "fill_color": _hex_to_rgba(layer.get("fill_color", "#FFFFFF"))[:3],
            "stroke_color": _hex_to_rgba(layer.get("stroke_color", "#000000"))[:3],
            "stroke_width": max(1, round(layer.get("stroke_width", 4) * scale)),
            "line_thickness": max(1, round(layer.get("line_thickness", 6) * scale)),
        }
        content = render_shape_layer(canvas_size, shape_params)
    elif layer.get("kind") == "path":
        path_params = {
            **layer,
            "fill_color": _hex_to_rgba(layer.get("fill_color", "#FFFFFF"))[:3],
            "stroke_color": _hex_to_rgba(layer.get("stroke_color", "#FFFFFF"))[:3],
            "stroke_width": max(1, round(layer.get("stroke_width", 3) * scale)),
        }
        content = render_path_layer(canvas_size, path_params)
    else:
        content = Image.open(io.BytesIO(base64.b64decode(layer["image_b64"]))).convert("RGBA")
        if layer.get("stretch", "x_pct" not in layer):
            # Layers added before image layers had placement (and "Stretch to fill" on) fill the canvas.
            if content.size != canvas_size:
                content = content.resize(canvas_size, Image.Resampling.LANCZOS)
        else:
            content = render_image_layer(
                canvas_size, content, layer.get("x_pct", 50.0), layer.get("y_pct", 50.0),
                layer.get("scale_pct", 100.0), layer.get("rotation_deg", 0.0),
            )

    styles = layer.get("styles")
    if styles and any(styles.get(f"{prefix}_enabled") for prefix in _STYLE_EFFECT_PREFIXES):
        content = apply_layer_styles(content, _resolved_layer_styles(_scaled_styles(styles, scale)))
    return content


def _render_layer_styles(layer: dict, layer_id: str) -> None:
    """Layer Styles sub-section inside one layer's own expander in _render_layer_stack below
    (Drop Shadow/Outer Glow/Stroke/Color Overlay -- see app.layer_styles.apply_layer_styles, which
    this panel's values ultimately feed via _resolved_layer_styles). Plain checkboxes + inline
    widget rows rather than a further st.expander: Streamlit doesn't allow nesting one expander
    inside another, and this already lives inside the layer's own."""
    layer.setdefault("styles", {})
    styles = layer["styles"]
    st.markdown("**Layer Styles**")

    styles["drop_shadow_enabled"] = st.checkbox(
        "Drop Shadow", styles.get("drop_shadow_enabled", False), key=f"l-ds-en-{layer_id}",
        help="Casts a soft shadow behind this layer's own silhouette, offset and blurred below.",
    )
    if styles["drop_shadow_enabled"]:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            styles["drop_shadow_offset_x"] = st.slider("Offset X", -50, 50, int(styles.get("drop_shadow_offset_x", 8)), key=f"l-ds-ox-{layer_id}", help="Horizontal shadow displacement in pixels -- negative shifts it left, positive right.")
        with c2:
            styles["drop_shadow_offset_y"] = st.slider("Offset Y", -50, 50, int(styles.get("drop_shadow_offset_y", 8)), key=f"l-ds-oy-{layer_id}", help="Vertical shadow displacement in pixels -- negative shifts it up, positive down.")
        with c3:
            styles["drop_shadow_blur"] = st.slider("Blur", 0, 40, int(styles.get("drop_shadow_blur", 6)), key=f"l-ds-bl-{layer_id}", help="How soft the shadow's edge is -- 0 is a hard-edged silhouette, higher spreads it into a soft blur.")
        with c4:
            styles["drop_shadow_color"] = st.color_picker("Color", styles.get("drop_shadow_color", "#000000"), key=f"l-ds-col-{layer_id}")
        styles["drop_shadow_opacity"] = st.slider("Shadow Opacity", 0.0, 1.0, float(styles.get("drop_shadow_opacity", 0.6)), 0.01, key=f"l-ds-op-{layer_id}", help="How dark/visible the shadow is, independent of the layer's own opacity.")

    styles["outer_glow_enabled"] = st.checkbox(
        "Outer Glow", styles.get("outer_glow_enabled", False), key=f"l-og-en-{layer_id}",
        help="Adds a soft halo of color radiating outward from this layer's silhouette.",
    )
    if styles["outer_glow_enabled"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            styles["outer_glow_blur"] = st.slider("Blur", 0, 60, int(styles.get("outer_glow_blur", 10)), key=f"l-og-bl-{layer_id}", help="How far the glow spreads outward from the layer's edge.")
        with c2:
            styles["outer_glow_color"] = st.color_picker("Color", styles.get("outer_glow_color", "#FFDC78"), key=f"l-og-col-{layer_id}")
        with c3:
            styles["outer_glow_opacity"] = st.slider("Glow Opacity", 0.0, 1.0, float(styles.get("outer_glow_opacity", 0.75)), 0.01, key=f"l-og-op-{layer_id}", help="How strong/visible the glow is.")

    styles["stroke_enabled"] = st.checkbox(
        "Stroke", styles.get("stroke_enabled", False), key=f"l-st-en-{layer_id}",
        help="Outlines this layer's silhouette with a solid-color border.",
    )
    if styles["stroke_enabled"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            styles["stroke_width"] = st.slider("Width", 1, 40, int(styles.get("stroke_width", 4)), key=f"l-st-w-{layer_id}", help="Border thickness in pixels.")
        with c2:
            styles["stroke_color"] = st.color_picker("Color", styles.get("stroke_color", "#FFFFFF"), key=f"l-st-col-{layer_id}")
        with c3:
            styles["stroke_opacity"] = st.slider("Stroke Opacity", 0.0, 1.0, float(styles.get("stroke_opacity", 1.0)), 0.01, key=f"l-st-op-{layer_id}", help="How solid/visible the outline is.")

    styles["color_overlay_enabled"] = st.checkbox(
        "Color Overlay", styles.get("color_overlay_enabled", False), key=f"l-co-en-{layer_id}",
        help="Tints this layer's visible pixels with a flat color, blended by the opacity below.",
    )
    if styles["color_overlay_enabled"]:
        c1, c2 = st.columns(2)
        with c1:
            styles["color_overlay_color"] = st.color_picker("Color", styles.get("color_overlay_color", "#FF0000"), key=f"l-co-col-{layer_id}")
        with c2:
            styles["color_overlay_opacity"] = st.slider("Overlay Opacity", 0.0, 1.0, float(styles.get("color_overlay_opacity", 0.6)), 0.01, key=f"l-co-op-{layer_id}", help="How strongly the color tints the layer -- 0 has no effect, 1 fully replaces its color.")


def _render_pen_path_editor(source: Image.Image, key_prefix: str, points: list[dict], closed: bool) -> tuple[list[dict], bool]:
    """The on-canvas Pen tool editor (_pen_editor_html) plus its Closed Path/Clear controls --
    shared between the "Add a layer" draft flow (building a brand new path before it becomes a
    layer) and an existing Path layer's own per-layer expander (refining it afterward). Returns the
    possibly-updated (points, closed); the caller is responsible for writing that back into
    wherever it actually lives (a draft session_state pair, or this specific layer's own dict),
    since this helper has no way to know which."""
    _show_component(lambda: _pen_editor_html(preview_image(source, (480, 480)), key_prefix, points, closed))
    with st.container(key=f"{key_prefix}-hidden"):
        pending_points = st.text_area("Pen path points", key=f"{key_prefix}-data", label_visibility="collapsed")
    if _consume_new_commit(pending_points, f"{key_prefix}-committed"):
        try:
            parsed = json.loads(pending_points)
            points = [
                {
                    "x_pct": float(point["x_pct"]), "y_pct": float(point["y_pct"]),
                    "handle_dx_pct": float(point.get("handle_dx_pct", 0.0)), "handle_dy_pct": float(point.get("handle_dy_pct", 0.0)),
                }
                for point in parsed
            ]
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            pass
        else:
            _request_rerun()
    closed = st.checkbox("Closed Path", closed, key=f"{key_prefix}-closed")
    if points and st.button("🗑️ Clear Points", key=f"{key_prefix}-clear"):
        points = []
        _request_rerun()
    return points, closed


def _render_layer_stack(source: Image.Image) -> str | None:
    """Layers tab: any number of stacked Image or Text layers, each with its own name/visibility/
    opacity/blend mode, reorderable and duplicable -- the same "add, then list/edit/remove" shape
    _render_selection_stack established for Masks/Repair. The compositing engine underneath
    (app.layer_workspace.LayerWorkspace) already supported an arbitrary number of layers; only this
    panel was ever limited to exactly one uploaded overlay image with no text option at all. Returns
    the stack's JSON signature for _layers_payload() (which builds process_via_backend's
    `layers_bytes`, app.layer_workspace.layers_to_json's wire format), or None when it's empty.

    Storage/composite order is bottom-to-top (index 0 sits directly on the photo, the last entry is
    on top) -- also the order layers are listed in below, called out in a caption since Photoshop's
    own panel lists the topmost layer first.
    """
    if "image_layers" not in st.session_state:
        st.session_state.image_layers = []
    layers = st.session_state.image_layers
    for layer in layers:
        layer.setdefault("id", uuid.uuid4().hex)
        layer.setdefault("kind", "image")  # backfill for layers added before Text layers existed

    with st.expander("➕ Add a layer", expanded=not layers):
        new_layer_kind = st.radio("Layer type", ["Image", "Text", "Shape", "Pen"], horizontal=True, key="l-new-kind")
        if new_layer_kind == "Text":
            if st.button("Add Text Layer", key="l-add-text-btn"):
                layers.append(
                    {
                        "id": uuid.uuid4().hex,
                        "kind": "text",
                        "name": f"Text {len(layers) + 1}",
                        "text": "Text",
                        "font_size_pct": 6.0,
                        "color": "#FFFFFF",
                        "align": "left",
                        "x_pct": 50.0,
                        "y_pct": 50.0,
                        "opacity": 1.0,
                        "blend_mode": "normal",
                        "visible": True,
                    }
                )
                _request_rerun()
        elif new_layer_kind == "Shape":
            new_shape_type = st.selectbox("Shape", SHAPE_TYPES, key="l-new-shape-type")
            if st.button("Add Shape Layer", key="l-add-shape-btn"):
                layers.append(
                    {
                        "id": uuid.uuid4().hex,
                        "kind": "shape",
                        "name": f"{new_shape_type} {len(layers) + 1}",
                        "shape_type": new_shape_type,
                        "x_pct": 50.0,
                        "y_pct": 50.0,
                        "width_pct": 30.0,
                        "height_pct": 30.0,
                        "rotation_deg": 0.0,
                        "fill_enabled": True,
                        "fill_color": "#FFFFFF",
                        "fill_opacity": 1.0,
                        "stroke_enabled": False,
                        "stroke_color": "#000000",
                        "stroke_width": 4,
                        "sides": 5,
                        "inner_ratio": 0.5,
                        "line_thickness": 6,
                        "opacity": 1.0,
                        "blend_mode": "normal",
                        "visible": True,
                    }
                )
                _request_rerun()
        elif new_layer_kind == "Pen":
            st.caption('Click points on the photo below to trace a path, then "Add Path Layer" when done.')
            # draft_closed itself needs no separate write-back: it's the "l-path-draft-closed"
            # checkbox's own widget key, already persisted by Streamlit the moment that widget
            # renders -- re-assigning it here would violate the "never write into an
            # already-instantiated widget's session_state key" rule (see CLAUDE.md) and crash.
            # draft_points has no widget of its own (it's a plain list, like mask_layers), so it
            # does need this explicit write-back to survive into the next rerun.
            draft_points, draft_closed = _render_pen_path_editor(
                source, "l-path-draft", st.session_state.get("l-path-draft-points", []), st.session_state.get("l-path-draft-closed", False),
            )
            st.session_state["l-path-draft-points"] = draft_points
            if st.button("➕ Add Path Layer", key="l-add-path-btn", disabled=len(draft_points) < 2):
                layers.append(
                    {
                        "id": uuid.uuid4().hex,
                        "kind": "path",
                        "name": f"Path {len(layers) + 1}",
                        "points": draft_points,
                        "closed": draft_closed,
                        "fill_enabled": False,
                        "fill_color": "#FFFFFF",
                        "fill_opacity": 1.0,
                        "stroke_enabled": True,
                        "stroke_color": "#FFFFFF",
                        "stroke_width": 3,
                        "opacity": 1.0,
                        "blend_mode": "normal",
                        "visible": True,
                    }
                )
                # Only the points draft is reset -- "l-path-draft-closed" is the Closed Path
                # checkbox's own widget key, already instantiated this run, so it can't be written
                # to directly (same rule as the comment above _render_pen_path_editor's call). Left
                # as whatever the user last set it to for the next new path being drafted, a minor,
                # harmless carry-over rather than worth a bigger _pending_restore-style mechanism.
                st.session_state["l-path-draft-points"] = []
                _request_rerun()
        else:
            new_layer_file = st.file_uploader("Layer image", type=["png", "jpg", "jpeg", "webp"], key="l-new-file")
            # Streamlit keeps a file_uploader's value across reruns until the user picks a different
            # file or clears it -- adding on every rerun where it's truthy would append the same
            # layer over and over. file_id is a fresh identifier per upload, so this only fires once
            # per genuinely new file, the same "add exactly once per real commit" shape
            # _consume_new_commit gives the on-canvas editors elsewhere in this file.
            if new_layer_file is not None:
                file_id = getattr(new_layer_file, "file_id", new_layer_file.name)
                if st.session_state.get("l-last-added-file-id") != file_id:
                    st.session_state["l-last-added-file-id"] = file_id
                    try:
                        decoded = Image.open(io.BytesIO(new_layer_file.getvalue())).convert("RGBA")
                    except OSError:
                        st.error("⚠️ Couldn't read that layer image — it may be corrupted.")
                    else:
                        buffer = io.BytesIO()
                        decoded.save(buffer, format="PNG")
                        layers.append(
                            {
                                "id": uuid.uuid4().hex,
                                "kind": "image",
                                "name": Path(new_layer_file.name).stem or f"Layer {len(layers) + 1}",
                                "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                                "x_pct": 50.0, "y_pct": 50.0, "scale_pct": 100.0, "rotation_deg": 0.0, "stretch": False,
                                "opacity": 1.0,
                                "blend_mode": "normal",
                                "visible": True,
                            }
                        )
                        _request_rerun()

    if not layers:
        st.caption("No layers yet — add one above.")
    else:
        st.caption("Listed bottom-to-top: the last layer here sits on top, same as the composite order.")

    blend_mode_choices = list(BLEND_MODE_LABELS)
    align_choices = ["left", "center", "right"]
    for idx, layer in enumerate(layers):
        layer_id = layer["id"]
        status_icon = "✅" if layer["visible"] else "⬜"
        kind_label = {"text": "Text", "shape": layer.get("shape_type", "Shape"), "path": "Path"}.get(layer["kind"], BLEND_MODE_LABELS[layer["blend_mode"]])
        with st.expander(f"{status_icon} {layer['name']} — {kind_label}", expanded=False):
            # Split into sub-tabs rather than one long scroll -- a layer's kind-specific fields,
            # its Layer Styles, and its name/opacity/blend/reorder controls used to all render one
            # after another in the same expander, which got dense once a layer could be four very
            # different kinds (Image/Text/Shape/Pen) each with several controls, on top of Layer
            # Styles' own four effects. Nesting st.tabs() inside st.expander() is allowed (unlike
            # nesting a second st.expander(), which Streamlit forbids -- the reason
            # _render_layer_styles already uses checkboxes instead of its own expander) and, like
            # every other tab in this app, every sub-tab's body still executes on every rerun
            # regardless of which one is visually selected -- so none of the widget-instantiation
            # rules elsewhere in this file (deferred reruns, the break-after-mutation buttons below)
            # need to change for this split.
            tab_content, tab_styles, tab_settings = st.tabs(["🎨 Content", "✨ Styles", "⚙️ Settings"])
            with tab_content:
                if layer["kind"] == "text":
                    layer["text"] = st.text_area("Text", layer.get("text", "Text"), key=f"l-text-{layer_id}")
                    text_col1, text_col2 = st.columns(2)
                    with text_col1:
                        if "font_size_pct" not in layer:  # a pixel size from an older layer -> % of the short edge
                            layer["font_size_pct"] = round(min(50.0, max(0.5, layer.get("font_size", 48) / min(source.size) * 100)), 1)
                        layer["font_size_pct"] = st.slider(
                            "Font size (% of short edge)", 0.5, 50.0, float(layer["font_size_pct"]), 0.5, key=f"l-fsizepct-{layer_id}",
                            help="Relative to the photo, so the text looks the same at any resolution.",
                        )
                    with text_col2:
                        layer["color"] = st.color_picker("Color", layer.get("color", "#FFFFFF"), key=f"l-fcolor-{layer_id}")
                    layer["align"] = st.selectbox(
                        "Alignment", align_choices, index=align_choices.index(layer.get("align", "left")), key=f"l-falign-{layer_id}",
                    )
                    pos_col1, pos_col2 = st.columns(2)
                    with pos_col1:
                        layer["x_pct"] = st.slider("Horizontal position (%)", 0.0, 100.0, float(layer.get("x_pct", 50.0)), key=f"l-fx-{layer_id}")
                    with pos_col2:
                        layer["y_pct"] = st.slider("Vertical position (%)", 0.0, 100.0, float(layer.get("y_pct", 50.0)), key=f"l-fy-{layer_id}")
                    # A small live preview, computed locally (not through the backend) -- same
                    # "cheap enough to recompute every rerun" pattern as the Masks tab's combined
                    # overlay preview. A dark neutral background so light-colored text stays visible,
                    # since the layer's own image is otherwise fully transparent outside the glyphs.
                    preview_ratio = min(260 / source.width, 260 / source.height, 1.0)
                    preview_size = (max(1, int(source.width * preview_ratio)), max(1, int(source.height * preview_ratio)))
                    preview_bg = Image.new("RGBA", preview_size, (40, 40, 40, 255))
                    preview_text = render_text_layer(
                        preview_size, layer["text"], layer["font_size_pct"] / 100.0 * min(preview_size), _hex_to_rgba(layer["color"]),
                        layer["align"], layer["x_pct"], layer["y_pct"],
                    )
                    if _TAB_VISIBLE:
                        st.image(Image.alpha_composite(preview_bg, preview_text))
                elif layer["kind"] == "shape":
                    layer["shape_type"] = st.selectbox(
                        "Shape", SHAPE_TYPES, index=SHAPE_TYPES.index(layer.get("shape_type", "Rectangle")), key=f"l-shp-type-{layer_id}",
                    )
                    shp_pos1, shp_pos2 = st.columns(2)
                    with shp_pos1:
                        layer["x_pct"] = st.slider("Horizontal position (%)", 0.0, 100.0, float(layer.get("x_pct", 50.0)), key=f"l-shp-x-{layer_id}")
                    with shp_pos2:
                        layer["y_pct"] = st.slider("Vertical position (%)", 0.0, 100.0, float(layer.get("y_pct", 50.0)), key=f"l-shp-y-{layer_id}")
                    shp_size1, shp_size2 = st.columns(2)
                    with shp_size1:
                        layer["width_pct"] = st.slider("Width (%)", 1.0, 100.0, float(layer.get("width_pct", 30.0)), key=f"l-shp-w-{layer_id}")
                    with shp_size2:
                        layer["height_pct"] = st.slider(
                            "Height (%)", 1.0, 100.0, float(layer.get("height_pct", 30.0)), key=f"l-shp-h-{layer_id}", disabled=layer["shape_type"] == "Line",
                        )
                    if layer["shape_type"] != "Ellipse":
                        layer["rotation_deg"] = st.slider("Rotation", 0.0, 360.0, float(layer.get("rotation_deg", 0.0)), key=f"l-shp-rot-{layer_id}")
                    if layer["shape_type"] in ("Polygon", "Star"):
                        shp_sides1, shp_sides2 = st.columns(2)
                        with shp_sides1:
                            layer["sides"] = st.slider(
                                "Sides" if layer["shape_type"] == "Polygon" else "Points", 3, 12, int(layer.get("sides", 5)), key=f"l-shp-sides-{layer_id}",
                            )
                        if layer["shape_type"] == "Star":
                            with shp_sides2:
                                layer["inner_ratio"] = st.slider("Inner Radius", 0.05, 0.95, float(layer.get("inner_ratio", 0.5)), key=f"l-shp-inner-{layer_id}")
                    if layer["shape_type"] == "Line":
                        shp_line1, shp_line2 = st.columns(2)
                        with shp_line1:
                            layer["line_thickness"] = st.slider("Thickness (px)", 1, 60, int(layer.get("line_thickness", 6)), key=f"l-shp-thick-{layer_id}")
                        with shp_line2:
                            layer["fill_color"] = st.color_picker("Color", layer.get("fill_color", "#FFFFFF"), key=f"l-shp-fcolor-{layer_id}")
                    else:
                        layer["fill_enabled"] = st.checkbox("Fill", layer.get("fill_enabled", True), key=f"l-shp-fen-{layer_id}")
                        if layer["fill_enabled"]:
                            shp_fill1, shp_fill2 = st.columns(2)
                            with shp_fill1:
                                layer["fill_color"] = st.color_picker("Fill Color", layer.get("fill_color", "#FFFFFF"), key=f"l-shp-fcolor-{layer_id}")
                            with shp_fill2:
                                layer["fill_opacity"] = st.slider("Fill Opacity", 0.0, 1.0, float(layer.get("fill_opacity", 1.0)), 0.01, key=f"l-shp-fop-{layer_id}")
                        layer["stroke_enabled"] = st.checkbox("Outline", layer.get("stroke_enabled", False), key=f"l-shp-sen-{layer_id}")
                        if layer["stroke_enabled"]:
                            shp_stroke1, shp_stroke2 = st.columns(2)
                            with shp_stroke1:
                                layer["stroke_color"] = st.color_picker("Outline Color", layer.get("stroke_color", "#000000"), key=f"l-shp-scolor-{layer_id}")
                            with shp_stroke2:
                                layer["stroke_width"] = st.slider("Outline Width", 1, 40, int(layer.get("stroke_width", 4)), key=f"l-shp-sw-{layer_id}")
                    # Same locally-computed live preview pattern as the Text branch above.
                    preview_ratio = min(260 / source.width, 260 / source.height, 1.0)
                    preview_size = (max(1, int(source.width * preview_ratio)), max(1, int(source.height * preview_ratio)))
                    preview_bg = Image.new("RGBA", preview_size, (40, 40, 40, 255))
                    shape_preview_params = {
                        **layer,
                        "fill_color": _hex_to_rgba(layer.get("fill_color", "#FFFFFF"))[:3],
                        "stroke_color": _hex_to_rgba(layer.get("stroke_color", "#000000"))[:3],
                    }
                    if _TAB_VISIBLE:
                        st.image(Image.alpha_composite(preview_bg, render_shape_layer(preview_size, shape_preview_params)))
                elif layer["kind"] == "path":
                    layer["points"], layer["closed"] = _render_pen_path_editor(
                        source, f"l-path-{layer_id}", layer.get("points", []), layer.get("closed", False),
                    )
                    if layer["closed"]:
                        layer["fill_enabled"] = st.checkbox("Fill", layer.get("fill_enabled", False), key=f"l-path-fen-{layer_id}")
                        if layer["fill_enabled"]:
                            path_fill1, path_fill2 = st.columns(2)
                            with path_fill1:
                                layer["fill_color"] = st.color_picker("Fill Color", layer.get("fill_color", "#FFFFFF"), key=f"l-path-fcolor-{layer_id}")
                            with path_fill2:
                                layer["fill_opacity"] = st.slider("Fill Opacity", 0.0, 1.0, float(layer.get("fill_opacity", 1.0)), 0.01, key=f"l-path-fop-{layer_id}")
                    layer["stroke_enabled"] = st.checkbox("Stroke", layer.get("stroke_enabled", True), key=f"l-path-sen-{layer_id}")
                    if layer["stroke_enabled"]:
                        path_stroke1, path_stroke2 = st.columns(2)
                        with path_stroke1:
                            layer["stroke_color"] = st.color_picker("Stroke Color", layer.get("stroke_color", "#FFFFFF"), key=f"l-path-scolor-{layer_id}")
                        with path_stroke2:
                            layer["stroke_width"] = st.slider("Stroke Width", 1, 40, int(layer.get("stroke_width", 3)), key=f"l-path-sw-{layer_id}")
                else:
                    layer["stretch"] = st.checkbox(
                        "Stretch to fill the photo", layer.get("stretch", "x_pct" not in layer), key=f"l-img-stretch-{layer_id}",
                        help="Off: the image keeps its proportions and can be moved, scaled and rotated.",
                    )
                    if not layer["stretch"]:
                        img_c1, img_c2 = st.columns(2)
                        layer["x_pct"] = img_c1.slider("Horizontal position (%)", 0.0, 100.0, float(layer.get("x_pct", 50.0)), key=f"l-img-x-{layer_id}")
                        layer["y_pct"] = img_c2.slider("Vertical position (%)", 0.0, 100.0, float(layer.get("y_pct", 50.0)), key=f"l-img-y-{layer_id}")
                        layer["scale_pct"] = img_c1.slider("Scale (%)", 1.0, 400.0, float(layer.get("scale_pct", 100.0)), key=f"l-img-scale-{layer_id}", help="100 % fits the image inside the photo.")
                        layer["rotation_deg"] = img_c2.slider("Rotation", -180.0, 180.0, float(layer.get("rotation_deg", 0.0)), key=f"l-img-rot-{layer_id}")
                    if _TAB_VISIBLE:
                        thumb = Image.open(io.BytesIO(base64.b64decode(layer["image_b64"])))
                        st.image(preview_image(thumb, (160, 160)))
            with tab_styles:
                _render_layer_styles(layer, layer_id)
            with tab_settings:
                layer["name"] = st.text_input("Name", layer["name"], key=f"l-name-{layer_id}")
                layer["visible"] = st.checkbox("Visible", layer["visible"], key=f"l-vis-{layer_id}")
                layer["opacity"] = st.slider("Opacity", 0.0, 1.0, layer["opacity"], 0.01, key=f"l-op-{layer_id}")
                layer["blend_mode"] = st.selectbox(
                    "Blend Mode", blend_mode_choices, format_func=lambda mode: BLEND_MODE_LABELS[mode],
                    index=blend_mode_choices.index(layer["blend_mode"]), key=f"l-blend-{layer_id}",
                )
                # Every mutation below (reorder/duplicate/remove) is followed by `break`, not just
                # _request_rerun() -- swapping or inserting into `layers` while an outer `enumerate()`
                # over that same list keeps iterating would re-visit the layer that just got swapped/
                # shifted into a later idx this same pass, rendering its widgets a second time under
                # the same id and raising StreamlitDuplicateElementKey. The deferred rerun (see
                # _request_rerun()) redraws the rest of the list correctly on the very next pass.
                up_col, down_col, dup_col, rm_col = st.columns(4)
                with up_col:
                    if st.button("⬆️ Up", key=f"l-up-{layer_id}", disabled=idx == len(layers) - 1):
                        layers[idx], layers[idx + 1] = layers[idx + 1], layers[idx]
                        _request_rerun()
                        break
                with down_col:
                    if st.button("⬇️ Down", key=f"l-down-{layer_id}", disabled=idx == 0):
                        layers[idx], layers[idx - 1] = layers[idx - 1], layers[idx]
                        _request_rerun()
                        break
                with dup_col:
                    if st.button("📄 Duplicate", key=f"l-dup-{layer_id}"):
                        layers.insert(idx + 1, {**layer, "id": uuid.uuid4().hex, "name": f"{layer['name']} copy"})
                        _request_rerun()
                        break
                with rm_col:
                    if st.button("🗑️ Remove", key=f"l-rm-{layer_id}"):
                        layers.pop(idx)
                        _request_rerun()
                        break

    if not layers:
        return None
    # The stack's own JSON -- rasterized + serialized later (_layers_payload, cached on it) at the
    # size the pipeline composites at: the full-resolution frame, or the preview proxy's.
    return json.dumps(layers, sort_keys=True)


# Every st.cache_data below has max_entries: without it Streamlit keeps every distinct input's
# result for the life of the server process, and each value a slider passes through becomes a new
# key -- for the full-resolution caches that's a whole decoded/processed photo (~70+ MB on a large
# one) per slider position, which grew without bound over an editing session. The full-resolution
# ones keep only a few recent entries (enough for Undo/Redo or nudging a slider back to hit the
# cache); the thumbnail-sized ones can afford more.
_FULL_RES_CACHE_ENTRIES = 4
# ...and drop them after an hour unused, so a long session doesn't keep full-resolution renders of
# photos edited long ago (P5).
_FULL_RES_CACHE_TTL = 3600
_THUMBNAIL_CACHE_ENTRIES = 16


# Content-digest cache keys (P2 in the improvement report): every cached function below that
# depends on the photo takes its SHA-256 digest (hashed once per upload, see _source_digest()) as
# the hashed key and the bytes/image itself as an underscore-prefixed argument, which
# st.cache_data/st.cache_resource skip when hashing -- the old versions hashed the whole upload
# (or a ~76 MB Stage 1 PNG) on every call, on every rerun.


def _source_digest(identity: object, content: bytes) -> str:
    """SHA-256 of the open photo, computed once per distinct upload rather than per rerun --
    `identity` is the uploader's file_id (or the restored bytes object's id()), which stays the
    same across reruns until a different file is opened."""
    cached = st.session_state.get("_source_digest")
    if cached and cached[0] == identity:
        return cached[1]
    digest = image_digest(content)
    st.session_state["_source_digest"] = (identity, digest)
    return digest


@st.cache_resource(show_spinner="Opening photo…", max_entries=2)
def _decoded_source(digest: str, _content: bytes, filename: str) -> Image.Image:
    """The decoded source photo (RAW included -- a full LibRaw demosaic), decoded once per upload
    instead of on every rerun. st.cache_resource hands back the same object each time (no 72 MB
    pickle copy per rerun, unlike st.cache_data), so callers must treat it as read-only -- every
    consumer in this script only reads it or works on a .copy()/resized copy."""
    return PhotoProcessor.from_bytes(_content, filename).image


@st.cache_data(show_spinner=False, max_entries=_THUMBNAIL_CACHE_ENTRIES)
def _preset_thumbnails(digest: str, _source: Image.Image) -> dict[str, bytes]:
    """Small local preview renders for the Preset Look dropdown, so a user can see what each
    preset looks like before picking one instead of guessing from its name. Decorative only --
    generated in-process from a small thumbnail, separate from the real edit pipeline, which
    always goes through app/backend.py regardless of what's previewed here."""
    base = _source.copy()
    base.thumbnail((96, 96), Image.Resampling.LANCZOS)
    thumbnails = {"None": base}
    for name in PRESETS:
        preview = PhotoProcessor.from_image(base)
        apply_preset(preview, name)
        thumbnails[name] = preview.image
    encoded = {}
    for name, thumb in thumbnails.items():
        buffer = io.BytesIO()
        thumb.save(buffer, format="PNG")
        encoded[name] = buffer.getvalue()
    return encoded


@st.cache_data(show_spinner=False, max_entries=_THUMBNAIL_CACHE_ENTRIES)
def _custom_preset_thumbnail(digest: str, _source: Image.Image, recipe: dict) -> bytes:
    """A live preview render for one saved custom preset -- the "Load a Preset" dropdown's and the
    Batch Processor's counterpart to `_preset_thumbnails()`'s built-in-look row above. Unlike a
    built-in look (a handful of fields applied via `apply_preset()`), a custom preset is a full
    recipe that can touch anything `RECIPE_WIDGET_KEYS` covers, so only the real pipeline
    (`recipe_to_options()` + `app.backend.process_image()`, called in-process on a small thumbnail
    rather than over HTTP) reproduces it faithfully -- the same "deliberately separate, decorative-
    only path" precedent `_preset_thumbnails()` already established; the real edit still always
    goes through `app/backend.py` regardless of what's previewed here. Deliberately shows only one
    preset at a time (the currently-selected one), not a whole row like the 4 built-in looks --
    there can be many custom presets, and each one here costs a real (if thumbnail-sized) pipeline
    run, unlike a built-in look's cheap `apply_preset()` call.

    Falls back to the plain, unprocessed thumbnail (rather than raising) if the recipe fails to
    apply -- the same exception tuple `app/backend.py`'s own `/process` route catches, so a
    preview glitch never blocks picking or running the preset itself, just shows an unstyled
    thumbnail instead of a styled one.
    """
    base = _source.copy()
    base.thumbnail((96, 96), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    base.save(buffer, format="PNG")
    try:
        preview = process_image(FileLike("thumbnail.png", buffer.getvalue()), recipe_to_options(sanitize_recipe(recipe)))
    except (KeyError, ValueError, OSError, RuntimeError, cv2.error):
        preview = base
    encoded = io.BytesIO()
    preview.convert("RGB").save(encoded, format="PNG")
    return encoded.getvalue()


# Keyboard shortcuts for the Library Catalog's "Rate & Flag" panel: 1-5 rate, 0 clears, X toggles
# reject, arrows move Prev/Next. Reaches into window.parent.document (the real Streamlit page --
# this component's iframe is same-origin via `srcdoc`, confirmed live) to `.click()` the real
# buttons inside `st.container(key="lib-rate-controls")`, by fixed position (see the Python code
# that builds that container -- the two must stay in the same order). Skips the parent's normal
# typing targets (inputs/textareas) so it doesn't hijack the Search or Tags fields, and only acts
# when the panel is actually visible (`offsetParent !== null`) so it doesn't fire from a hidden
# tab -- every tab's Python code runs on every rerun in this app, visible or not. A static string,
# not a per-render function, since it has no values to interpolate.
_LIBRARY_KEYBOARD_SHORTCUTS_HTML = """
<script>
  (function() {
    const KEY_TO_BUTTON_INDEX = {
      'ArrowLeft': 0, 'ArrowRight': 1,
      '1': 2, '2': 3, '3': 4, '4': 5, '5': 6,
      '0': 7, 'x': 8, 'X': 8,
    };
    function handler(e) {
      const active = window.parent.document.activeElement;
      const tag = active && active.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      const idx = KEY_TO_BUTTON_INDEX[e.key];
      if (idx === undefined) return;
      const container = window.parent.document.querySelector('.st-key-lib-rate-controls');
      if (!container || container.offsetParent === null) return;
      const buttons = container.querySelectorAll('button');
      const target = buttons[idx];
      if (!target) return;
      target.click();
      e.preventDefault();
    }
    if (!window.parent._photoStudioLibraryShortcutsBound) {
      window.parent.document.addEventListener('keydown', handler);
      window.parent._photoStudioLibraryShortcutsBound = true;
    }
  })();
</script>
"""

# Keys belonging to the pipeline's Lens/Tone/Color/Detail stage (the "through Detail" half
# `_cached_process_through_detail` caches) -- every other key the `options = {...}` dict below
# builds belongs to the Repair-onward stage (`_cached_process_from_repair_onward`) instead. Derived
# from app.edit_schema.OPTION_SPECS' own per-option "stage", the same table the backend validates
# against, rather than a hand-kept copy here. "preview_scale" is a request option both stages need.
_DETAIL_STAGE_OPTION_KEYS = DETAIL_STAGE_OPTION_KEYS
_BOTH_STAGES_OPTION_KEYS = ("preview_scale",)

# Interactive previews of a photo larger than PROXY_TRIGGER_LONG_EDGE are rendered from a proxy
# whose long edge is PROXY_LONG_EDGE (P1 in the improvement report): the canvas never shows more
# than 1200x900, and a typical Stage 1 at 45 MP took ~44 s -- over the client's 30 s timeout, so
# the UI showed "Could not reach the backend". Full resolution is rendered only for export and for
# the zoomed (>100%) canvas views, where real pixels matter.
PROXY_LONG_EDGE = 2048
PROXY_TRIGGER_LONG_EDGE = 2560


def _png_bytes(image: Image.Image, **extra) -> bytes:
    buffer = io.BytesIO()
    # compress_level=1: every PNG built here is an internal hand-off, not a delivered file.
    image.save(buffer, format="PNG", compress_level=1, **extra)
    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=2)
def _proxy_source_png(digest: str, _source: Image.Image, long_edge: int) -> bytes:
    """The reduced-size stand-in the backend edits for the interactive preview (see
    PROXY_LONG_EDGE). Keeps the source's ICC profile (so the pipeline's sRGB conversion still
    happens) and EXIF (so the preview result carries the same metadata as the export would)."""
    proxy = _source.copy()
    proxy.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    extra = {}
    if _source.info.get("icc_profile"):
        extra["icc_profile"] = _source.info["icc_profile"]
    exif = _source.getexif()
    if len(exif):
        extra["exif"] = exif.tobytes()
    return _png_bytes(proxy, **extra)


@st.cache_data(show_spinner="🎨 Applying edits (tone/color/detail)…", max_entries=_FULL_RES_CACHE_ENTRIES, ttl=_FULL_RES_CACHE_TTL)
def _cached_process_through_detail(source_key: str, _image_bytes: bytes, filename: str, detail_options: dict) -> tuple[bytes, str | None]:
    """Stage 1 of 2 of the edit pipeline (Lens -> Tone -> Color -> Detail -- see app.pipeline's
    module docstring for the full order and the `stop_after`/`resume_from_detail` flags this
    two-stage split is built on). Cached independently of every Repair/Layers/Masks/Portrait/
    Crop/Effects option, so a Mask slider doesn't force the expensive Non-Local Means denoise to
    rerun. Keyed on `source_key` (the photo's digest, plus which proxy size) rather than the image
    bytes themselves. Returns (PNG bytes, the backend's handle for this result or None). The PNG carries ICC
    and EXIF, so metadata survives the hop into Stage 2, and is the backend's own response, cached
    as-is rather than decoded and re-encoded (P3). Stage 2 sends the handle instead of re-uploading
    the PNG and only falls back to the bytes when the backend no longer holds that result."""
    options = dict(detail_options)
    options["stop_after"] = "detail"
    result = process_via_backend(_image_bytes, filename, options)
    png = result.info.get("response_png")
    if png is None:
        extra = {key: result.info[key] for key in ("icc_profile", "exif") if result.info.get(key)}
        png = _png_bytes(result, **extra)
    return png, result.info.get("stage_handle")


@st.cache_data(show_spinner="🎨 Applying edits…", max_entries=_FULL_RES_CACHE_ENTRIES, ttl=_FULL_RES_CACHE_TTL)
def _cached_process_from_repair_onward(
    stage_a_key: str, _stage_a_png_bytes: bytes, remaining_options: dict,
    local_mask_key: str | None, _local_mask_bytes: bytes | None,
    layers_key: str | None, _layers_bytes: bytes | None,
    removal_mask_key: str | None, _removal_mask_bytes: bytes | None,
    _stage_handle: str | None = None,
    local_masks_key: str | None = None, _local_masks_bytes: bytes | None = None,
) -> Image.Image:
    """Stage 2 of 2 -- Repair through Effects, resuming from Stage 1's PNG via
    `options["resume_from_detail"]`. Cached independently of every Lens/Tone/Color/Detail option.
    Each large input is passed twice -- a short key that's hashed (Stage 1's own cache key, or a
    digest of the mask PNG / layer stack) and the underscore-prefixed bytes that aren't -- so a
    cache lookup no longer hashes a ~76 MB PNG plus full-resolution masks on every rerun. The
    spinner is the user's feedback that an edit is being processed on app/backend.py."""
    options = dict(remaining_options)
    options["resume_from_detail"] = True
    extra = {}
    if _stage_handle:
        extra["stage_handle"] = _stage_handle
    if _local_masks_bytes:
        extra["local_masks_bytes"] = _local_masks_bytes
    return process_via_backend(_stage_a_png_bytes, "stage_a.png", options, _local_mask_bytes, _layers_bytes, _removal_mask_bytes, **extra)


@st.cache_resource(show_spinner=False, max_entries=2)
def _lens_corrected_source(digest: str, _source: Image.Image, geometry_json: str) -> Image.Image:
    """The base image the Crop preview and the Masks/Repair on-canvas selection tools paint
    against: the source after the pipeline's Geometry step (rotate/flip, Lens corrections,
    straighten, constrain crop -- app.geometry.apply_geometry, the same function the backend runs).
    A crop box or mask drawn on the raw, uncorrected source would land on the wrong pixels once the
    backend actually warps or rotates the image. A no-op (the shared decoded source itself) when
    nothing changes the geometry. st.cache_resource: returned by reference, so read-only, like
    _decoded_source()."""
    geometry = json.loads(geometry_json)
    if not any(value for key, value in geometry.items() if key not in ("lens_vignette_midpoint", "constrain_crop")):
        return _source
    return apply_geometry(_source, geometry)


def _numeric_crop_changed() -> None:
    """on_change of the Free-form crop's Left/Top/Width/Height fields: turn the pixel values into
    the fractional box the crop editor and the recipe store (crop-free-*). A callback runs before
    the script, so writing those (non-widget) keys here is allowed."""
    width, height = st.session_state.get("_crop_frame_size", (1, 1))
    left = min(max(0, int(st.session_state.get("crop-num-left", 0))), width - 1)
    top = min(max(0, int(st.session_state.get("crop-num-top", 0))), height - 1)
    box_w = min(max(1, int(st.session_state.get("crop-num-width", width))), width - left)
    box_h = min(max(1, int(st.session_state.get("crop-num-height", height))), height - top)
    st.session_state["crop-free-x0"], st.session_state["crop-free-y0"] = left / width, top / height
    st.session_state["crop-free-x1"], st.session_state["crop-free-y1"] = (left + box_w) / width, (top + box_h) / height


@st.cache_data(show_spinner=False, max_entries=_FULL_RES_CACHE_ENTRIES, ttl=_FULL_RES_CACHE_TTL)
def _cached_gamut_warning_overlay(digest: str, _source: Image.Image, icc_bytes: bytes):
    """gamut_warning_overlay() runs two full-image littlecms conversions -- cached so it only
    recomputes when the photo or its embedded profile actually changes, not on every Streamlit
    rerun triggered by an unrelated control while the "Show out-of-gamut pixels" checkbox stays
    on."""
    return gamut_warning_overlay(_source, icc_bytes)


@st.cache_data(show_spinner=False, max_entries=24)
def _selection_mask(source_key: str, _mask_source: Image.Image, kind: str, params_json: str, soften: int) -> Image.Image:
    """One selection's full-resolution mask (before invert/opacity, which MaskStack applies),
    cached per (photo + lens settings, kind, params, soften) -- rebuilding every selection at full
    resolution on every rerun used to cost ~1.2 s per selection at 24 MP, even for a checkbox
    toggle elsewhere in the app."""
    return _build_selection_mask(_mask_source, kind, json.loads(params_json), soften)


def _selections_signature(layers: list[dict]) -> str:
    """Everything about a selection list that changes the combined mask, as one hashable string."""
    return json.dumps(
        [[layer["kind"], layer["params"], layer["soften"], layer["enabled"], layer["invert"], layer.get("opacity", 1.0), layer.get("combine", "add")] for layer in layers],
        sort_keys=True,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _combined_mask_png(source_key: str, _mask_source: Image.Image, signature: str, size: tuple[int, int] | None = None) -> bytes:
    """The union of every enabled selection in `signature` (see _selections_signature), PNG-encoded
    for the backend -- at full resolution, or resized to `size` for a proxy preview. Cached, so the
    union + PNG encode (~0.5 s per full-resolution selection) only reruns when the selections or
    the photo change."""
    return _combine_selections(
        _mask_source, signature, size,
        lambda kind, params, soften: _selection_mask(source_key, _mask_source, kind, json.dumps(params, sort_keys=True), soften),
    )


def _combine_selections(mask_source: Image.Image, signature: str, size, build) -> bytes:
    mask_stack = MaskStack(mask_source.size)
    for kind, params, soften, enabled, invert, opacity, combine in json.loads(signature):
        mask_stack.add(MaskLayer(kind, build(kind, params, soften), enabled, invert, opacity, combine))
    combined = mask_stack.combined()
    if size and combined.size != tuple(size):
        combined = combined.resize(tuple(size), Image.Resampling.BILINEAR)
    return _png_bytes(combined)


def _build_selection_mask(mask_source: Image.Image, kind: str, params: dict, soften: int) -> Image.Image:
    base_mask = _build_mask_image(mask_source, kind, params)
    return soften_mask(base_mask, soften) if soften else base_mask


def _combined_mask_png_uncached(mask_source: Image.Image, signature: str) -> bytes:
    """_combined_mask_png() without st.cache_data, at full resolution -- for the export callable,
    which Streamlit runs on a thread where st.* caching isn't available."""
    return _combine_selections(mask_source, signature, None, lambda kind, params, soften: _build_selection_mask(mask_source, kind, params, soften))


def _build_layers_payload(signature: str, canvas_size: tuple[int, int], scale: float = 1.0) -> bytes:
    """Uncached _layers_payload(), for the export callable's thread."""
    layers = json.loads(signature)
    workspace_layers = [
        WorkspaceLayer(layer["name"], _layer_dict_to_image(layer, tuple(canvas_size), scale), layer["opacity"], layer["blend_mode"], layer["visible"])
        for layer in layers
    ]
    return layers_to_json(workspace_layers).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=4)
def _layers_payload(signature: str, canvas_size: tuple[int, int], scale: float = 1.0) -> bytes:
    """app.layer_workspace.layers_to_json()'s wire payload for the Layers tab's stack, rendered at
    the full canvas size. Cached on the stack's own JSON, so rasterizing every layer (Layer Styles
    cost ~1 s per styled layer, a photo-sized image layer ~3 s) and base64-encoding it only happens
    when a layer actually changes."""
    return _build_layers_payload(signature, canvas_size, scale)


@st.cache_data(show_spinner=False, max_entries=4)
def _face_count(digest: str, _source: Image.Image) -> int:
    """Face detection for the Portrait tab's caption -- once per photo (app.portrait_tools runs it
    on a 1024 px proxy), not on every rerun."""
    return len(detect_faces(_source))


def _encode_rendered_export(image: Image.Image, settings: dict, exif: bytes | None) -> bytes:
    """Save-button producer when the canvas already holds the full-resolution result."""
    return encode_export(image, settings, exif=exif)[0]


def _own_adjustment_layers(layers: list[dict]) -> list[dict]:
    """Enabled selections that carry their own adjustment set (the Masks tab's "Own adjustments")."""
    return [layer for layer in layers if layer.get("adjust") and layer.get("enabled", True)]


def _local_masks_payload(masks: list[bytes]) -> bytes:
    """/process's "local_masks" file: one PNG per local_adjustments entry, in order."""
    return json.dumps({"masks": [base64.b64encode(mask).decode("ascii") for mask in masks]}).encode("utf-8")


def _render_full_resolution_export(
    content: bytes, filename: str, options: dict, settings: dict, mask_source: Image.Image,
    local_signature: str | None, removal_signature: str | None, layers_signature: str | None,
    own_signatures: list[str] | None = None, layers_canvas: tuple[int, int] | None = None,
) -> bytes:
    """Save-button producer for a proxy-previewed photo or an AI Upscale export: one full-
    resolution backend render that also encodes the delivery file (see export_via_backend).
    Streamlit runs a download_button's data callable on a separate thread when it's clicked --
    no st.* calls work there, and every input is bound up front (functools.partial) rather than
    read from script globals a later rerun may have rebound -- so the full-resolution masks are
    built with the plain, uncached helper."""
    extra = {}
    if own_signatures:
        extra["local_masks_bytes"] = _local_masks_payload([_combined_mask_png_uncached(mask_source, signature) for signature in own_signatures])
    return export_via_backend(
        content, filename, options, settings,
        _combined_mask_png_uncached(mask_source, local_signature) if local_signature else None,
        _build_layers_payload(layers_signature, layers_canvas) if layers_signature else None,
        _combined_mask_png_uncached(mask_source, removal_signature) if removal_signature else None,
        **extra,
    )


def _project_archive(source_name: str, source_bytes: bytes, recipe: dict, layers: list[dict]) -> bytes:
    """The .photoproject bytes for Save Project -- built only when the download button is clicked
    (Streamlit runs a download_button's data callable on its own thread; inputs are bound up front
    with functools.partial). Layers are saved parametrically (app.project_bundle.ProjectBundle's
    layer_stack): a Text/Shape/Path layer reopens editable, and each Image layer's pixels go into
    their own PNG entry instead of base64 inside JSON."""
    stack: list[dict] = []
    images: dict[str, bytes] = {}
    for layer in layers:
        entry = {key: value for key, value in layer.items() if key != "image_b64"}
        if layer.get("kind", "image") == "image":
            name = f"{uuid.UUID(hex=str(layer.get('id') or uuid.uuid4().hex)).hex}.png"
            images[name] = base64.b64decode(layer["image_b64"])
            entry["image_file"] = name
        stack.append(entry)
    return ProjectBundle(
        source_name=source_name, source_bytes=source_bytes, recipe=recipe,
        layer_stack=stack if stack else None, layer_images=images or None,
    ).to_bytes()


def _estimated_project_bytes(source_bytes: bytes, layers: list[dict]) -> int:
    """Roughly what _project_archive() will write (PNG entries are already compressed, so the
    archive is about the sum of its parts) -- checked before offering the download, since an error
    raised inside the download callable can't be shown to the user."""
    return len(source_bytes) + sum(len(layer.get("image_b64", "")) * 3 // 4 for layer in layers) + 64 * 1024


def _layers_from_project(project: ProjectBundle) -> list[dict]:
    """The Layers tab's layer dicts for a loaded project: parametric (editable) for a current
    project, flattened Image layers for one saved before layers were stored parametrically."""
    restored: list[dict] = []
    if project.layer_stack is not None:
        for layer in project.layer_stack:
            entry = {key: value for key, value in layer.items() if key != "image_file"}
            entry["id"] = uuid.uuid4().hex
            entry.setdefault("kind", "image")
            entry.setdefault("name", f"Layer {len(restored) + 1}")
            entry.setdefault("opacity", 1.0)
            entry.setdefault("blend_mode", "normal")
            entry.setdefault("visible", True)
            if entry["blend_mode"] not in BLEND_MODE_LABELS:
                entry["blend_mode"] = "normal"
            if entry["kind"] == "image":
                content = (project.layer_images or {}).get(str(layer.get("image_file")))
                if content is None:
                    continue
                Image.open(io.BytesIO(content)).verify()  # an unreadable PNG fails the load, cleanly
                entry["image_b64"] = base64.b64encode(content).decode("ascii")
            restored.append(entry)
        return restored
    workspace = project.layer_workspace_object()
    if workspace:
        # workspace.layers[0] is the LayerWorkspace-internal "Background" (the source image itself,
        # already restored separately) -- everything from index 1 on is a real overlay layer.
        for overlay_layer in workspace.layers[1:]:
            overlay_buffer = io.BytesIO()
            overlay_layer.image.save(overlay_buffer, format="PNG")
            restored.append(
                {
                    "id": uuid.uuid4().hex,
                    "kind": "image",  # older projects stored flattened pixels, even for a Text layer
                    "name": overlay_layer.name,
                    "image_b64": base64.b64encode(overlay_buffer.getvalue()).decode("ascii"),
                    "opacity": overlay_layer.opacity,
                    "blend_mode": overlay_layer.blend_mode,
                    "visible": overlay_layer.visible,
                }
            )
    return restored


def _open_project_bytes(content: bytes) -> None:
    """Load a .photoproject into this session (the caller reruns). Raises ValueError/KeyError/
    OSError for an unreadable file -- never a raw traceback on the page."""
    loaded_project = ProjectBundle.load(content)
    # Merged over RECIPE_DEFAULTS so any control the saved recipe doesn't mention (a project saved
    # before that control existed, or a hand-trimmed one) is reset to its default, rather than
    # silently keeping whatever the previous photo's edit had there.
    pending_recipe = {
        **RECIPE_DEFAULTS, "mask_layers": [], "removal_mask_layers": [], "curve_extra_points": [],
        **migrate_legacy_keys(loaded_project.recipe),  # before the merge, so old keys aren't shadowed by new defaults
    }
    pending_recipe["layers"] = _layers_from_project(loaded_project)
    st.session_state["restored_source_bytes"] = loaded_project.source_bytes
    st.session_state["restored_source_name"] = loaded_project.source_name
    # Empties the main "Open Image" uploader on the next run (see its key) so the project's bundled
    # source, not a previously opened photo, is what gets edited.
    st.session_state["_main_upload_generation"] = st.session_state.get("_main_upload_generation", 0) + 1
    st.session_state["_pending_restore"] = pending_recipe


def _open_library_photo(path: Path) -> None:
    """Open a catalogued photo in the editor (F8): the same restored-source path a reopened project
    uses, emptying the main uploader so it doesn't take priority. The adjustments are kept, so a
    look can be carried from one photo of a shoot to the next."""
    st.session_state["restored_source_bytes"] = path.read_bytes()
    st.session_state["restored_source_name"] = path.name
    st.session_state["_main_upload_generation"] = st.session_state.get("_main_upload_generation", 0) + 1


def _open_launch_file() -> None:
    """A file the desktop launcher was started with (double-clicking a .photoproject, or "Open
    with" on a photo -- see app.desktop.OPEN_PATH_ENV), opened once per session."""
    path_text = os.environ.get("PHOTO_STUDIO_OPEN_PATH")
    if not path_text or st.session_state.get("_launch_file_opened"):
        return
    st.session_state["_launch_file_opened"] = True
    path = Path(path_text)
    try:
        if path.suffix.lower() in (".photoproject", ".zip"):
            _open_project_bytes(path.read_bytes())
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
            st.session_state["restored_source_bytes"] = path.read_bytes()
            st.session_state["restored_source_name"] = path.name
        else:
            return
    except (ValueError, KeyError, OSError) as error:
        st.session_state["_launch_file_error"] = f"Could not open {path.name}: {error}"
        return
    st.rerun()


# Recipe keys each per-section "↺" button resets to its default (F2 in the improvement report --
# Streamlit sliders have no double-click reset).
RESET_SECTIONS = {
    "tone": ("exposure_ev", "contrast_amount", "highlights", "shadows", "preset"),
    "lens": ("lens_profile", "lens_distortion", "lens_vignette_amount", "lens_vignette_midpoint", "lens_chromatic_aberration", "lens_perspective_v", "lens_perspective_h"),
    "geometry": ("rotate", "flip_horizontal", "flip_vertical", "straighten", "constrain_crop"),
    "color": ("temperature", "tint", "saturation_amount", "vibrance"),
    "detail": ("texture", "clarity", "dehaze", "denoise", "denoise_color", "sharpen"),
    "effects": ("effects_vignette_amount", "effects_vignette_midpoint", "effects_grain_amount"),
    "mask_targets": ("mask_target_exposure_ev", "mask_target_contrast_amount", "mask_target_saturation_amount"),
}


def _section_reset_button(section: str, label: str) -> None:
    """A small "↺ Reset <label>" button that puts one section's controls back to their defaults,
    through the same _pending_restore path as Undo/Reset."""
    if st.button(f"↺ Reset {label}", key=f"reset-{section}-btn", help=f"Reset the {label} controls to their defaults"):
        st.session_state["_pending_restore"] = {key: RECIPE_DEFAULTS[key] for key in RESET_SECTIONS[section]}
        _request_rerun()


def _apply_export_preset(preset: dict) -> None:
    """on_click of the Export panel's "Apply": writes a saved export preset into the export widgets
    (a callback runs before the script, so it may set widget values)."""
    widget_keys = {
        "format": "exp-fmt", "quality": "exp-qual", "long_edge": "exp-long-edge", "sharpen": "exp-sharpen",
        "metadata": "exp-metadata", "remove_gps": "exp-remove-gps", "name_template": "exp-name-template",
    }
    for field, widget_key in widget_keys.items():
        if field in preset:
            st.session_state[widget_key] = preset[field]


def _toggle_one_to_one() -> None:
    """Shortcut Z: switch between Fit and a true 1:1 view (a callback runs before the script, so it
    may set the zoom widget's value)."""
    st.session_state["cv-zoom"] = 0.0 if float(st.session_state.get("cv-zoom", 0.0)) >= 1.0 else 1.0


def _toggle_original() -> None:
    """Shortcut B: before/after."""
    st.session_state["cv-orig"] = not st.session_state.get("cv-orig", False)


def _toggle_grid() -> None:
    """Shortcut G: grid on/off (remembering the last grid chosen)."""
    current = st.session_state.get("grid-type", "None")
    if current != "None":
        st.session_state["_last-grid-type"] = current
        st.session_state["grid-type"] = "None"
    else:
        st.session_state["grid-type"] = st.session_state.get("_last-grid-type", "Rule of Thirds")


def _request_rerun() -> None:
    """Use instead of calling st.rerun() directly from ANY widget callback below this point in the
    script -- a tab body (Repair/Layers/Masks/Portrait/Presets/Batch & Library), or canvas_col
    (Undo/Redo/Reset, zoom, split view, pan, export). Streamlit only keeps a keyed widget's
    session_state value if that widget was actually instantiated during the run immediately before
    it; the whole script (every tab, then canvas_col) runs top-to-bottom in one pass regardless of
    which tab is visible, so an immediate st.rerun() from anywhere before the last widget in that
    sequence aborts the script right there -- none of the not-yet-reached widgets get instantiated
    this pass, so Streamlit garbage-collects their session_state and silently resets them to their
    `value=` defaults on the very next run. (Same underlying mechanism as the HSL Color Mixer's
    dropdown-hidden-band bug documented in docs/PROJECT_STATUS.md, just triggered by an early
    rerun instead of conditional rendering -- a real instance of exactly this was found in review:
    the consume-check below used to sit before canvas_col rendered, so any tab's _request_rerun()
    call was already resetting canvas_col's zoom/pan/split-view/export settings, and Library
    Catalog's and canvas_col's own Undo/Redo/Reset buttons called st.rerun() directly and had the
    same effect.) Setting this flag and rerunning only after the very last widget in the script has
    rendered (see the check at the bottom of this file) means nothing gets GC'd."""
    st.session_state["_rerun_pending"] = True


def _consume_new_commit(pending_value: str, committed_key: str) -> bool:
    """True exactly once per genuinely new value written into an on-canvas editor's hidden
    `st.text_area` (see any `_*_editor_html()`/`_wand_picker_html()`'s `commit()` JS, and
    `_paint_editor_html`'s docstring for why a hidden text_area is used instead of touching a real
    widget directly). Marks `committed_key` immediately, before the caller even looks at
    `pending_value`'s content -- so a second rerun carrying the exact same string (e.g. triggered
    by an unrelated control elsewhere on the page, or a value that turns out to fail parsing) is
    correctly a no-op rather than retried forever or reapplied as a duplicate.

    Shared guard behind every on-canvas editor's commit handler (Magic Wand in both Masks and
    Repair, the White Balance picker, adding a Painted selection in both Masks and Repair,
    re-editing an existing Painted selection, the tone curve editor, and the free-form crop
    editor) -- previously duplicated inline at each of those ~8 call sites with the exact same
    three-line shape (`if pending and pending != st.session_state.get(key): st.session_state[key]
    = pending; ...`), flagged in "Known Issues" as one of three independent, inline workarounds
    for Streamlit's session_state-vs-widget-instantiation rules with no shared helper between
    them -- this closes that gap for the two of the three (`_request_rerun` and the hidden-
    textarea commit pattern) that share an actual "defer/guard a session_state write" shape. The
    HSL Color Mixer's "always render every band" fix is a different strategy entirely (an
    unconditional-rendering choice, not a guarded write) and isn't a natural fit for this helper;
    unifying it would mean redesigning how that panel renders, which "Known Issues" explicitly
    called out as not worth rushing.
    """
    if not pending_value or pending_value == st.session_state.get(committed_key):
        return False
    st.session_state[committed_key] = pending_value
    return True


st.set_page_config(page_title="Photo Studio Pro", page_icon="🎨", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* Hide Streamlit default header/footer to prevent cutting off the top */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Clean padding and tight vertical layout */
.block-container {
    max-width: 1800px;
    padding: 1.2rem 1.5rem 1.5rem !important;
}
[data-testid="stVerticalBlock"] { gap: 0.35rem; }

/* Top Header Bar */
.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid rgba(255, 255, 255, 0.12);
    padding: 0.2rem 0 0.5rem 0;
    margin-bottom: 0.6rem;
}
.brand-title {
    font: 800 1.5rem/1.1 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.brand-badge {
    background: linear-gradient(135deg, #ea580c, #c2410c);
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.header-info {
    font-size: 0.8rem;
    font-weight: 600;
    color: #cbd5e1;
    background: rgba(255, 255, 255, 0.06);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

/* #94a3b8 matches the muted caption color already used elsewhere in this file for secondary text
   on a dark background (e.g. the crop editor's own "drag a corner to resize..." caption) -- this
   class previously hardcoded #475569 (a dark slate meant for text on a LIGHT background), which
   both the "Tool Workspace" header above the tabs and the canvas header's own filename/dimensions
   label (see .canvas-live-view below) used, and neither sits inside any light-background wrapper.
   #475569 was a pre-dark-mode leftover: the same class of bug already found and fixed once for
   .brand-title below (that fix pinned the theme and relit .brand-title's own color, but didn't
   audit the rest of the stylesheet for the same leftover-light-color issue -- this is that audit,
   done afterward rather than caught live, since this session has no browser tool to visually
   confirm contrast with). Deliberately not quoting the canvas header label's own exact text here --
   see the comment above .canvas-live-view for why that specific phrase breaks a test. */
.panel-header-title {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.35rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
/* The canvas header's filename/dimensions label (see the top_c1 st.markdown call below), unlike
   the static "Tool Workspace" header sharing .panel-header-title's flex layout above, carries a
   variable-length uploaded filename -- at the narrow width this label's own column gets
   (canvas_col's top_c1), even the label's own fixed text prefix could wrap onto its own line
   inside a flex item, let alone a long filename, splitting the label across two illegible lines.
   (Deliberately not quoting that prefix's exact wording here: tests/test_frontend_unified_app.py's
   canvas_dims() helper locates this label in the rendered app by searching st.markdown() output
   for that same text, and this comment is itself rendered through st.markdown() a few hundred
   lines above the real label -- spelling the phrase out here previously made that search match
   this comment block instead, before the real label ever rendered.) Overrides the shared class's
   flex layout with plain single-line block truncation instead -- the entire label (icon, prefix,
   filename, dimensions) is one truncatable unit, never wraps, and a `title=` attribute on the div
   surfaces the full text on hover regardless of how much got cut off. */
.canvas-live-view {
    display: block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.canvas-live-view span { text-transform: none; letter-spacing: normal; }

/* Image Canvas */
.canvas-container {
    background: #090d16;
    border-radius: 8px;
    padding: 0.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid #1e293b;
    margin-bottom: 0.35rem;
}
.canvas-container img {
    max-width: 100%;
    max-height: 60vh;
    object-fit: contain;
    border-radius: 4px;
}

/* Export Panel -- same pre-dark-mode leftover-light-color issue as .header-info/.panel-header-title
   above, fixed the same way (a subtle translucent-white panel instead of a near-white one). */
.export-panel {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 0.55rem 0.85rem;
}

/* Workspace tab styling -- wrap onto a second row instead of hiding tabs behind a scroll arrow;
   with 9 tabs in a narrow column, scrolling silently hid Crop/Presets/More from first-time users.
   Streamlit 1.6x renders tabs via React Aria (role="tablist"/"tab"), not the older BaseWeb
   data-baseweb attributes -- these selectors target the current markup (data-testid="stTab(s)"). */
[data-testid="stTab"] {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    padding: 0.35rem 0.6rem !important;
}
[data-testid="stTabs"] [role="tablist"] {
    flex-wrap: wrap !important;
    overflow-x: visible !important;
    row-gap: 0.2rem;
}

/* On a narrow browser window the two-column workspace stacks vertically; without this, the
   whole (long) tool panel renders above the canvas, pushing the actual photo out of view until
   the user scrolls past every slider. Flip the visual order so the canvas -- the thing being
   edited -- appears first, while leaving the underlying script order (and therefore which
   column's widgets execute first) untouched.

   The intent was to scope this to just the outer tools_col/canvas_col split via the
   "workspace-split" keyed container, but "A B" is a *descendant* selector -- it matched every
   st.columns() row nested anywhere inside canvas_col too (the top toolbar row, Pan sliders, and
   the Export Format/Quality/Save row), reordering and wrapping all of them at this width as a
   real bug found live (reported as "widgets one above the other" below the Export row). Fixed by
   matching only the one immediate child chain from the keyed container down to that single outer
   row -- confirmed live via the DOM that "st-key-workspace-split" (a stVerticalBlock) has exactly
   one child, a stLayoutWrapper, which directly contains the tools_col/canvas_col
   stHorizontalBlock -- so a nested row several containers deeper can no longer match. */
@media (max-width: 900px) {
    .st-key-workspace-split > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    .st-key-workspace-split > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-of-type(1) { order: 2; }
    .st-key-workspace-split > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-of-type(2) { order: 1; }
}

/* Holds the real <textarea> the on-canvas paint editor's "Add to selections" button writes a
   base64 mask PNG into (see _paint_editor_html / _render_selection_stack's "Painted" branch), the
   one the tone curve editor's drag-commit writes curve-point JSON into (see _curve_editor_html /
   the RGB Tone Curve expander), and the one every on-canvas click-to-sample picker
   (_wand_picker_html -- Magic Wand in Masks/Repair, and the Color tab's White Balance picker)
   writes its clicked fractional position into -- plumbing details, not something the user should
   ever see or type into directly. */
.st-key-curve-hidden, .st-key-crop-editor-hidden, .st-key-cv-nav-wand-hidden,
.st-key-m-wand-hidden, .st-key-rm-wand-hidden, .st-key-wb-wand-hidden,
[class*="-paint-hidden"], [class*="l-path-"][class*="-hidden"] {
    display: none !important;
}
/* [class*="-paint-hidden"] (substring match, not an exact ".st-key-..." pair) covers both the
   original two Painted-selection commit containers (m-paint-hidden/rm-paint-hidden) *and* the
   per-layer "re-edit an existing Painted selection" containers below -- those are keyed by the
   layer's own uuid (a fresh one per layer, unknown ahead of time), so a fixed list of exact class
   names can't enumerate them the way it can for the small, fixed set of other hidden containers
   above. [class*="l-path-"][class*="-hidden"] is the same idea for the Pen tool's own commit
   container (_render_pen_path_editor): "l-path-draft-hidden" for the "Add a layer" draft, or
   "l-path-<uuid>-hidden" per existing Path layer -- caught in a UI review after the fact, not live
   (this app currently has no browser-testing tool available), by noticing this rule's existing
   selectors have no entry that would match either one; the exact bug class the wand-picker fix
   above already exists to prevent. */
</style>
""", unsafe_allow_html=True)

# Top Header
st.markdown("""
<div class="top-header">
    <div class="brand-title">
        <span>🎨 Photo Studio</span>
        <span class="brand-badge">CPU Pro</span>
    </div>
    <div class="header-info">⚡ Professional Photo Workstation</div>
</div>
""", unsafe_allow_html=True)

_open_launch_file()
if st.session_state.get("_launch_file_error"):
    st.error(f"⚠️ {st.session_state.pop('_launch_file_error')}")

# Apply any pending Undo/Redo or Load-Project restoration. This MUST run before any tab widget
# below is created (see _restore_recipe_to_state's docstring) -- it only ever fires on the rerun
# immediately triggered by clicking Undo, Redo, or loading a project.
if "_pending_restore" in st.session_state:
    _restore_recipe_to_state(st.session_state.pop("_pending_restore"))

# File loading section
upload_col1, upload_col2 = st.columns([2.5, 1], gap="small")
with upload_col1:
    # Keyed by a generation counter rather than a fixed key: Streamlit doesn't allow a
    # file_uploader's value to be set (or cleared) through session_state at all, so the only way
    # to empty it is to render it under a new key. Load Project bumps the counter -- otherwise a
    # photo still sitting in this uploader kept taking priority over the project's own bundled
    # source (see upload_bytes below), silently applying the loaded recipe to the wrong photo.
    # Only ever bumped right before a st.rerun(), from a handler above every tab, so no other
    # widget's state is at risk from the key change.
    uploaded = st.file_uploader(
        "Open Image",
        # One list for every place that accepts a photo (app.raw_import.SUPPORTED_EXTENSIONS) --
        # this one used to omit Fuji/Panasonic/Olympus/Pentax/Samsung RAW (.raf/.rw2/.orf/.pef/.srw).
        type=uploader_types(),
        help="Open image (JPG, PNG, WEBP, TIFF, or Camera RAW). Original is never overwritten.",
        label_visibility="collapsed",
        key=f"main-upload-{st.session_state.get('_main_upload_generation', 0)}",
    )
with upload_col2:
    st.caption("📂 Open any JPG, PNG, WEBP, TIFF, or camera RAW file.")
    # Only offered while the workspace is still empty -- once a real photo (or a reopened
    # project) is loaded, clicking this would silently swap it out for the placeholder, which
    # would be surprising rather than helpful.
    if not uploaded and "restored_source_bytes" not in st.session_state:
        if st.button(
            "🖼️ Try a sample photo",
            key="sample-photo-btn",
            use_container_width=True,
            help="No photo of your own handy? Load a small bundled placeholder image so you can explore every tool right away.",
        ):
            st.session_state["restored_source_bytes"] = SAMPLE_PHOTO_PATH.read_bytes()
            st.session_state["restored_source_name"] = SAMPLE_PHOTO_NAME
            st.rerun()

with st.expander("📁 Open a saved project (.photoproject)"):
    project_upload = st.file_uploader("Project file", type=["photoproject", "zip"], key="proj-upload")
    if project_upload and st.button("Load Project", key="proj-load-btn"):
        # A hand-edited or corrupted .photoproject (bad ZIP, malformed JSON manifest, a manifest
        # pointing at a file that isn't actually in the archive, unreadable embedded PNGs, ...)
        # must not crash the whole app with a raw traceback.
        try:
            _open_project_bytes(project_upload.getvalue())
        except (ValueError, KeyError, OSError) as error:
            st.error(f"⚠️ Could not open this project file: {error}")
        else:
            st.rerun()

if not uploaded and "restored_source_bytes" not in st.session_state:
    st.info("👆 Upload an image above, load a saved project, or click \"Try a sample photo\" to open the editing workspace.")
    st.stop()

# A freshly uploaded file always takes priority; otherwise fall back to the image bundled inside a
# loaded .photoproject so re-opening a project doesn't also require re-uploading its source image.
upload_name = uploaded.name if uploaded else st.session_state["restored_source_name"]
upload_bytes = uploaded.getvalue() if uploaded else st.session_state["restored_source_bytes"]
# The backend refuses uploads over MAX_UPLOAD_BYTES (a full-resolution export always sends the
# original file), so say so up front instead of after every edit fails. The uploader itself is
# capped to the same size by server.maxUploadSize; this also covers a reopened project/launch file.
if len(upload_bytes) > MAX_UPLOAD_BYTES:
    st.error(f"⚠️ '{upload_name}' is {len(upload_bytes) / 1024**2:.0f} MB -- over this app's {MAX_UPLOAD_BYTES // 1024**2} MB limit per photo.")
    st.stop()

# Load image safely. PhotoProcessor.from_bytes() routes camera RAW files (.cr2/.cr3/.nef/.arw/.dng)
# through rawpy via app.raw_import, and falls back to a plain PIL open for every other format --
# using it here (instead of Image.open() directly) is what makes the RAW types advertised in the
# uploader above actually work instead of failing with an unreadable-file error.
#
# Decoded once per upload (see _decoded_source) and keyed by a content digest hashed once per
# upload (_source_digest) -- this used to be a full decode (a whole LibRaw demosaic for RAW) on
# every single rerun.
source_digest = _source_digest(getattr(uploaded, "file_id", None) if uploaded else ("restored", id(upload_bytes)), upload_bytes)
try:
    source = _decoded_source(source_digest, upload_bytes, upload_name)
except RuntimeError as error:
    st.error(f"⚠️ {error}")
    st.stop()
except OSError:
    # PIL raises OSError (UnidentifiedImageError, "image file is truncated", ...) for a corrupted,
    # incomplete, or unrecognizable non-RAW file -- this used to propagate as an unhandled
    # exception, showing the end user a raw Python traceback (including local file paths) instead
    # of a plain explanation. The technical message isn't worth surfacing (PIL's own text for an
    # unidentifiable file includes a raw memory address), so this doesn't interpolate `error`.
    st.error(f"⚠️ Couldn't open '{upload_name}' — the file may be corrupted, incomplete, or not a supported image format.")
    st.stop()
# The source's own EXIF (Orientation already applied and removed by the decode), attached to every
# export below via app.export -- the Save button used to write neither EXIF nor ICC.
source_exif = source.getexif().tobytes() if len(source.getexif()) else None

# The actual edit pipeline runs on app/backend.py, not in this process (see docs/PROJECT_STATUS.md);
# `source` above is only used locally for decoding, mask geometry, and previews.
try:
    ensure_backend_running()
except RuntimeError as error:
    st.error(f"⚠️ {error}")
    st.stop()
# Tags this run's backend requests with the session, so the backend can drop a superseded render
# when a newer one from the same session arrives ("latest wins", see app.backend).
st.session_state.setdefault("_edit_session", uuid.uuid4().hex)
bind_edit_session(st.session_state["_edit_session"])

# Main Two Columns: Level & Parallel Layout
with st.container(key="workspace-split"):
    tools_col, canvas_col = st.columns([1.1, 1.4], gap="large")

    with tools_col:
        st.markdown('<div class="panel-header-title">🛠️ Tool Workspace</div>', unsafe_allow_html=True)
        
        # Crop sits right after Tone in this visual order -- a workflow convenience (see most photo
        # editors: framing first, before fine-tuning tone/color) that's independent of the actual
        # processing order. app/backend.py::process_image() always applies the crop last, after
        # Masks/Portrait, regardless of where this tab sits in the UI -- every tab's code already
        # runs on every rerun no matter which one is visually selected (see the note below the
        # tabs), so reordering this tuple/list is purely cosmetic and doesn't change what gets sent
        # to the backend or when.
        # Repair sits right after Detail, matching where app/backend.py::process_image() applies
        # removal_mask in the actual pipeline (right after Detail, before Layers) -- unlike Crop,
        # there's no workflow reason to decouple this one's visual position from its pipeline
        # position, so they're kept in sync.
        # Lens sits right after Tone, before Crop -- fix distortion/vignette/chromatic aberration
        # before framing the crop against the now-corrected geometry. app/backend.py::process_image()
        # applies lens corrections first in the actual pipeline (before Tone even), same reasoning
        # as Crop above: its visual tab position is a workflow convenience, decoupled from where it
        # actually runs.
        # on_change="rerun" + key: Streamlit reports which tab is on screen (`.open`), so hidden tabs
        # skip their display-only work (see _TAB_VISIBLE). Switching tabs is a (cheap, cached) rerun.
        tab_tone, tab_lens, tab_crop, tab_color, tab_detail, tab_repair, tab_layers, tab_masks, tab_portrait, tab_presets, tab_more = st.tabs([
            "☼ Tone", "◎ Lens", "⛶ Crop", "🎨 Color", "🔬 Detail", "🩹 Repair", "◈ Layers", "◉ Masks", "◌ Portrait", "✦ Presets", "☰ Batch & Library"
        ], key="tool-tab", on_change="rerun")

        show_original = False
        compare_mode = False
        zoom_level = 1.0
        pan_x = 0.5
        pan_y = 0.5
        edited_image = source
        layers_signature = None
        aspect_code = "Original"
        crop_box = None

        # TAB 1: TONE & EXPOSURE
        _TAB_VISIBLE = _tab_open(tab_tone)
        with tab_tone:
            # Mirrors PhotoProcessor.auto_enhance() (app/image_processor.py) exactly -- that method
            # is otherwise only reachable from the CLI's --auto-enhance flag. Writing straight into
            # _pending_restore reuses the same "one rerun to apply" mechanism as Undo/Redo/Reset,
            # and only touches these five recipe fields, leaving everything else (crop, masks,
            # curve, temperature, ...) exactly as the user left it.
            # Image-adaptive (app.image_processor.auto_enhance_recipe, measured on a small proxy):
            # exposure, black/white points, white balance, vibrance and -- only for a noisy photo --
            # denoise. Applied through _pending_restore like Undo/Reset; everything else (crop,
            # masks, HSL, ...) stays as the user left it.
            if st.button("✨ Auto Enhance", key="auto-enhance-btn", help="Measure this photo and set exposure, black/white points, white balance and vibrance to suit it", use_container_width=True):
                auto_settings = auto_enhance_recipe(source)
                st.session_state["_pending_restore"] = auto_settings
                changed = [
                    f"exposure {auto_settings['exposure_ev']:+.2f} EV" if auto_settings["exposure_ev"] else None,
                    f"black/white points {auto_settings['curve_black']}/{auto_settings['curve_white']}" if auto_settings["curve_enabled"] else None,
                    f"temperature {auto_settings['temperature']:+d}, tint {auto_settings['tint']:+d}" if auto_settings["temperature"] or auto_settings["tint"] else None,
                    f"vibrance +{auto_settings['vibrance']}" if auto_settings["vibrance"] else None,
                    "denoise" if auto_settings["denoise"] else None,
                ]
                st.toast("Auto Enhance: " + (", ".join(item for item in changed if item) or "this photo already looks balanced"), icon="✨")
                _request_rerun()
            preset_names = ["None", *sorted(PRESETS)]
            if _TAB_VISIBLE:
                preset_thumbs = _preset_thumbnails(source_digest, source)
                thumb_cols = st.columns(len(preset_names))
                for thumb_col, name in zip(thumb_cols, preset_names):
                    with thumb_col:
                        st.image(preset_thumbs[name], use_container_width=True)
                        st.caption(name)
            preset_choice = st.selectbox("Preset Look", preset_names, key="tone-preset")
            exp_ev = st.slider("Exposure (EV)", -5.0, 5.0, 0.0, 0.05, key="t-ev", help="In stops of light: +1 doubles the light, -1 halves it. Bright highlights roll off smoothly instead of clipping.")
            cont_amount = st.slider("Contrast", -100, 100, 0, key="t-contrast", help="Spreads tones apart around midtone gray -- positive makes shadows darker and highlights brighter, negative flattens them.")
            high_val = st.slider("Highlights", -100, 100, 0, key="t-high", help="Recovers or boosts the brightest parts of the image -- negative pulls blown highlights back, positive brightens them further.")
            shad_val = st.slider("Shadows", -100, 100, 0, key="t-shad", help="Lifts or deepens the darkest parts of the image -- positive opens up shadow detail, negative deepens shadows.")
            _section_reset_button("tone", "Tone")

        # TAB: LENS CORRECTIONS
        _TAB_VISIBLE = _tab_open(tab_lens)
        with tab_lens:
            st.markdown("**Lens Corrections**")
            st.caption(
                "Not calibrated to a specific camera/lens model -- there's no profile database here, "
                "just typical starting points by lens category. Pick the closest one, then fine-tune "
                "with the sliders below."
            )
            lens_profile_choice = st.selectbox("Lens Profile", list(LENS_PROFILES), key="lens-profile")
            # No help= here: st.button(..., help=...) renders a second, separately-focusable
            # tooltip-trigger <button> in this Streamlit version (see the Library rate/flag panel's
            # star-button fix in docs/PROJECT_STATUS.md) -- harmless for a plain click target like
            # this one, but a caption is simpler and avoids the duplicate DOM node regardless.
            st.caption("Loads this profile's starting values -- you can still adjust the sliders afterward.")
            if st.button("Apply Profile Defaults", key="lens-apply-profile-btn"):
                profile_values = LENS_PROFILES[lens_profile_choice]
                st.session_state["_pending_restore"] = {
                    "lens_distortion": profile_values["distortion"],
                    "lens_vignette_amount": profile_values["vignette_amount"],
                    "lens_vignette_midpoint": profile_values["vignette_midpoint"],
                    "lens_chromatic_aberration": profile_values["chromatic_aberration"],
                }
                _request_rerun()
            lens_distortion = st.slider(
                "Distortion", -100, 100, 0, key="lens-distortion",
                help="Negative corrects barrel distortion (straight lines bowing outward); positive corrects pincushion (bowing inward).",
            )
            lens_vignette_amount = st.slider(
                "Vignette Amount", -100, 100, 0, key="lens-vignette-amt",
                help="Negative darkens the corners (adds a vignette); positive brightens them (removes one).",
            )
            lens_vignette_midpoint = st.slider(
                "Vignette Midpoint", 0, 100, 50, key="lens-vignette-mid",
                help="How far from center the vignette effect starts -- higher concentrates it closer to the corners.",
            )
            lens_chromatic_aberration = st.slider(
                "Chromatic Aberration", -100, 100, 0, key="lens-ca",
                help="Corrects red/cyan and blue/yellow fringing near high-contrast edges.",
            )
            st.markdown("**Perspective (Keystone)**")
            st.caption("Not part of a lens profile -- this depends on the shooting angle, not the lens, so it isn't filled in by 'Apply Profile Defaults' above.")
            lens_perspective_v = st.slider(
                "Vertical Keystone", -100, 100, 0, key="lens-persp-v",
                help="Corrects converging vertical lines (e.g. a building shot from below) -- positive narrows the top edge, negative narrows the bottom.",
            )
            lens_perspective_h = st.slider(
                "Horizontal Keystone", -100, 100, 0, key="lens-persp-h",
                help="Corrects converging horizontal lines (e.g. a wall shot at an angle) -- positive narrows the left edge, negative narrows the right.",
            )
            _section_reset_button("lens", "Lens")

        # TAB: CROP & ASPECT RATIO (+ rotate/flip/straighten) -- placed right after Tone for
        # workflow convenience (frame the shot before fine-tuning tone/color); the geometry runs
        # first in the actual pipeline and the crop last, see the note above the tabs() call.
        _TAB_VISIBLE = _tab_open(tab_crop)
        with tab_crop:
            st.markdown("**Rotate & Straighten**")
            rot_left_col, rot_right_col, rot_value_col = st.columns([1, 1, 1.2])
            current_rotation = int(st.session_state.get("geo-rotate", 0))
            if rot_left_col.button("⟲ Rotate left", key="geo-rot-left-btn", use_container_width=True, help="Rotate 90° counter-clockwise"):
                st.session_state["_pending_restore"] = {"rotate": (current_rotation - 90) % 360}
                _request_rerun()
            if rot_right_col.button("⟳ Rotate right", key="geo-rot-right-btn", use_container_width=True, help="Rotate 90° clockwise"):
                st.session_state["_pending_restore"] = {"rotate": (current_rotation + 90) % 360}
                _request_rerun()
            with rot_value_col:
                geo_rotate = st.selectbox("Rotation", ROTATIONS, key="geo-rotate", format_func=lambda angle: f"{angle}° clockwise", label_visibility="collapsed")
            flip_h_col, flip_v_col = st.columns(2)
            geo_flip_h = flip_h_col.checkbox("⇆ Flip horizontal", key="geo-flip-h")
            geo_flip_v = flip_v_col.checkbox("⇅ Flip vertical", key="geo-flip-v")
            straighten_col, auto_col = st.columns([3, 1])
            with straighten_col:
                geo_straighten = st.slider(
                    "Straighten (°)", -45.0, 45.0, 0.0, 0.05, key="geo-straighten",
                    help="Rotates the photo by a small angle to level a tilted horizon -- positive turns it clockwise.",
                )
            auto_level_requested = auto_col.button("Auto", key="geo-auto-btn", use_container_width=True, help="Detect the photo's dominant horizontal/vertical lines and level them")
            geo_constrain = st.checkbox(
                "Constrain crop", value=True, key="geo-constrain",
                help="Crop to the largest rectangle that Lens corrections, keystone and straighten leave fully covered -- no black or smeared corners.",
            )
            _section_reset_button("geometry", "Rotate & Straighten")

        geometry_options = {
            "rotate": geo_rotate, "flip_horizontal": geo_flip_h, "flip_vertical": geo_flip_v,
            "straighten": geo_straighten, "constrain_crop": geo_constrain,
            "lens_distortion": lens_distortion, "lens_vignette_amount": lens_vignette_amount,
            "lens_vignette_midpoint": lens_vignette_midpoint, "lens_chromatic_aberration": lens_chromatic_aberration,
            "lens_perspective_v": lens_perspective_v, "lens_perspective_h": lens_perspective_h,
        }
        geometry_json = json.dumps(geometry_options, sort_keys=True)
        # The Crop preview and the Masks/Repair selection tools below all paint against this, not
        # the raw `source` -- see _lens_corrected_source()'s docstring: the backend applies the
        # geometry before Crop/Masks/Repair, so those tools must agree with it on which pixel is where.
        mask_source = _lens_corrected_source(source_digest, source, geometry_json)
        # Cache key for everything built against mask_source (selection masks): the photo plus the
        # geometry settings that warped it.
        mask_source_key = f"{source_digest}|{geometry_json}"
        if auto_level_requested:
            # Measured on the frame before straightening (the angle replaces the current one).
            level_base = apply_geometry(preview_image(source, (1024, 1024)), {**geometry_options, "straighten": 0.0, "constrain_crop": False})
            detected_angle = estimate_straighten_angle(level_base)
            st.session_state["_pending_restore"] = {"straighten": detected_angle}
            st.toast(f"Straighten set to {detected_angle:+.2f}°" if detected_angle else "No dominant lines found to level", icon="📐")
            _request_rerun()

        _TAB_VISIBLE = _tab_open(tab_crop)
        with tab_crop:
            st.markdown("**Crop / Aspect Ratio**")
            crop_choice = st.selectbox("Aspect Ratio", ["Original", "Free-form", "Custom", *ASPECT_RATIOS], key="crop-aspect")
            crop_swap = st.checkbox("Swap orientation (portrait ↔ landscape)", key="crop-swap", disabled=crop_choice in ("Original", "Free-form"))
            if crop_choice == "Free-form":
                # Fractional (0..1) storage, not pixel coordinates: a fraction of the frame scales to
                # any image size, so the box persists through Undo/Redo and Save/Load Project via
                # RECIPE_WIDGET_KEYS/RECIPE_DEFAULTS like every other control.
                crop_box_fractional = (
                    st.session_state.get("crop-free-x0", 0.0), st.session_state.get("crop-free-y0", 0.0),
                    st.session_state.get("crop-free-x1", 1.0), st.session_state.get("crop-free-y1", 1.0),
                )
                # Read (not written) here -- the Guides & Grid panel lives in canvas_col, which
                # renders later in the script; session_state still carries last rerun's value. Drawn
                # onto the crop preview's background and fed to the editor as snap targets.
                crop_grid_type = st.session_state.get("grid-type", "None")
                crop_guides = st.session_state.get("guides", [])
                snap_x = list(grid_line_fractions(crop_grid_type)) + [g["position_pct"] / 100.0 for g in crop_guides if g.get("orientation") == "vertical"]
                snap_y = list(grid_line_fractions(crop_grid_type)) + [g["position_pct"] / 100.0 for g in crop_guides if g.get("orientation") == "horizontal"]
                crop_preview_bg = preview_image(mask_source, (480, 480))
                if crop_grid_type != "None" or crop_guides:
                    crop_preview_bg = draw_guides_and_grid(crop_preview_bg, crop_grid_type, crop_guides)
                _show_component(lambda: _crop_editor_html(
                    crop_preview_bg, "crop-editor", crop_box_fractional, mask_source.size, snap_x=snap_x, snap_y=snap_y,
                ))
                with st.container(key="crop-editor-hidden"):
                    pending_crop = st.text_area("Free-form crop box", key="crop-editor-data", label_visibility="collapsed")
                if _consume_new_commit(pending_crop, "crop-editor-committed"):
                    try:
                        parsed_box = json.loads(pending_crop)
                        fx0, fy0, fx1, fy1 = (float(parsed_box[k]) for k in ("x0", "y0", "x1", "y1"))
                    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                        pass
                    else:
                        st.session_state["crop-free-x0"], st.session_state["crop-free-y0"] = fx0, fy0
                        st.session_state["crop-free-x1"], st.session_state["crop-free-y1"] = fx1, fy1
                        crop_box_fractional = (fx0, fy0, fx1, fy1)
                        _request_rerun()
                src_w, src_h = mask_source.size
                fx0, fy0, fx1, fy1 = crop_box_fractional
                crop_box = (int(fx0 * src_w), int(fy0 * src_h), round(fx1 * src_w), round(fy1 * src_h))
                # Numeric entry (F4): the same box in pixels, editable as numbers -- a keyboard
                # alternative to dragging. Their values are set from the box before they render (so
                # they always show the editor's latest box); typing calls _numeric_crop_changed().
                st.session_state["_crop_frame_size"] = (src_w, src_h)
                st.session_state["crop-num-left"], st.session_state["crop-num-top"] = crop_box[0], crop_box[1]
                st.session_state["crop-num-width"] = max(1, crop_box[2] - crop_box[0])
                st.session_state["crop-num-height"] = max(1, crop_box[3] - crop_box[1])
                num_cols = st.columns(4)
                for num_col, (num_key, num_label, num_max) in zip(num_cols, (
                    ("crop-num-left", "Left (px)", src_w - 1), ("crop-num-top", "Top (px)", src_h - 1),
                    ("crop-num-width", "Width (px)", src_w), ("crop-num-height", "Height (px)", src_h),
                )):
                    num_col.number_input(num_label, min_value=0, max_value=max(1, num_max), step=1, key=num_key, on_change=_numeric_crop_changed)
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    crop_box = None
                    st.warning("Select a non-empty region.")
            elif crop_choice != "Original":
                if crop_choice == "Custom":
                    custom_w_col, custom_h_col = st.columns(2)
                    crop_w_ratio = int(custom_w_col.number_input("Ratio width", 1, 100, 3, key="crop-custom-w"))
                    crop_h_ratio = int(custom_h_col.number_input("Ratio height", 1, 100, 2, key="crop-custom-h"))
                else:
                    crop_w_ratio, crop_h_ratio = ASPECT_RATIOS[crop_choice]
                if crop_swap:
                    crop_w_ratio, crop_h_ratio = crop_h_ratio, crop_w_ratio
                st.caption(f"Center-cropping to {crop_w_ratio}:{crop_h_ratio}.")
                aspect_code = f"{crop_w_ratio}:{crop_h_ratio}"
            else:
                st.caption("No crop applied — showing the full frame.")

        # TAB 2: COLOR & CURVES
        _TAB_VISIBLE = _tab_open(tab_color)
        with tab_color:
            temp_val = st.slider("Temperature (Cool/Warm)", -100, 100, 0, key="c-temp", help="Shifts the overall white balance -- negative cools the image toward blue, positive warms it toward orange.")
            tint_val = st.slider("Tint (Green/Magenta)", -100, 100, 0, key="c-tint", help="Corrects a color cast along the green/magenta axis, independent of Temperature.")
            sat_amount = st.slider("Saturation", -100, 100, 0, key="c-saturation", help="Uniformly boosts or reduces color intensity across the whole image -- -100 is grayscale, 0 is unchanged.")
            vib_val = st.slider(
                "Vibrance", -100, 100, 0, key="c-vib",
                help="A smarter saturation boost -- protects skin tones and already-saturated colors, pushing washed-out ones harder instead.",
            )
            _section_reset_button("color", "Color")

            with st.expander("🎯 Pick White Balance", expanded=False):
                st.caption("Click a spot in the photo that should be neutral gray or white -- Temperature and Tint above are set to whatever would neutralize it. A starting point to fine-tune from, not a guaranteed perfect result.")
                _show_component(lambda: _wand_picker_html(preview_image(mask_source, (480, 480)), "wb"))
                with st.container(key="wb-wand-hidden"):
                    wb_click_raw = st.text_area("White balance click", key="wb-wand-data", label_visibility="collapsed")
                if _consume_new_commit(wb_click_raw, "wb-wand-committed"):
                    try:
                        wb_click = json.loads(wb_click_raw)
                        wb_x, wb_y = float(wb_click["x"]), float(wb_click["y"])
                    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                        pass
                    else:
                        new_temp, new_tint = estimate_white_balance(mask_source, wb_x, wb_y)
                        st.session_state["_pending_restore"] = {"temperature": new_temp, "tint": new_tint}
                        _request_rerun()

            st.markdown("**Color Mixer (HSL)**")
            st.caption("Adjust Hue, Saturation, and Luminance independently for each color band, like a professional editor's Color Mixer.")
            # Every band's sliders render unconditionally every rerun (not just a "currently
            # selected" one behind a picker) -- an earlier version of this used a selectbox to
            # show only one band's 3 sliders at a time, but Streamlit garbage-collects a widget's
            # session_state value when its key isn't instantiated in a run, so switching away from
            # a band silently reset it back to 0. Always rendering every widget sidesteps that
            # entirely, and matches the same "list of expandable items" pattern already used for
            # Mask Layers below -- collapsed by default, so this isn't much bulkier in practice.
            # (Nested inside an outer expander it would be: st.expander cannot nest inside
            # another st.expander, which is also why this whole section isn't wrapped in one.)
            hsl_bands = {}
            for band in HSL_BAND_ORDER:
                h_key, s_key, l_key = _hsl_widget_key(band, "h"), _hsl_widget_key(band, "s"), _hsl_widget_key(band, "l")
                is_active = any(st.session_state.get(k, 0) for k in (h_key, s_key, l_key))
                with st.expander(f"{'🟢' if is_active else '⚪'} {band}", expanded=False):
                    hue = st.slider("Hue", -100, 100, 0, key=h_key)
                    sat = st.slider("Saturation", -100, 100, 0, key=s_key)
                    lum = st.slider("Luminance", -100, 100, 0, key=l_key)
                hsl_bands[band] = {"hue": hue, "sat": sat, "lum": lum}

            with st.expander("Color Grading (Split Toning)", expanded=False):
                st.caption("Tint shadows, midtones, and highlights toward independent colors — the same three-range color grading every professional editor offers, separate from the single global Temperature/Tint above.")
                cg_s_col, cg_m_col, cg_h_col = st.columns(3)
                with cg_s_col:
                    st.markdown("**Shadows**")
                    cg_shadows_hue = st.slider("Hue", 0, 360, 0, key="cg-sh-hue", label_visibility="visible")
                    cg_shadows_intensity = st.slider("Intensity", 0, 100, 0, key="cg-sh-int")
                with cg_m_col:
                    st.markdown("**Midtones**")
                    cg_midtones_hue = st.slider("Hue", 0, 360, 0, key="cg-mid-hue")
                    cg_midtones_intensity = st.slider("Intensity", 0, 100, 0, key="cg-mid-int")
                with cg_h_col:
                    st.markdown("**Highlights**")
                    cg_highlights_hue = st.slider("Hue", 0, 360, 0, key="cg-hi-hue")
                    cg_highlights_intensity = st.slider("Intensity", 0, 100, 0, key="cg-hi-int")
                cg_balance = st.slider(
                    "Balance", -100, 100, 0, key="cg-balance",
                    help="Shifts where shadows end and highlights begin -- negative gives shadows more of the tonal range, positive gives highlights more.",
                )

            with st.expander("RGB Tone Curve", expanded=False):
                st.session_state.setdefault("curve_extra_points", [])
                curve_bounds = [(0, 64), (0, 128), (64, 192), (128, 255), (192, 255)]
                preview_points = [
                    (0, st.session_state.get("c-p0", 0)),
                    (64, st.session_state.get("c-p1", 64)),
                    (128, st.session_state.get("c-p2", 128)),
                    (192, st.session_state.get("c-p3", 192)),
                    (255, st.session_state.get("c-p4", 255)),
                ]
                _show_component(lambda: (_curve_editor_html(preview_points, curve_bounds, st.session_state["curve_extra_points"]), 270))
                with st.container(key="curve-hidden"):
                    curve_commit = st.text_area("Curve commit data", key="curve-commit-data", label_visibility="collapsed")
                # See _curve_editor_html's docstring: a script-dispatched event on a range slider
                # doesn't reliably commit in this Streamlit version, so the drag handler writes
                # here instead and this reuses the same "set _pending_restore, then rerun" path
                # Undo/Redo/Reset already use to update a slider from outside the widget itself.
                if _consume_new_commit(curve_commit, "curve-committed"):
                    try:
                        parsed = json.loads(curve_commit)
                    except (ValueError, TypeError):
                        parsed = None
                    if isinstance(parsed, dict):
                        st.session_state["_pending_restore"] = parsed
                        _request_rerun()
                with st.container(key="curve-points"):
                    use_curve = st.checkbox("Enable RGB Curve", value=False, key="c-curve-en")
                    c_black = st.slider("Blacks", 0, 64, 0, key="c-p0")
                    c_low = st.slider("Shadows", 0, 128, 64, key="c-p1")
                    c_mid = st.slider("Midtones", 64, 192, 128, key="c-p2")
                    c_high = st.slider("Highlights", 128, 255, 192, key="c-p3")
                    c_white = st.slider("Whites", 192, 255, 255, key="c-p4")
                st.caption("Drag a point on the curve, or use the sliders below for exact values.")

            with st.expander("🎯 Soft Proof: Gamut Warning", expanded=False):
                source_icc_bytes = source.info.get("icc_profile")
                if not source_icc_bytes:
                    st.caption("This photo has no embedded color profile to check against, so there's nothing to proof.")
                else:
                    st.caption(
                        "Highlights, in magenta, pixels that get clipped when this app's own sRGB pipeline converts "
                        "the photo's embedded color profile. Not a simulation of a specific printer/press profile -- "
                        "there's no profile database bundled -- just where this app's own color management can't "
                        "fully represent the source."
                    )
                    if st.checkbox("Show out-of-gamut pixels", key="gamut-warning-toggle"):
                        try:
                            st.image(
                                _cached_gamut_warning_overlay(source_digest, source, source_icc_bytes),
                                caption="Magenta = clipped when converted to this app's sRGB working space",
                                use_container_width=True,
                            )
                        except ICC_TRANSFORM_ERRORS:
                            st.caption("Could not read this photo's embedded color profile.")

        # TAB 3: DETAIL & SHARPNESS
        _TAB_VISIBLE = _tab_open(tab_detail)
        with tab_detail:
            text_val = st.slider("Texture", -100, 100, 0, key="d-text", help="Enhances or softens fine surface detail (skin pores, fabric, foliage) without affecting large tonal shapes.")
            clar_val = st.slider("Clarity", -100, 100, 0, key="d-clar", help="Boosts or reduces midtone contrast at a larger scale than Texture -- adds a sense of punch/depth; negative gives a soft, dreamy look.")
            haze_val = st.slider(
                "Dehaze", 0, 100, 0, key="d-haze",
                help="Reduces atmospheric haze (Dark Channel Prior, a classical algorithm -- not AI). Pushed too high, it can crush shadows or halo around strong edges.",
            )
            denoise_val = st.slider("Denoise", 0, 9, 0, step=2, key="d-den", help="Smooths luminance noise/grain -- higher removes more noise but can soften fine detail.")
            denoise_color_val = st.slider(
                "Color Noise", 0, 9, 0, step=2, key="d-den-color", disabled=denoise_val == 0,
                help="Independent chroma (color speckle) denoising strength, on top of Denoise's luminance/detail smoothing -- push higher than Denoise to clean up color blotches without softening detail as much. Has no effect while Denoise is 0.",
            )
            sharp_en = st.checkbox("Sharpen (Unsharp Mask)", value=False, key="d-sharp", help="Increases edge contrast to make the image look crisper -- applied last in the Detail group, after Texture/Clarity/Denoise.")
            _section_reset_button("detail", "Detail")

            # AI Upscale moved to the Export panel (below the canvas): as a live Detail control it
            # re-ran Real-ESRGAN on every Stage 2 change, and its 2x/4x result could exceed the
            # pixel limit this process decodes images under, crashing the page.
            st.caption("🔎 AI Upscale is an export option now -- see **Export settings** below the canvas.")

            with st.expander("✨ Effects", expanded=False):
                st.caption("Creative finishing touches applied to the final, already-cropped frame -- separate from the Lens tab's corrective vignette, which runs before Crop.")
                fx_v_col, fx_g_col = st.columns(2)
                with fx_v_col:
                    st.markdown("**Post-Crop Vignette**")
                    fx_vignette_amount = st.slider(
                        "Amount", -100, 100, 0, key="fx-vignette-amt",
                        help="Negative darkens the corners; positive brightens them.",
                    )
                    fx_vignette_midpoint = st.slider("Midpoint", 0, 100, 50, key="fx-vignette-mid")
                with fx_g_col:
                    st.markdown("**Film Grain**")
                    fx_grain_amount = st.slider("Amount", 0, 100, 0, key="fx-grain-amt")
                _section_reset_button("effects", "Effects")

        # TAB: REPAIR (OBJECT REMOVAL)
        _TAB_VISIBLE = _tab_open(tab_repair)
        with tab_repair:
            st.markdown("**Remove Objects**")
            st.caption("Select the blemish or object to remove with the same selection tools as Masks (below). The selected area is painted over by blending in nearby texture.")

            removal_signature = _render_selection_stack("removal_mask_layers", "rm", mask_source, include_opacity=False)
            removal_active = any(layer["enabled"] for layer in st.session_state.removal_mask_layers)

            if not st.session_state.removal_mask_layers:
                st.caption("No selections yet — nothing will be removed.")
            elif not removal_active:
                st.caption("All selections are disabled — nothing will be removed.")

            removal_radius = st.slider(
                "Repair Softness", 1, 15, 3, key="rm-radius", disabled=not removal_active,
                help="How far the repair blends into the surrounding area -- higher softens a wider edge.",
            )

            if st.checkbox("👁️ Preview removal selection", key="rm-preview-overlay") and st.session_state.removal_mask_layers:
                # Yellow (vs. Masks' red) so the two overlays are never confused with each other.
                removal_preview_base = preview_image(mask_source, (1024, 1024))
                removal_preview_mask = Image.open(io.BytesIO(_combined_mask_png(mask_source_key, mask_source, removal_signature, removal_preview_base.size)))
                removal_style = st.radio("Overlay", OVERLAY_STYLES, horizontal=True, key="rm-overlay-style")
                st.image(apply_mask_overlay(removal_preview_base, removal_preview_mask, color=(255, 200, 0), style=removal_style), caption="Highlighted = will be removed", use_container_width=True)

        # TAB 4: LAYERS & COMPOSITING
        _TAB_VISIBLE = _tab_open(tab_layers)
        with tab_layers:
            st.markdown("**Layers**")
            st.caption("Add any number of image layers, each with its own opacity and blend mode -- composited on the processing backend when the image renders.")
            layers_signature = _render_layer_stack(source)

        # TAB 5: MASKS & LOCAL ADJUSTMENTS
        _TAB_VISIBLE = _tab_open(tab_masks)
        with tab_masks:
            st.markdown("**Mask Layers**")
            st.caption("Build several named selections and combine them into one mask (union). The target adjustments below apply only where the combined mask is active.")

            # Union of every enabled layer (or an all-zero, no-op mask when there are none) is built
            # from this signature by _combined_mask_png(), cached.
            mask_signature = _render_selection_stack("mask_layers", "m", mask_source)

            if not st.session_state.mask_layers:
                st.caption("No mask layers yet — the sliders below have no effect until you add one.")
            elif not any(layer["enabled"] for layer in st.session_state.mask_layers):
                st.caption("All mask layers are disabled — the sliders below have no effect.")

            if st.checkbox("👁️ Preview combined mask overlay", key="m-preview-overlay") and st.session_state.mask_layers:
                # Overlaid on mask_source (lens-corrected, but otherwise untouched) -- the
                # backend-processed preview isn't available locally, so this debug view shows the
                # selection rather than the live edited result.
                mask_preview_base = preview_image(mask_source, (1024, 1024))
                mask_preview = Image.open(io.BytesIO(_combined_mask_png(mask_source_key, mask_source, mask_signature, mask_preview_base.size)))
                mask_style_col, mask_color_col = st.columns([3, 1])
                mask_style = mask_style_col.radio("Overlay", OVERLAY_STYLES, horizontal=True, key="m-overlay-style")
                mask_color = mask_color_col.color_picker("Color", "#DC3723", key="m-overlay-color")
                st.image(apply_mask_overlay(mask_preview_base, mask_preview, color=_hex_to_rgba(mask_color)[:3], style=mask_style), caption="Highlighted = selected by the combined mask", use_container_width=True)

            shared_mask_layers = [layer for layer in st.session_state.mask_layers if not layer.get("adjust")]
            own_mask_layers = _own_adjustment_layers(st.session_state.mask_layers)
            own_signatures = [_selections_signature([layer]) for layer in own_mask_layers]
            mask_inert = not any(layer["enabled"] for layer in shared_mask_layers)
            m_ev = st.slider("Target Exposure (EV)", -5.0, 5.0, 0.0, 0.05, key="m-tev", disabled=mask_inert, help="Exposure change in stops, applied only inside the combined mask's selected area.")
            m_cont = st.slider("Target Contrast", -100, 100, 0, key="m-tcontrast", disabled=mask_inert, help="Contrast adjustment applied only inside the combined mask's selected area.")
            m_sat = st.slider("Target Saturation", -100, 100, 0, key="m-tsaturation", disabled=mask_inert, help="Saturation adjustment applied only inside the combined mask's selected area.")
            _section_reset_button("mask_targets", "Mask targets")

        # TAB 6: PORTRAIT RETOUCHING
        _TAB_VISIBLE = _tab_open(tab_portrait)
        with tab_portrait:
            st.markdown("**Face & Skin Retouching**")
            # Defaults to 0 (no change) so skin smoothing is opt-in, not silently forced onto every
            # export just because this tab exists in the workspace.
            p_smooth = st.slider(
                "Skin Smoothing", 0.0, 1.0, 0.0, 0.01, key="port-sm",
                help="0 leaves skin untouched. Increase to smooth detected skin tones.",
            )
            if face_detection_backend() == "none":
                st.caption("👤 Face detection is unavailable in this OpenCV build (see System Health).")
            else:
                if _TAB_VISIBLE:
                    st.caption(f"👤 Detected faces: {_face_count(source_digest, source)}")

        # TAB 7: PRESETS
        _TAB_VISIBLE = _tab_open(tab_presets)
        with tab_presets:
            st.markdown("**Preset Library**")
            p_store = PresetStore()
            # This tab runs on every rerun, so an unreadable presets.json (hand-edited, or truncated
            # by a crash mid-write in an older build -- writes are atomic now) used to raise here
            # and take the whole page down. Degrade to "no presets" plus a way out instead.
            try:
                p_names = p_store.names()
                p_store_ok = True
            except ValueError:
                p_names, p_store_ok = [], False
                st.error(f"⚠️ The saved presets file can't be read: `{p_store.path}`")
                if st.button("Move it aside and start fresh", key="p-reset-btn"):
                    backup = p_store.reset()
                    st.toast(f"Moved the unreadable file to {backup.name}", icon="🗂️")
                    _request_rerun()
            st.caption(f"Saved: {', '.join(p_names) if p_names else 'None'}")

            # Load/apply/delete a saved custom preset -- p_store.get()/.delete() already existed
            # and were tested, but nothing in the UI ever called them: a preset could be saved,
            # exported, and imported, but never actually reapplied to a photo. Routed through the
            # same "_pending_restore" + _request_rerun() mechanism as Undo/Redo/Load Project, since
            # a preset is just a recipe like any other.
            if p_names:
                st.markdown("**Load a Preset**")
                p_load_name = st.selectbox("Preset to load", p_names, key="p-load-sel", label_visibility="collapsed")
                if _TAB_VISIBLE:
                    st.image(
                        _custom_preset_thumbnail(source_digest, source, p_store.get(p_load_name)),
                        width=96, caption=p_load_name,
                    )
                p_load_col, p_delete_col = st.columns(2)
                if p_load_col.button("📂 Apply", key="p-load-btn", use_container_width=True):
                    # without_per_photo_only_keys(): a preset saved before AI Upscale was excluded
                    # from presets must not switch it on for this photo either.
                    st.session_state["_pending_restore"] = without_per_photo_only_keys(p_store.get(p_load_name))
                    st.toast(f"Applied preset '{p_load_name}'", icon="📂")
                    _request_rerun()
                if p_delete_col.button("🗑️ Delete", key="p-delete-btn", use_container_width=True):
                    p_store.delete(p_load_name)
                    st.toast(f"Deleted preset '{p_load_name}'", icon="🗑️")
                    _request_rerun()

            p_new_name = st.text_input("New Preset Name", key="p-name-in")
            if st.button("Save Current Look as Preset", key="p-save-btn", disabled=not p_store_ok):
                if p_new_name.strip():
                    # The full recipe (every Tone/Lens/Color/Detail/Masks/... control, plus mask
                    # layers), not just 5 basic fields -- a preset saved after building a mask or a
                    # lens correction used to silently drop all of that. Deliberately excludes the
                    # layer overlay image, same as Undo/Redo history (see _build_recipe's
                    # docstring): a preset is a reusable style, not tied to one photo's overlay.
                    # Also excludes AI Upscale (see without_per_photo_only_keys()): a look shouldn't
                    # carry a minutes-long neural upscale into every photo it's applied to.
                    p_store.save(p_new_name.strip(), without_per_photo_only_keys(_build_recipe()))
                    st.toast(f"Preset '{p_new_name.strip()}' saved", icon="💾")
                    _request_rerun()
                else:
                    st.toast("Type a name before saving a preset", icon="⚠️")

            st.markdown("---")
            st.markdown("**Export / Import Presets**")
            if p_names:
                st.download_button(
                    "⬇️ Export all presets",
                    export_presets(p_store),
                    file_name="photo-studio-presets.json",
                    mime="application/json",
                    key="preset-export-btn",
                )
            else:
                st.caption("No saved presets to export yet.")
            preset_import_file = st.file_uploader("Import presets (.json)", type=["json"], key="preset-import-upload")
            if preset_import_file and st.button("Import", key="preset-import-btn"):
                try:
                    imported = import_presets(p_store, preset_import_file.getvalue())
                except ValueError as error:
                    st.toast(f"Could not import presets: {error}", icon="⚠️")
                else:
                    if imported:
                        st.toast(f"Imported {len(imported)} preset(s)", icon="📥")
                        _request_rerun()
                    else:
                        # Existing names win (import_presets(overwrite=False)) -- every name in the
                        # file already exists locally, so nothing changed and nothing needs a rerun.
                        st.toast("No new presets to import — names already exist", icon="ℹ️")

            st.markdown("---")
            if "edit_history" not in st.session_state:
                st.session_state.edit_history = EditHistory()
            history = st.session_state.edit_history
            history.record(_build_recipe())
            st.caption(
                "↶ Undo / ↷ Redo and 📌 Named Snapshots are always available in the bar above the "
                "canvas, not just here."
            )

            st.markdown("---")
            st.markdown("**Project File**")
            st.caption("Save every adjustment (plus the source image and any layers) into one file you can reopen later.")
            # One click: the archive is built only when the button is pressed (a data callable), from
            # the recipe and layer stack as they are on this rerun. It used to be two steps (package,
            # then a separate download button), and it flattened Text/Shape/Path layers to pixels.
            project_layers = json.loads(json.dumps(st.session_state.get("image_layers", [])))
            project_estimate = _estimated_project_bytes(upload_bytes, project_layers)
            project_too_big = project_estimate > PROJECT_MAX_BYTES
            if project_too_big:
                st.warning(
                    f"⚠️ This project would be about {project_estimate / 1024**2:.0f} MB -- over the "
                    f"{PROJECT_MAX_BYTES // 1024**2} MB a project can be reopened with. Remove or shrink some image layers."
                )
            st.download_button(
                "💾 Save Project (.photoproject)",
                data=functools.partial(_project_archive, upload_name, upload_bytes, json.loads(json.dumps(_build_recipe())), project_layers),
                file_name=f"{Path(upload_name).stem}.photoproject",
                mime="application/zip",
                key="proj-dl-btn",
                disabled=project_too_big,
            )
            st.caption("Text, Shape and Path layers stay editable when the project is reopened.")

        # TAB 8: MORE (Catalog, Batch, System)
        _TAB_VISIBLE = _tab_open(tab_more)
        with tab_more:
            more_mode = st.radio("Tool", ["Library Catalog", "Batch Processor", "System Health"], horizontal=True, label_visibility="collapsed", key="more-mode")
            if more_mode == "Library Catalog":
                lib_folder = st.text_input("Folder", str(Path.home() / "Pictures"), key="lib-f-in")
                # The per-user data folder (migrated from the old ~/.photo-studio on first use).
                lib_db = user_data_file("catalog.sqlite")
                if st.button("Scan Folder", key="lib-btn"):
                    lib_progress_bar = st.progress(0.0)
                    lib_progress_status = st.empty()

                    def _update_scan_progress(index: int, total: int, info) -> None:
                        lib_progress_bar.progress(index / total if total else 1.0)
                        lib_progress_status.caption(f"Scanning {index}/{total} — {info.get('name', '')}")

                    st.session_state.records = index_folder(lib_folder, lib_db, True, progress=_update_scan_progress)
                    st.session_state["lib-cursor"] = 0
                    lib_progress_bar.progress(1.0)
                    lib_progress_status.caption(f"Done — indexed {len(st.session_state.records)} photo(s)")
                    st.toast(f"Indexed {len(st.session_state.records)} photo(s)", icon="✅")
                # The catalog is shown without a rescan (it used to appear only after "Scan Folder"
                # in the same session): photos catalogued earlier that are still on disk.
                rec = st.session_state.get("records") or [
                    {"path": entry["path"], "name": Path(entry["path"]).name}
                    for entry in PhotoCatalog(lib_db).search() if Path(entry["path"]).is_file()
                ]
                q = st.text_input("Search", key="lib-q-in")
                r_min = st.slider("Min Stars", 0, 5, 0, key="lib-r-in")
                search_results = PhotoCatalog(lib_db).search(q, r_min) if rec else []
                if rec and not search_results:
                    st.caption("No photos match this search / star filter.")
                if search_results and _TAB_VISIBLE:
                    # Thumbnail grid (cached JPEG thumbnails under the user data folder, RAW via its
                    # embedded preview) with an Edit button that opens the photo in the editor.
                    st.caption("✏️ Edit opens the photo in the editor with the current adjustments kept (↺ resets them).")
                    grid_cols = st.columns(4)
                    for grid_index, entry in enumerate(search_results[:24]):
                        entry_path = Path(entry["path"])
                        with grid_cols[grid_index % 4]:
                            try:
                                st.image(str(thumbnail(entry_path, user_data_dir() / "thumbs")), caption=f"{'⭐' * int(entry['rating'])} {entry_path.name}", use_container_width=True)
                            except (OSError, ValueError):
                                st.caption(f"{entry_path.name} (no preview)")
                            if st.button("✏️ Edit", key=f"lib-edit-{image_digest(str(entry_path).encode('utf-8'))[:16]}", use_container_width=True):
                                _open_library_photo(entry_path)
                                _request_rerun()
                elif search_results:
                    st.dataframe(search_results, use_container_width=True)

                if rec:
                    st.markdown("**Rate & Flag**")
                    st.caption("Keyboard: 1-5 rate, 0 clears, X toggles reject, ←/→ move Prev/Next.")
                    cursor = max(0, min(st.session_state.get("lib-cursor", 0), len(rec) - 1))
                    current_path = rec[cursor]["path"]
                    catalog = PhotoCatalog(lib_db)
                    existing = catalog.get(current_path)
                    is_rejected = "rejected" in existing["tags"]

                    st.caption(f"{cursor + 1} / {len(rec)} — {Path(current_path).name}" + (" — 🚫 rejected" if is_rejected else ""))
                    try:
                        # The cached thumbnail, not the full-size file decoded on every rerun.
                        st.image(str(thumbnail(current_path, user_data_dir() / "thumbs")), width=240)
                    except Exception:
                        st.caption("Preview unavailable.")
                    if Path(current_path).is_file() and st.button("✏️ Edit this photo", key="lib-edit-current-btn"):
                        _open_library_photo(Path(current_path))
                        _request_rerun()

                    with st.container(key="lib-rate-controls"):
                        nav_prev, nav_next = st.columns(2)
                        if nav_prev.button("⬅ Prev", key="lib-prev-btn", use_container_width=True):
                            st.session_state["lib-cursor"] = (cursor - 1) % len(rec)
                            _request_rerun()
                        if nav_next.button("Next ➡", key="lib-next-btn", use_container_width=True):
                            st.session_state["lib-cursor"] = (cursor + 1) % len(rec)
                            _request_rerun()

                        st.caption("Rating (keys 1-5, 0 to clear, X to reject, ←/→ to move)")
                        star_cols = st.columns(5)
                        for i in range(5):
                            label = "⭐" if i < existing["rating"] else "☆"
                            if star_cols[i].button(label, key=f"lib-star-{i + 1}", use_container_width=True):
                                catalog.upsert(current_path, rating=i + 1, tags=existing["tags"])
                                _request_rerun()

                        clear_col, reject_col = st.columns(2)
                        if clear_col.button("Clear rating", key="lib-clear-btn", use_container_width=True):
                            catalog.upsert(current_path, rating=0, tags=existing["tags"])
                            _request_rerun()
                        reject_label = "🚫 Un-reject" if is_rejected else "🚫 Reject"
                        if reject_col.button(reject_label, key="lib-reject-btn", use_container_width=True):
                            new_tags = [t for t in existing["tags"] if t != "rejected"] if is_rejected else existing["tags"] + ["rejected"]
                            catalog.upsert(current_path, rating=existing["rating"], tags=new_tags)
                            _request_rerun()

                        visible_tags = ", ".join(t for t in existing["tags"] if t != "rejected")
                        tags_text = st.text_input("Tags (comma-separated)", value=visible_tags, key=f"lib-tags-{cursor}")
                        if st.button("Save tags", key="lib-tags-save-btn"):
                            new_tags = [t.strip() for t in tags_text.split(",") if t.strip()]
                            if is_rejected:
                                new_tags.append("rejected")
                            catalog.upsert(current_path, rating=existing["rating"], tags=new_tags)
                            st.toast("Tags saved", icon="🏷️")

                    components.html(_LIBRARY_KEYBOARD_SHORTCUTS_HTML, height=0)
            elif more_mode == "Batch Processor":
                # Every supported format, RAW included (app.raw_import.SUPPORTED_EXTENSIONS) -- this
                # used to accept only JPG/PNG.
                b_uploaded = st.file_uploader("Batch Files", type=uploader_types(), accept_multiple_files=True, key="b-up")
                b_prof = st.selectbox("Profile", list(EXPORT_PROFILES), key="b-p-sel")
                # "None" + built-in named looks (app.presets.PRESETS, a handful of fixed fields) +
                # every saved custom preset (app.custom_presets.PresetStore, a full recipe --
                # everything the Presets tab above can save, minus mask_layers/removal_mask_layers/
                # layer, which are photo-specific and don't transfer across different photos in a
                # batch -- see app.recipe_options.recipe_to_options()'s docstring). The " (custom)"
                # suffix is just this dropdown's own label, stripped back off below before the
                # lookup -- p_store.names() themselves are never touched.
                b_preset_options = ["None", *sorted(PRESETS), *[f"{name} (custom)" for name in p_names]]
                b_preset_choice = st.selectbox(
                    "Preset", b_preset_options, key="b-preset-sel",
                    help="A saved custom preset applies its full recipe (tone/color/detail/lens) to every photo in the batch -- the same thing Lightroom calls syncing develop settings across a shoot.",
                )
                if b_preset_choice.endswith(" (custom)"):
                    # Only the custom presets lacked any visual preview at all before -- the 4
                    # built-in looks already show themselves clearly by name and don't carry a
                    # per-preview pipeline-run cost, so this stays scoped to the gap that existed.
                    st.image(
                        _custom_preset_thumbnail(source_digest, source, p_store.get(b_preset_choice[: -len(" (custom)")])),
                        width=96,
                    )
                run_col, stop_col = st.columns([2, 1])
                # Clicking Stop while a batch runs makes Streamlit rerun this script, which ends the
                # running batch at its next progress update (the files finished so far stay in the ZIP).
                stop_col.button("⏹ Stop", key="b-stop-btn", use_container_width=True, help="Stop the running batch")
                if b_uploaded and run_col.button("Run Batch", key="b-run-btn", use_container_width=True):
                    if b_preset_choice.endswith(" (custom)"):
                        b_preset, b_recipe = None, p_store.get(b_preset_choice[: -len(" (custom)")])
                    else:
                        b_preset, b_recipe = (b_preset_choice if b_preset_choice != "None" else None), None

                    b_progress_bar = st.progress(0.0)
                    b_progress_status = st.empty()

                    def _update_batch_progress(index: int, total: int, last_result) -> None:
                        b_progress_bar.progress(index / total if total else 1.0)
                        label = f"Processing {index}/{total}"
                        if last_result is not None:
                            label += f" — {last_result.source.name} ({last_result.status})"
                        b_progress_status.caption(label)

                    # Straight from the uploaded bytes (no temp files) into a ZIP on disk under the
                    # user data folder; only the newest few are kept.
                    b_out_dir = user_data_dir() / "batch-output"
                    b_out_dir.mkdir(parents=True, exist_ok=True)
                    for old_zip in sorted(b_out_dir.glob("batch-*.zip"))[:-2]:
                        old_zip.unlink(missing_ok=True)
                    b_zip_path = b_out_dir / f"batch-{uuid.uuid4().hex[:12]}.zip"
                    rep = stream_to_zip(
                        [FileLike(f.name, f.getvalue()) for f in b_uploaded], b_zip_path,
                        preset=b_preset, profile=b_prof, progress=_update_batch_progress, recipe=b_recipe,
                    )
                    b_progress_bar.progress(1.0)
                    b_progress_status.caption(f"Done — {sum(1 for x in rep if x.status == 'ok')}/{len(rep)} succeeded")
                    st.session_state["_batch_result"] = (str(b_zip_path), [{"source": x.source.name, "status": x.status, "error": x.error or ""} for x in rep])
                batch_result = st.session_state.get("_batch_result")
                if batch_result and Path(batch_result[0]).exists():
                    st.dataframe(batch_result[1], use_container_width=True)
                    st.download_button(
                        "Download ZIP", functools.partial(Path(batch_result[0]).read_bytes), "batch.zip", "application/zip", key="b-dl",
                    )
            else:
                from app.diagnostics import check_environment
                st.dataframe([{"check": c.name, "status": c.status, "message": c.message} for c in check_environment()], use_container_width=True)

        # Every tab above runs on every Streamlit rerun regardless of which tab is visible, and none of
        # them mutate `processor` anymore -- they only capture control values. The actual pixel pipeline
        # (Lens -> Tone -> Color -> Detail -> Repair -> Layers -> Masks -> Portrait -> Crop, matching
        # the tab order above) runs once, here, on app/backend.py -- see that module's process_image()
        # for the mirrored order.
        #
        # Note: the deferred-rerun consume check for _request_rerun() (used by tab bodies above)
        # deliberately does NOT live here -- canvas_col, below, still has its own widgets left to
        # instantiate this pass (zoom, split view, pan, export, Undo/Redo/Reset), and rerunning
        # before they render would garbage-collect them exactly like the bug _request_rerun() exists
        # to prevent (a real instance of this was found in review). See the check at the very bottom
        # of this file, after canvas_col.

        options = {
            "rotate": geo_rotate,
            "flip_horizontal": geo_flip_h,
            "flip_vertical": geo_flip_v,
            "straighten": geo_straighten,
            "constrain_crop": geo_constrain,
            "lens_distortion": lens_distortion,
            "lens_vignette_amount": lens_vignette_amount,
            "lens_vignette_midpoint": lens_vignette_midpoint,
            "lens_chromatic_aberration": lens_chromatic_aberration,
            "lens_perspective_v": lens_perspective_v,
            "lens_perspective_h": lens_perspective_h,
            "exposure_ev": exp_ev,
            "contrast": 1.0 + cont_amount / 100.0,
            "highlights": high_val,
            "shadows": shad_val,
            "preset": preset_choice,
            "temperature": temp_val,
            "tint": tint_val,
            "saturation": 1.0 + sat_amount / 100.0,
            "vibrance": vib_val,
            "hsl_bands": hsl_bands,
            "cg_shadows_hue": cg_shadows_hue, "cg_shadows_intensity": cg_shadows_intensity,
            "cg_midtones_hue": cg_midtones_hue, "cg_midtones_intensity": cg_midtones_intensity,
            "cg_highlights_hue": cg_highlights_hue, "cg_highlights_intensity": cg_highlights_intensity,
            "cg_balance": cg_balance,
            "curve": merge_curve_points(
                [(0, c_black), (64, c_low), (128, c_mid), (192, c_high), (255, c_white)], st.session_state.get("curve_extra_points", [])
            ) if use_curve else None,
            "texture": text_val,
            "clarity": clar_val,
            "dehaze": haze_val,
            "denoise": denoise_val,
            "denoise_color": denoise_color_val,
            "sharpen": sharp_en,
            "effects_vignette_amount": fx_vignette_amount,
            "effects_vignette_midpoint": fx_vignette_midpoint,
            "effects_grain_amount": fx_grain_amount,
            "removal_radius": removal_radius,
            "mask_exposure_ev": m_ev,
            "mask_contrast": 1.0 + m_cont / 100.0,
            "mask_saturation": 1.0 + m_sat / 100.0,
            "local_adjustments": [dict(layer["adjustments"]) for layer in own_mask_layers] or None,
            "portrait_smoothing": p_smooth,
            "aspect": aspect_code,
            "crop_box": list(crop_box) if crop_box else None,
        }

        # Interactive preview at proxy resolution (P1 in the improvement report) for a photo much
        # larger than the canvas can show -- see PROXY_LONG_EDGE. The 1:1 zoom views (>100%) read
        # real pixels, so they keep the full-resolution render. The zoom selectbox renders later
        # (canvas_col), so its last value is read from session_state, like the Crop tab reads the
        # Guides & Grid panel's.
        full_res_view = float(st.session_state.get("cv-zoom", 0.0)) >= 1.0
        use_proxy = max(source.size) > PROXY_TRIGGER_LONG_EDGE and not full_res_view
        mask_size = None
        if use_proxy:
            stage_input = _proxy_source_png(source_digest, source, PROXY_LONG_EDGE)
            stage_input_name, stage_input_key = "proxy.png", f"{source_digest}|proxy{PROXY_LONG_EDGE}"
            proxy_size = Image.open(io.BytesIO(stage_input)).size
            # Scales the pipeline's pixel-unit radii and the full-resolution crop_box (app.pipeline).
            options["preview_scale"] = proxy_size[0] / source.width
            # Masks apply after the geometry step, so they're built at the proxy's *post-geometry*
            # size (a 90° rotation swaps it, constrain crop shrinks it).
            mask_size = geometry_output_size(proxy_size, options)
        else:
            stage_input, stage_input_name, stage_input_key = upload_bytes, upload_name, source_digest

        # Masks and layers come out of their own caches (_combined_mask_png/_layers_payload) --
        # each is passed to the Stage 2 cache as a short key plus unhashed bytes.
        local_mask_bytes = local_mask_key = None
        if shared_mask_layers:
            local_mask_bytes = _combined_mask_png(mask_source_key, mask_source, mask_signature, mask_size)
            local_mask_key = image_digest(f"{mask_source_key}|{mask_signature}|{mask_size}".encode("utf-8"))
        local_masks_bytes = local_masks_key = None
        if own_signatures:
            local_masks_bytes = _local_masks_payload([_combined_mask_png(mask_source_key, mask_source, signature, mask_size) for signature in own_signatures])
            local_masks_key = image_digest(f"{mask_source_key}|{own_signatures}|{mask_size}".encode("utf-8"))
        removal_mask_bytes = removal_mask_key = None
        if removal_active:
            removal_mask_bytes = _combined_mask_png(mask_source_key, mask_source, removal_signature, mask_size)
            removal_mask_key = image_digest(f"{mask_source_key}|{removal_signature}|{mask_size}".encode("utf-8"))
        # Layers are composited after the geometry step, so they're rendered at that frame's size --
        # the proxy's when previewing through one (P1: a photo-sized layer used to be rasterized
        # and sent at full resolution for every preview), with pixel sizes scaled to match.
        full_canvas = geometry_output_size(source.size, options)
        layers_bytes = None
        if layers_signature:
            if use_proxy:
                layers_bytes = _layers_payload(layers_signature, tuple(mask_size), mask_size[0] / max(1, full_canvas[0]))
            else:
                layers_bytes = _layers_payload(layers_signature, tuple(full_canvas))
        layers_key = image_digest(layers_bytes) if layers_bytes else None

        detail_options = {key: options[key] for key in (*_DETAIL_STAGE_OPTION_KEYS, *_BOTH_STAGES_OPTION_KEYS) if key in options}
        remaining_options = {key: value for key, value in options.items() if key not in _DETAIL_STAGE_OPTION_KEYS}
        try:
            stage_a_png_bytes, stage_a_handle = _cached_process_through_detail(stage_input_key, stage_input, stage_input_name, detail_options)
            stage_a_key = image_digest(f"{stage_input_key}|{json.dumps(detail_options, sort_keys=True)}".encode("utf-8"))
            edited_image = _cached_process_from_repair_onward(
                stage_a_key, stage_a_png_bytes, remaining_options,
                local_mask_key, local_mask_bytes, layers_key, layers_bytes, removal_mask_key, removal_mask_bytes,
                stage_a_handle, local_masks_key, local_masks_bytes,
            )
        except SupersededError:
            # A newer edit from this session replaced this render on the backend -- this run is
            # already obsolete (its rerun is underway), so there is nothing to show or report.
            st.stop()
        except RuntimeError as error:
            st.error(f"⚠️ {error}")
            st.stop()
        # What an export will measure -- on a proxy preview, the real (full-resolution) output size,
        # not the proxy's.
        output_size = expected_output_size(source.size, options) if use_proxy else edited_image.size

    _TAB_VISIBLE = True  # the canvas column is always on screen
    with canvas_col:
        # "Original" / Split View compare against the source in the edit's orientation (rotate/flip
        # only), so a rotated edit isn't compared with a sideways original.
        oriented_source = orient(source, geo_rotate, geo_flip_h, geo_flip_v)
        # Canvas Header Bar
        top_c1, top_undo, top_redo, top_reset, top_snap, top_c2, top_c3 = st.columns(
            [0.85, 0.18, 0.18, 0.18, 0.22, 0.65, 0.95], gap="small"
        )
        with top_c1:
            # html.escape() here: upload_name is the uploaded file's own filename (user-controlled),
            # interpolated into both an HTML attribute (title=) and element text below via
            # unsafe_allow_html -- unescaped, a filename containing a quote or angle bracket could
            # break out of the attribute or inject markup.
            live_view_label = html.escape(f"{upload_name} ({output_size[0]}×{output_size[1]})")
            st.markdown(
                f'<div class="panel-header-title canvas-live-view" title="{live_view_label}">🖼️ Live View: <span>{live_view_label}</span></div>',
                unsafe_allow_html=True,
            )
        with top_undo:
            if st.button("↶", key="hist-undo-btn", disabled=not history.can_undo, help="Undo (Ctrl+Z)", use_container_width=True, shortcut="Ctrl+Z"):
                st.session_state["_pending_restore"] = history.undo() or {}
                _request_rerun()
        with top_redo:
            if st.button("↷", key="hist-redo-btn", disabled=not history.can_redo, help="Redo (Ctrl+Shift+Z)", use_container_width=True, shortcut="Ctrl+Shift+Z"):
                st.session_state["_pending_restore"] = history.redo() or {}
                _request_rerun()
        with top_reset:
            # Resets exactly the same "recipe" fields Undo/Redo operate on (see _build_recipe), so it
            # deliberately leaves an uploaded layer/RAW source alone -- only the adjustment values reset.
            if st.button("↺", key="hist-reset-btn", help="Reset all adjustments to defaults", use_container_width=True):
                st.session_state["_pending_restore"] = {**RECIPE_DEFAULTS, "mask_layers": [], "removal_mask_layers": [], "curve_extra_points": []}
                _request_rerun()
        with top_snap:
            # Named Snapshots used to live buried at the bottom of the Presets tab (10th of 11
            # tabs), below preset import/export -- Undo/Redo/Reset were already moved up here for
            # exactly this "buried in Presets" reason (see docs/PROJECT_STATUS.md's history), but
            # Snapshots wasn't, until now. `history` (and its .record() call for this rerun) is
            # still created inside tab_presets above, which -- like every other tab body -- runs
            # earlier in the same script pass than canvas_col, so it's already populated by the
            # time this popover reads history.snapshot_names below.
            with st.popover(
                "📌", use_container_width=True,
                help="Named Snapshots -- bookmark the current look under a name and jump back to it any time without losing Undo/Redo history",
            ):
                st.markdown("**Named Snapshots**")
                st.caption(
                    "Bookmark the current look under a name and jump back to it any time -- unlike "
                    "Undo/Redo, restoring a snapshot never discards it (or any other saved snapshot)."
                )
                snap_new_name = st.text_input(
                    "Snapshot name", key="snap-name-in", label_visibility="collapsed", placeholder="Snapshot name",
                    help="Name for the bookmark you're about to save.",
                )
                if st.button("📌 Save Snapshot", key="snap-save-btn", use_container_width=True):
                    if snap_new_name.strip():
                        history.save_snapshot(snap_new_name.strip(), _build_recipe())
                        st.toast(f"Snapshot '{snap_new_name.strip()}' saved", icon="📌")
                        _request_rerun()
                    else:
                        st.toast("Type a name before saving a snapshot", icon="⚠️")
                if history.snapshot_names:
                    # Keyed by the snapshot's own name (stable), not list position -- the same
                    # UUID-keyed-list convention CLAUDE.md documents for mask/image layers, so a
                    # Restore/Delete here never risks the "later item rendered twice under a shifted
                    # index" StreamlitDuplicateElementKey bug a positionally-keyed list could hit.
                    # Still `break` right after either mutation, matching this codebase's established
                    # per-item-list-widget convention exactly rather than relying only on that
                    # argument.
                    for snap_name in history.snapshot_names:
                        snap_row1, snap_row2, snap_row3 = st.columns([2, 1, 0.6])
                        snap_row1.caption(snap_name)
                        if snap_row2.button("↩️ Restore", key=f"snap-restore-{snap_name}", use_container_width=True, help=f"Restore snapshot '{snap_name}'"):
                            st.session_state["_pending_restore"] = history.restore_snapshot(snap_name) or {}
                            st.toast(f"Restored snapshot '{snap_name}'", icon="↩️")
                            _request_rerun()
                            break
                        if snap_row3.button("🗑️", key=f"snap-delete-{snap_name}", use_container_width=True, help=f"Delete snapshot '{snap_name}'"):
                            history.delete_snapshot(snap_name)
                            _request_rerun()
                            break
                else:
                    st.caption("No snapshots saved yet.")
        with top_c2:
            # Fit shows the whole photo; 100 % is a true 1:1 view of its pixels (rendered at full
            # resolution even for a proxy-previewed photo); 200 %/400 % enlarge those pixels.
            zoom_level = st.selectbox(
                "Zoom View", [0.0, 1.0, 2.0, 4.0], key="cv-zoom", label_visibility="collapsed",
                format_func=lambda z: "🔍 Fit" if z < 1 else ("🔍 100% (1:1)" if z == 1 else f"🔍 {int(z * 100)}%"),
            )
        with top_c3:
            compare_mode = st.checkbox("⇄ Split View", value=False, key="cv-split", help="Drag to compare before/after")
            show_original = st.checkbox("👁️ Original", value=False, key="cv-orig", disabled=compare_mode)

        # Keyboard shortcuts (native, F11): Z = 1:1 / Fit, B = before/after, G = grid. Ctrl+Z /
        # Ctrl+Shift+Z are on Undo/Redo, Ctrl+E on the Save button.
        key_c1, key_c2, key_c3 = st.columns(3)
        key_c1.button("🔍 1:1 / Fit", key="kb-zoom-btn", on_click=_toggle_one_to_one, shortcut="Z", use_container_width=True, help="Toggle a true 1:1 view (Z)")
        key_c2.button("👁 Before / After", key="kb-orig-btn", on_click=_toggle_original, shortcut="B", use_container_width=True, help="Show the original (B)")
        key_c3.button("# Grid", key="kb-grid-btn", on_click=_toggle_grid, shortcut="G", use_container_width=True, help="Toggle the composition grid (G)")

        if zoom_level >= 1.0:
            # Navigator: click the overview to center the zoomed view there (the same hidden
            # text_area commit pattern as the other on-canvas pickers). Handled before the pan
            # sliders render, so it may set their values this run.
            nav_view = preview_image(edited_image, (360, 360))
            nav_box = zoom_viewport(edited_image.size, zoom_level, (st.session_state.get("cv-panx", 0.5), st.session_state.get("cv-pany", 0.5)))
            scale_x, scale_y = nav_view.width / edited_image.width, nav_view.height / edited_image.height
            nav_view = draw_viewport_box(nav_view, tuple(int(v * s) for v, s in zip(nav_box, (scale_x, scale_y, scale_x, scale_y))))
            nav_html, nav_height = _wand_picker_html(nav_view, "cv-nav", hint="Click to center the zoomed view there.")
            components.html(nav_html, height=nav_height)
            with st.container(key="cv-nav-wand-hidden"):
                pending_nav = st.text_area("Navigator click", key="cv-nav-wand-data", label_visibility="collapsed")
            if _consume_new_commit(pending_nav, "cv-nav-committed"):
                try:
                    nav_click = json.loads(pending_nav)
                    st.session_state["cv-panx"] = round(min(1.0, max(0.0, float(nav_click["x"]))), 2)
                    st.session_state["cv-pany"] = round(min(1.0, max(0.0, float(nav_click["y"]))), 2)
                except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                    pass
            pan_c1, pan_c2 = st.columns(2)
            with pan_c1:
                pan_x = st.slider("Pan Horizontal", 0.0, 1.0, 0.5, 0.01, key="cv-panx")
            with pan_c2:
                pan_y = st.slider("Pan Vertical", 0.0, 1.0, 0.5, 0.01, key="cv-pany")

        # Guides & Grid -- a pure composition/alignment aid, the same "view-only, never sent
        # through the backend or included in an export" pattern as the clipping-warning and Masks
        # preview overlays below. Lives here (not in a tool tab) because, like zoom/pan/Split View,
        # it's a canvas viewing setting rather than a per-photo edit -- present regardless of which
        # tool tab is open. Read (not written) by the Crop tab above for on-canvas snapping, which
        # is why session_state is initialized with plain dict/list defaults rather than only via
        # the widgets below: the Crop tab renders earlier in the script and needs a value to read
        # even on the very first run, before this expander has created its own widgets yet.
        if "guides" not in st.session_state:
            st.session_state.guides = []
        with st.expander("🧭 Guides & Grid", expanded=False):
            st.caption("View-only -- never affects the exported image. The free-form Crop tool's corner handles snap to these lines.")
            grid_type = st.selectbox("Grid overlay", list(GRID_PRESETS), key="grid-type")
            guides = st.session_state.guides
            for guide in guides:
                guide.setdefault("id", uuid.uuid4().hex)
            add_col1, add_col2, add_col3 = st.columns([1, 1.4, 0.8])
            with add_col1:
                new_guide_orientation = st.selectbox("Orientation", ["Horizontal", "Vertical"], key="guide-new-orientation", label_visibility="collapsed")
            with add_col2:
                new_guide_position = st.slider("Position %", 0.0, 100.0, 50.0, key="guide-new-position", label_visibility="collapsed")
            with add_col3:
                if st.button("➕ Add", key="guide-add-btn"):
                    guides.append({"id": uuid.uuid4().hex, "orientation": new_guide_orientation.lower(), "position_pct": new_guide_position})
                    _request_rerun()
            for idx, guide in enumerate(guides):
                guide_col1, guide_col2 = st.columns([3, 1])
                with guide_col1:
                    st.caption(f"{guide['orientation'].capitalize()} guide at {guide['position_pct']:.0f}%")
                with guide_col2:
                    if st.button("🗑️ Remove", key=f"guide-rm-{guide['id']}"):
                        # Same "stop iterating a list just mutated mid-loop" reasoning as every
                        # other per-item Remove button in this file (see _render_layer_stack).
                        guides.pop(idx)
                        _request_rerun()
                        break

        # Main Image Canvas
        if compare_mode:
            # Split View: a draggable before/after slider (see _before_after_html) rather than a
            # static side-by-side -- takes priority over the plain "Original" toggle since it
            # already shows both. The two previews share the same zoom/pan as the normal view; if
            # a crop left them different sizes, resize "before" to match so the overlay lines up.
            before_preview = zoom_preview(oriented_source, zoom_level, (pan_x, pan_y))
            after_preview = zoom_preview(edited_image, zoom_level, (pan_x, pan_y))
            if before_preview.size != after_preview.size:
                before_preview = before_preview.resize(after_preview.size, Image.Resampling.LANCZOS)
            compare_html, compare_height = _before_after_html(before_preview, after_preview)
            components.html(compare_html, height=compare_height + 10)
        else:
            st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
            if show_original:
                canvas_display, canvas_caption = preview_image(oriented_source), "Original Untouched"
            else:
                canvas_display, canvas_caption = zoom_preview(edited_image, zoom_level, (pan_x, pan_y)), "Current Edited Image"
            if grid_type != "None" or guides:
                canvas_display = draw_guides_and_grid(canvas_display, grid_type, guides)
            st.image(canvas_display, caption=canvas_caption, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Histogram -- always reflects edited_image (the backend-processed result), not whichever
        # view (Original / Split) happens to be toggled above, since that's the reference exposure
        # decisions are actually made against.
        # Computed on a display-sized copy (the same shape of distribution, a fraction of the
        # work) -- the full-resolution histogram cost ~0.7 s on every rerun at 24 MP. The waveform
        # and clipping overlay below use the same copy.
        scope_image = preview_image(edited_image, (1024, 1024))
        histogram_bins = 64
        histogram = compute_histogram(scope_image, bins=histogram_bins)
        # st.line_chart plots a dict's values against a plain 0..N-1 index by default, which would
        # label this axis in bin numbers instead of the pixel brightness (0-255) each bin covers --
        # give it an explicit "Pixel Value" column (each bin's center) to chart against instead.
        histogram["Pixel Value"] = [round(255 * (i + 0.5) / histogram_bins) for i in range(histogram_bins)]
        st.line_chart(histogram, x="Pixel Value", y=["Red", "Green", "Blue"], height=90, color=["#f87171", "#4ade80", "#60a5fa"])

        # Waveform monitor -- a horizontal-position vs. luminance/RGB-intensity scope (the same
        # concept as a video editor's waveform), complementing the histogram above (which plots
        # value distribution with no position information at all). Off by default so it doesn't
        # clutter the canvas -- both widgets are still always instantiated every rerun (not just
        # the toggle) so toggling it off and back on doesn't lose the channel-mode choice the way
        # a widget only rendered inside the `if` block below would (see CLAUDE.md's "every tab
        # runs every rerun" / conditional-widget-loses-state note); only the image itself (not a
        # stateful widget) is skipped when off. Reflects edited_image, same as the histogram.
        # A short visible caption (not just the checkbox's own help= tooltip on hover) so a
        # first-time user notices this exists at all -- easy to miss otherwise as a plain
        # checkbox below the histogram.
        st.caption("〰️ **Waveform monitor**: an extra scope for checking exposure across the frame, next to the histogram above.")
        waveform_toggle_col, waveform_mode_col = st.columns([1, 1])
        with waveform_toggle_col:
            show_waveform = st.checkbox(
                "〰️ Show waveform monitor", key="waveform-toggle",
                help="Horizontal position vs. luminance/RGB intensity, like a video editor's waveform scope.",
            )
        with waveform_mode_col:
            waveform_mode = st.radio(
                "Waveform channels", ["Luma", "RGB"], key="waveform-mode", horizontal=True,
                label_visibility="collapsed", disabled=not show_waveform,
            )
        if show_waveform:
            st.image(
                compute_waveform(scope_image, mode=waveform_mode),
                caption=f"Waveform ({waveform_mode})",
                use_container_width=True,
            )

        # Clipping warning -- the same idea as Lightroom's blown-highlight/blocked-shadow
        # triangles on the histogram, shown as a full preview overlay instead (matching this app's
        # existing overlay-checkbox pattern -- the Masks/Repair preview checkboxes, and the Soft
        # Proof gamut warning). Reflects edited_image, same as the histogram right above it.
        if st.checkbox("⚠️ Show clipping", key="clip-warning-toggle", help="Highlights blocked shadows (blue) and blown highlights (red)."):
            st.image(
                clipping_warning_overlay(scope_image),
                caption="Blue = shadows clipped to black, Red = highlights clipped to white",
                use_container_width=True,
            )

        # Watermark -- an optional text stamp applied only to exported files (single-format Save
        # below, and Export Multiple Sizes further down), never to the canvas preview or histogram
        # above. Defined ahead of the Export Footer Bar so its values (wm_text/wm_position/
        # wm_opacity/wm_font_size) are already in hand by the time that bar builds out_img, rather
        # than re-reading them back out of session_state after the fact.
        with st.expander("🏷️ Watermark"):
            st.caption("Optional text stamp applied only to exported files, not the canvas preview above.")
            wm_text = st.text_input("Watermark text (blank = none)", key="wm-text-in")
            wm_pos_col, wm_op_col, wm_size_col = st.columns(3)
            with wm_pos_col:
                wm_position = st.selectbox("Position", list(WATERMARK_POSITIONS), index=3, key="wm-pos-sel")
            with wm_op_col:
                wm_opacity = st.slider("Opacity %", 5, 100, 50, key="wm-opacity-sl")
            with wm_size_col:
                wm_font_size = st.slider("Font size", 12, 120, 32, key="wm-size-sl")

        # Export settings -- everything the main Save button applies (P1 "F6" in the improvement
        # report). Every widget renders on every rerun (like every other panel), but nothing is
        # encoded until the button is clicked: the Save button used to encode the full-resolution
        # image on *every* rerun (WEBP ~3.9 s, PNG ~1.6 s at 24 MP) and wrote neither EXIF nor ICC.
        with st.expander("⚙️ Export settings"):
            export_store = ExportPresetStore()
            export_presets = export_store.load()
            if export_presets:
                preset_pick_col, preset_apply_col, preset_delete_col = st.columns([2, 1, 1])
                export_preset_name = preset_pick_col.selectbox("Export preset", sorted(export_presets), key="exp-preset-sel", label_visibility="collapsed")
                preset_apply_col.button(
                    "Apply", key="exp-preset-apply-btn", use_container_width=True,
                    on_click=_apply_export_preset, args=(export_presets[export_preset_name],),
                )
                if preset_delete_col.button("🗑️", key="exp-preset-delete-btn", use_container_width=True, help="Delete this export preset"):
                    export_store.delete(export_preset_name)
                    _request_rerun()
            exp_set1, exp_set2 = st.columns(2)
            with exp_set1:
                exp_long_edge = st.selectbox(
                    "Resize (long edge)", [0, 4096, 3000, 2048, 1600, 1080], key="exp-long-edge",
                    format_func=lambda edge: "Original size" if not edge else f"{edge} px",
                )
                exp_sharpen = st.selectbox(
                    "Output sharpening", list(OUTPUT_SHARPENING), key="exp-sharpen",
                    help="Applied after the resize, for where the image will be seen.",
                )
            with exp_set2:
                exp_metadata = st.selectbox(
                    "Metadata", list(METADATA_POLICIES), format_func=lambda policy: METADATA_POLICY_LABELS[policy], key="exp-metadata",
                    help="EXIF from the original photo (camera, lens, capture time, copyright...). The color profile is always kept.",
                )
                # On by default (R4): a location shouldn't leave the machine unless asked for.
                exp_remove_gps = st.checkbox("Remove GPS location", value=True, key="exp-remove-gps", disabled=exp_metadata != "all")
            # AI Upscale lives here, not in the Detail tab: it runs once, on the final full-resolution
            # result, only when Save is clicked (it used to re-run Real-ESRGAN on every Stage 2
            # change, and its 2x/4x result could crash this page on decode). ai_upscale_installed(),
            # not ai_upscale_available(): it only checks the packages exist, without importing torch
            # into this Streamlit process.
            if ai_upscale_installed():
                exp_upscale = st.selectbox(
                    "AI Upscale (Real-ESRGAN)", [0, 2, 4], key="exp-ai-upscale",
                    format_func=lambda factor: "Off" if not factor else f"{factor}x",
                    help="Neural super-resolution -- the one deliberate exception to this app's otherwise classical, "
                    "CPU-only adjustments (see docs/PROJECT_STATUS.md). Runs on the processing backend (GPU when "
                    "available, much slower on CPU) after every other edit, when you click Save.",
                )
            else:
                exp_upscale = 0
                st.caption("Install requirements-gpu.txt to enable AI Upscale on export.")
            exp_name_template = st.text_input(
                "File name", DEFAULT_NAME_TEMPLATE, key="exp-name-template",
                help="Fields: {stem} (original name), {date}, {width}, {height}, {format}. The extension is added for you.",
            )
            save_preset_col, save_preset_btn_col = st.columns([2, 1])
            new_export_preset = save_preset_col.text_input("Save these settings as", key="exp-preset-name", placeholder="Export preset name", label_visibility="collapsed")
            if save_preset_btn_col.button("💾 Save preset", key="exp-preset-save-btn", use_container_width=True):
                if new_export_preset.strip():
                    export_store.save(new_export_preset.strip(), {
                        "format": st.session_state.get("exp-fmt", "JPEG"), "quality": st.session_state.get("exp-qual", 92),
                        "long_edge": exp_long_edge, "sharpen": exp_sharpen, "metadata": exp_metadata,
                        "remove_gps": exp_remove_gps, "name_template": exp_name_template,
                    })
                    st.toast(f"Export preset '{new_export_preset.strip()}' saved", icon="💾")
                    _request_rerun()
                else:
                    st.toast("Type a name before saving an export preset", icon="⚠️")

        # Export Footer Bar
        st.markdown('<div class="export-panel">', unsafe_allow_html=True)
        exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 1.2], gap="small")
        with exp_c1:
            out_format = st.selectbox("Export Format", list(EXPORT_FORMATS), key="exp-fmt", label_visibility="collapsed")
        with exp_c2:
            out_quality = st.slider("Quality", 50, 100, 92, key="exp-qual", disabled=out_format in ["PNG", "TIFF"], label_visibility="collapsed")
        with exp_c3:
            export_settings = {
                "format": out_format, "quality": out_quality, "long_edge": exp_long_edge, "sharpen": exp_sharpen,
                "metadata": exp_metadata, "remove_gps": exp_remove_gps,
                "watermark": {"text": wm_text, "position": wm_position, "opacity": wm_opacity, "font_size": wm_font_size} if wm_text.strip() else None,
            }
            export_extension, export_mime = EXPORT_FORMATS[out_format]
            # A full-resolution render on the backend is needed when the canvas shows a proxy, or
            # for AI Upscale (whose output this process may not even be able to decode) -- the
            # backend then encodes the file itself (app.export.encode_export via /process's
            # "export" option). Otherwise the already-rendered result is encoded right here.
            full_options = {key: value for key, value in options.items() if key != "preview_scale"}
            if exp_upscale:
                full_options.update(ai_upscale=True, ai_upscale_factor=exp_upscale)
            upscale_error = None
            if exp_upscale:
                try:
                    check_upscale_output(expected_output_size(source.size, options), exp_upscale)
                except ValueError as error:
                    upscale_error = str(error)
                    st.warning(f"⚠️ {error}")
            export_key = image_digest(
                json.dumps([source_digest, full_options, mask_source_key, mask_signature, own_signatures, removal_signature if removal_active else None, layers_key, export_settings, use_proxy], sort_keys=True, default=str).encode("utf-8")
            )
            if use_proxy or exp_upscale:
                export_producer = functools.partial(
                    _render_full_resolution_export, upload_bytes, upload_name, full_options, export_settings, mask_source,
                    mask_signature if shared_mask_layers else None, removal_signature if removal_active else None, layers_signature,
                    own_signatures, tuple(full_canvas),
                )
            else:
                export_producer = functools.partial(_encode_rendered_export, edited_image, export_settings, source_exif)

            st.download_button(
                label=f"💾 Save {out_format}",
                data=functools.partial(cached_export, export_key, export_producer),
                file_name=render_export_name(exp_name_template, Path(upload_name).stem, output_size, out_format, export_extension),
                mime=export_mime,
                use_container_width=True,
                key="exp-download-btn",
                shortcut="Ctrl+E",
                disabled=upscale_error is not None,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Export Multiple Sizes -- "Export As" for several named EXPORT_PROFILES (Web/Social/Print/
        # Archive) at once, distinct from the single-format download button above and from the
        # Batch Processor's "one recipe, many photos" (this is one photo, many output variants --
        # the more common design-student need: the same poster at a web size, a social-post size,
        # and a print-ready size, in one click). Reuses export_multiple()/EXPORT_PROFILES so the
        # actual per-size format/quality/max-size values are never a second copy of the single-
        # format export's own list above.
        with st.expander("📦 Export Multiple Sizes"):
            selected_profiles = st.multiselect("Sizes to include", list(EXPORT_PROFILES), default=["Web", "Social"], key="exp-multi-profiles")
            multi_remove_gps = st.checkbox(
                "Remove GPS location", value=True, key="exp-multi-remove-gps",
                help="Web and Social carry no EXIF at all; this also strips the location from Print and Archive.",
            )
            if st.button("Prepare ZIP", key="exp-multi-btn", disabled=not selected_profiles):
                multi_source = edited_image
                if use_proxy:
                    # Print/Archive are full-resolution: render it (lossless) on the backend first.
                    with st.spinner("Rendering at full resolution…"):
                        multi_source = Image.open(io.BytesIO(export_via_backend(
                            upload_bytes, upload_name, full_options | {"ai_upscale": False}, {"format": "PNG", "metadata": "all"},
                            _combined_mask_png(mask_source_key, mask_source, mask_signature) if shared_mask_layers else None,
                            _layers_payload(layers_signature, tuple(full_canvas)) if layers_signature else None,
                            _combined_mask_png(mask_source_key, mask_source, removal_signature) if removal_active else None,
                            **({"local_masks_bytes": _local_masks_payload([_combined_mask_png(mask_source_key, mask_source, signature) for signature in own_signatures])} if own_signatures else {}),
                        )))
                zip_bytes = export_multiple(
                    multi_source, selected_profiles, Path(upload_name).stem,
                    watermark_text=wm_text if wm_text.strip() else None,
                    watermark_position=wm_position,
                    watermark_opacity=wm_opacity,
                    watermark_font_size=wm_font_size,
                    exif=source_exif,
                    remove_gps=multi_remove_gps,
                )
                st.session_state["_exp_multi_zip"] = zip_bytes
                _request_rerun()
            if st.session_state.get("_exp_multi_zip"):
                st.download_button(
                    "⬇️ Download ZIP",
                    st.session_state["_exp_multi_zip"],
                    file_name=f"{Path(upload_name).stem}-exports.zip",
                    mime="application/zip",
                    key="exp-multi-dl-btn",
                )

# A tab body or canvas_col control above may have called _request_rerun() instead of st.rerun()
# directly (adding/removing a mask selection, Auto Enhance, committing the curve editor, importing
# presets, Undo/Redo/Reset, Library Catalog's Prev/Next/rating buttons -- see that function's
# docstring). Actually rerunning only here, at the very end of the script, now that every widget in
# every tab AND in canvas_col has been instantiated this pass, is what stops Streamlit from
# garbage-collecting a not-yet-rendered widget's session_state and silently resetting it to its
# `value=` default.
if st.session_state.pop("_rerun_pending", False):
    st.rerun()
