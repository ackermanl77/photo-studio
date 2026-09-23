from __future__ import annotations

import base64
import html
import io
import json
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

import cv2

from app.advanced_masks import MaskLayer, MaskStack, color_mask, gradient_mask, luminance_mask, magic_wand_mask, radial_mask, soften_mask
from app.backend import process_image
from app.backend_client import ensure_backend_running, process_via_backend
from app.batch_release import FileLike, export_multiple, make_zip, process_files
from app.custom_presets import PresetStore
from app.edit_history import EditHistory
from app.image_processor import PhotoProcessor
from app.layer_styles import apply_layer_styles
from app.layer_workspace import LayerWorkspace, WorkspaceLayer, layers_to_json, render_text_layer
from app.vector_shapes import SHAPE_TYPES, render_path_layer, render_shape_layer
from app.library import index_folder
from app.major_features import BLEND_MODE_LABELS, PhotoCatalog
from app.presets import PRESETS, apply_preset
from app.preset_exchange import export_presets, import_presets
from app.portrait_tools import detect_faces
from app.professional_color import HSL_BAND_ORDER, ICC_TRANSFORM_ERRORS, clipping_warning_overlay, estimate_white_balance, gamut_warning_overlay
from app.lens_corrections import LENS_PROFILES, apply_lens_corrections
from app.professional_tools import EXPORT_PROFILES, GRID_PRESETS, WATERMARK_POSITIONS, apply_mask_overlay, apply_watermark, compute_histogram, compute_waveform, draw_guides_and_grid, export_profile, grid_line_fractions, preview_image, zoom_preview
from app.project_bundle import ProjectBundle
from app.recipe_options import ASPECT_RATIOS, merge_curve_points, recipe_to_options

# A small bundled placeholder image (generated via PIL -- gradient background plus a few flat
# shapes, not a real photo) so a first-time visitor can explore every tool without needing to
# supply their own file first. "Try a sample photo" below loads it through the exact same
# restored_source_bytes/restored_source_name fallback path a reopened .photoproject uses.
SAMPLE_PHOTO_PATH = Path(__file__).resolve().parent / "assets" / "sample_photo.jpg"
SAMPLE_PHOTO_NAME = "sample_photo.jpg"

# Maps a JSON-serializable recipe key to the session_state key of the widget that reads/writes it.
# Shared by Undo/Redo (EditHistory snapshots) and Save/Load Project (ProjectBundle.recipe) so both
# features capture and restore exactly the same set of controls.
RECIPE_WIDGET_KEYS = {
    "exposure": "t-exp", "contrast": "t-cont", "highlights": "t-high", "shadows": "t-shad", "preset": "tone-preset",
    "temperature": "c-temp", "tint": "c-tint", "saturation": "c-sat", "vibrance": "c-vib",
    "curve_enabled": "c-curve-en", "curve_black": "c-p0", "curve_shadows": "c-p1", "curve_mid": "c-p2",
    "curve_highlights": "c-p3", "curve_white": "c-p4",
    "texture": "d-text", "clarity": "d-clar", "denoise": "d-den", "denoise_color": "d-den-color",
    "sharpen": "d-sharp", "dehaze": "d-haze",
    "effects_vignette_amount": "fx-vignette-amt", "effects_vignette_midpoint": "fx-vignette-mid",
    "effects_grain_amount": "fx-grain-amt",
    "portrait_smoothing": "port-sm",
    "mask_target_exposure": "m-texp", "mask_target_contrast": "m-tcont", "mask_target_saturation": "m-tsat",
    "crop_aspect": "crop-aspect", "removal_radius": "rm-radius",
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
RECIPE_DEFAULTS = {
    "exposure": 1.0, "contrast": 1.0, "highlights": 0, "shadows": 0, "preset": "None",
    "temperature": 0, "tint": 0, "saturation": 1.0, "vibrance": 0,
    "curve_enabled": False, "curve_black": 0, "curve_shadows": 64, "curve_mid": 128, "curve_highlights": 192, "curve_white": 255,
    "texture": 0, "clarity": 0, "denoise": 0, "denoise_color": 0, "sharpen": False, "dehaze": 0,
    "effects_vignette_amount": 0, "effects_vignette_midpoint": 50, "effects_grain_amount": 0,
    "portrait_smoothing": 0.0,
    "mask_target_exposure": 1.0, "mask_target_contrast": 1.0, "mask_target_saturation": 1.0,
    "crop_aspect": "Original", "removal_radius": 3,
    "crop_free_x0": 0.0, "crop_free_y0": 0.0, "crop_free_x1": 1.0, "crop_free_y1": 1.0,
    "lens_profile": "None", "lens_distortion": 0, "lens_vignette_amount": 0, "lens_vignette_midpoint": 50,
    "lens_chromatic_aberration": 0,
    "lens_perspective_v": 0, "lens_perspective_h": 0,
    "cg_shadows_hue": 0, "cg_shadows_intensity": 0,
    "cg_midtones_hue": 0, "cg_midtones_intensity": 0,
    "cg_highlights_hue": 0, "cg_highlights_intensity": 0,
    "cg_balance": 0,
}


def _hsl_widget_key(band: str, channel: str) -> str:
    """channel is one of "h"/"s"/"l" -- e.g. _hsl_widget_key("Orange", "s") -> "hsl-orange-s"."""
    return f"hsl-{band.lower()}-{channel}"


# Every band/channel combination participates in Undo/Redo/Save-Project exactly like every other
# control, via the same RECIPE_WIDGET_KEYS/RECIPE_DEFAULTS mechanism -- generated here instead of
# 24 hand-written entries since it's a direct product of HSL_BAND_ORDER x {hue, sat, lum}.
for _band in HSL_BAND_ORDER:
    for _channel, _suffix in (("hue", "h"), ("sat", "s"), ("lum", "l")):
        RECIPE_WIDGET_KEYS[f"hsl_{_band.lower()}_{_channel}"] = _hsl_widget_key(_band, _suffix)
        RECIPE_DEFAULTS[f"hsl_{_band.lower()}_{_channel}"] = 0


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
    """
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
        return radial_mask(source.size)
    if kind == "Painted":
        return _decode_painted_mask(params["mask_b64"])
    if kind == "Magic Wand":
        return magic_wand_mask(source, params["click_x"], params["click_y"], params["tolerance"], params.get("contiguous", True))
    return gradient_mask(source.size, "horizontal" if "Horizontal" in kind else "vertical")


def _render_selection_stack(state_key: str, key_prefix: str, source: Image.Image, include_opacity: bool = True) -> MaskStack:
    """Shared "add a named selection (Luminance/Color/gradient/Radial/Painted/Magic Wand), then
    list/edit/remove existing ones" UI used by both the Masks tab (local tone/color adjustments)
    and the Repair tab (object removal) -- same selection tools, a different session_state list
    and widget-key prefix per caller so the two tabs' widgets don't collide. Returns the combined
    MaskStack; the caller decides what to do with it (masked adjustments vs. inpainting).
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
            wand_html, wand_height = _wand_picker_html(preview_image(source, (480, 480)), key_prefix)
            components.html(wand_html, height=wand_height)
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
            paint_html, paint_height = _paint_editor_html(preview_image(source, (480, 480)), key_prefix)
            components.html(paint_html, height=paint_height)
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

    mask_stack = MaskStack(source.size)
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
            if layer["kind"] == "Painted":
                # The only selection kind with a spatial shape that can't be re-derived from a
                # handful of numeric params (Luminance/Color/gradient/Radial/Magic Wand all can,
                # from their existing name/enabled/invert/opacity/soften controls above) -- a
                # second pass at the canvas, not just parameter tweaks, needs its own toggle.
                if st.checkbox("🎨 Edit Painting", key=f"{key_prefix}-editpaint-toggle-{layer_id}"):
                    edit_prefix = f"{key_prefix}-ep-{layer_id}"
                    edit_html, edit_height = _paint_editor_html(
                        preview_image(source, (480, 480)), edit_prefix,
                        preload_mask_b64=layer["params"].get("mask_b64"), commit_label="💾 Update Painting",
                    )
                    components.html(edit_html, height=edit_height)
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
        base_mask = _build_mask_image(source, layer["kind"], layer["params"])
        if layer["soften"]:
            base_mask = soften_mask(base_mask, layer["soften"])
        mask_stack.add(MaskLayer(layer["name"], base_mask, layer["enabled"], layer["invert"], layer["opacity"]))

    return mask_stack


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


def _layer_dict_to_image(layer: dict, canvas_size: tuple[int, int]) -> Image.Image:
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
        content = render_text_layer(
            canvas_size, layer.get("text", ""), layer.get("font_size", 48), _hex_to_rgba(layer.get("color", "#FFFFFF")),
            layer.get("align", "left"), layer.get("x_pct", 50.0), layer.get("y_pct", 50.0),
        )
    elif layer.get("kind") == "shape":
        # fill_color/stroke_color are stored as st.color_picker hex strings, same UI convention as
        # a Text layer's "color" -- render_shape_layer() wants plain RGB int tuples.
        shape_params = {
            **layer,
            "fill_color": _hex_to_rgba(layer.get("fill_color", "#FFFFFF"))[:3],
            "stroke_color": _hex_to_rgba(layer.get("stroke_color", "#000000"))[:3],
        }
        content = render_shape_layer(canvas_size, shape_params)
    elif layer.get("kind") == "path":
        path_params = {
            **layer,
            "fill_color": _hex_to_rgba(layer.get("fill_color", "#FFFFFF"))[:3],
            "stroke_color": _hex_to_rgba(layer.get("stroke_color", "#FFFFFF"))[:3],
        }
        content = render_path_layer(canvas_size, path_params)
    else:
        content = Image.open(io.BytesIO(base64.b64decode(layer["image_b64"]))).convert("RGBA")
        if content.size != canvas_size:
            content = content.resize(canvas_size, Image.Resampling.LANCZOS)

    styles = layer.get("styles")
    if styles and any(styles.get(f"{prefix}_enabled") for prefix in _STYLE_EFFECT_PREFIXES):
        content = apply_layer_styles(content, _resolved_layer_styles(styles))
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
    editor_html, editor_height = _pen_editor_html(preview_image(source, (480, 480)), key_prefix, points, closed)
    components.html(editor_html, height=editor_height)
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


def _render_layer_stack(source: Image.Image) -> bytes | None:
    """Layers tab: any number of stacked Image or Text layers, each with its own name/visibility/
    opacity/blend mode, reorderable and duplicable -- the same "add, then list/edit/remove" shape
    _render_selection_stack established for Masks/Repair. The compositing engine underneath
    (app.layer_workspace.LayerWorkspace) already supported an arbitrary number of layers; only this
    panel was ever limited to exactly one uploaded overlay image with no text option at all. Returns
    the serialized payload for process_via_backend's `layers_bytes` argument
    (app.layer_workspace.layers_to_json's wire format), or None when the stack is empty.

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
                        "font_size": 48,
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
                        layer["font_size"] = st.slider("Font size (px)", 8, 400, int(layer.get("font_size", 48)), key=f"l-fsize-{layer_id}")
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
                        preview_size, layer["text"], layer["font_size"] * preview_ratio, _hex_to_rgba(layer["color"]),
                        layer["align"], layer["x_pct"], layer["y_pct"],
                    )
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
    workspace_layers = [
        WorkspaceLayer(layer["name"], _layer_dict_to_image(layer, source.size), layer["opacity"], layer["blend_mode"], layer["visible"])
        for layer in layers
    ]
    return layers_to_json(workspace_layers).encode("utf-8")


def _before_after_html(before: Image.Image, after: Image.Image) -> tuple[str, int]:
    """A draggable before/after comparison: a single range input drives a CSS clip-path on the
    "before" image layered over the "after" image. Pure inline HTML/CSS/JS via
    st.components.v1.html -- no component library dependency, consistent with keeping this app's
    UI free of extra dependencies after removing streamlit-drawable-canvas.

    `before` and `after` must already be the same pixel size (the caller resizes if a crop made
    them differ) -- that's what makes the overlay line up.
    """
    width, height = after.size
    display_height = max(220, min(560, int(700 * height / width))) if width else 320

    def _b64_png(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    before_b64 = _b64_png(before)
    after_b64 = _b64_png(after)
    html = f"""
    <div style="position:relative;width:100%;background:#090d16;border-radius:8px;overflow:hidden;line-height:0;">
      <img src="data:image/png;base64,{after_b64}" style="display:block;width:100%;height:auto;">
      <img id="ba-before" src="data:image/png;base64,{before_b64}"
           style="position:absolute;inset:0;width:100%;height:100%;object-fit:fill;clip-path:inset(0 50% 0 0);">
      <div id="ba-handle" style="position:absolute;top:0;bottom:0;left:50%;width:2px;margin-left:-1px;
           background:#fff;box-shadow:0 0 6px rgba(0,0,0,.7);pointer-events:none;"></div>
      <div style="position:absolute;top:8px;left:12px;color:#fff;font:600 11px system-ui;text-shadow:0 1px 3px rgba(0,0,0,.8);letter-spacing:.05em;">BEFORE</div>
      <div style="position:absolute;top:8px;right:12px;color:#fff;font:600 11px system-ui;text-shadow:0 1px 3px rgba(0,0,0,.8);letter-spacing:.05em;">AFTER</div>
      <input id="ba-range" type="range" min="0" max="100" value="50" aria-label="Before/after comparison position"
             style="position:absolute;left:8px;right:8px;bottom:10px;width:calc(100% - 16px);">
    </div>
    <script>
      const range = document.getElementById('ba-range');
      const before = document.getElementById('ba-before');
      const handle = document.getElementById('ba-handle');
      range.addEventListener('input', () => {{
        const v = range.value;
        before.style.clipPath = `inset(0 ${{100 - v}}% 0 0)`;
        handle.style.left = v + '%';
      }});
    </script>
    """
    return html, display_height


def _curve_editor_html(points: list[tuple[int, int]], bounds: list[tuple[int, int]], extra_points: list[tuple[int, int]] | None = None) -> str:
    """A draggable RGB tone curve: an inline SVG with five fixed points (Blacks/Shadows/Midtones/
    Highlights/Whites, matching the five sliders below it), each vertically draggable within its
    own [min, max], plus any number of user-added "extra" points -- click empty space on the curve
    to add one, drag it freely in both X and Y, double-click to remove it. `PhotoProcessor.
    apply_curves()` already accepted an arbitrary point list; the 5-fixed-point limit was purely a
    UI one (see "Known Issues" in docs/PROJECT_STATUS.md before this). Dragging/adding/removing
    only updates the curve's own SVG locally; each of those three actions commits on its own
    (point release, the add-click itself, the remove-double-click).

    Earlier version of this function wrote the dragged value directly into the real
    `<input type="range">` via a native-value-setter trick, then dispatched `input`/`change`.
    Confirmed live (by comparing the actual rendered image before/after, not just the visible
    slider label) that this does NOT reliably reach Streamlit's backend for range sliders in this
    Streamlit version -- the slider's own displayed number updates as a client-side echo, but the
    value the script actually sees on its next run can still be the old one, so a dragged curve
    could silently fail to affect the photo. Real OS-level key/mouse input (a genuine, trusted
    event) does commit reliably; a script-dispatched one does not, for this specific widget type
    (button clicks and a text_area's Ctrl+Enter were confirmed to work synthetically, just not a
    slider's input/change).

    So instead of touching the sliders at all, commit writes the whole curve as JSON into a hidden
    `st.text_area` (`st.container(key="curve-hidden")`, same Ctrl+Enter trick used by
    `_paint_editor_html`, confirmed reliable) using the RECIPE_WIDGET_KEYS names
    (curve_black/curve_shadows/curve_mid/curve_highlights/curve_white/curve_enabled) plus
    `curve_extra_points` (a plain list, handled like `mask_layers` rather than through
    RECIPE_WIDGET_KEYS -- see `_build_recipe`/`_restore_recipe_to_state`). Python reads it and
    stores it in `st.session_state["_pending_restore"]` -- the exact same "write before any widget
    instantiates, then rerun" mechanism Undo/Redo/Reset already use to update a slider's value from
    outside the widget itself, which sidesteps the whole reliability question since it never
    depends on a script-dispatched DOM event reaching a slider at all.
    """
    extra_points = extra_points or []
    margin_left, margin_top, plot_size = 34, 12, 210
    svg_w, svg_h = margin_left + plot_size + 12, margin_top + plot_size + 12

    def _to_px(x: float, y: float) -> tuple[float, float]:
        return (margin_left + (x / 255) * plot_size, margin_top + plot_size - (y / 255) * plot_size)

    all_points_sorted = sorted(list(points) + list(extra_points), key=lambda point: point[0])
    poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in (_to_px(x, y) for x, y in all_points_sorted))
    fixed_circles = "\n".join(
        f'<circle class="pt fixed" data-idx="{i}" cx="{px:.1f}" cy="{py:.1f}" r="7"></circle>'
        for i, (px, py) in enumerate(_to_px(x, y) for x, y in points)
    )
    diag_x0, diag_y0 = _to_px(0, 0)
    diag_x1, diag_y1 = _to_px(255, 255)
    points_json = json.dumps(points)
    bounds_json = json.dumps(bounds)
    extra_points_json = json.dumps(extra_points)
    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:6px 8px;">
    <svg id="curve-svg" width="100%" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}"
         preserveAspectRatio="xMinYMin meet" style="touch-action:none;display:block;">
      <rect id="curve-bg" x="{margin_left}" y="{margin_top}" width="{plot_size}" height="{plot_size}"
            fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.15)" style="cursor:copy;"/>
      <line x1="{diag_x0:.1f}" y1="{diag_y0:.1f}" x2="{diag_x1:.1f}" y2="{diag_y1:.1f}"
            stroke="rgba(255,255,255,0.2)" stroke-dasharray="3,3"/>
      <polyline id="curve-line" points="{poly}" fill="none" stroke="#fbbf24" stroke-width="2"/>
      {fixed_circles}
      <g id="extra-points"></g>
    </svg>
    </div>
    <div style="color:#94a3b8;font:11px system-ui;margin-top:4px;padding:0 2px;">
      Click an empty spot on the curve to add a point, drag it, double-click to remove it.
    </div>
    <style>
      #curve-svg .pt.fixed {{ fill:#fbbf24; stroke:#1e293b; stroke-width:2; cursor:ns-resize; }}
      #curve-svg .pt.fixed:hover {{ fill:#fde68a; }}
      #curve-svg .pt.extra {{ fill:#60a5fa; stroke:#1e293b; stroke-width:2; cursor:move; }}
      #curve-svg .pt.extra:hover {{ fill:#93c5fd; }}
    </style>
    <script>
      (function() {{
        const marginLeft = {margin_left}, marginTop = {margin_top}, plotSize = {plot_size}, svgW = {svg_w};
        const bounds = {bounds_json};
        const points = {points_json};
        let extraPoints = {extra_points_json};
        const svg = document.getElementById('curve-svg');
        const bg = document.getElementById('curve-bg');
        const line = document.getElementById('curve-line');
        const extraGroup = document.getElementById('extra-points');
        function toPx(x, y) {{ return [marginLeft + (x / 255) * plotSize, marginTop + plotSize - (y / 255) * plotSize]; }}
        function fromPy(py) {{ return Math.max(0, Math.min(255, ((marginTop + plotSize - py) / plotSize) * 255)); }}
        function fromPx(px) {{ return Math.max(0, Math.min(255, ((px - marginLeft) / plotSize) * 255)); }}
        function redraw() {{
          const all = points.concat(extraPoints).slice().sort((a, b) => a[0] - b[0]);
          line.setAttribute('points', all.map(p => toPx(p[0], p[1]).join(',')).join(' '));
          svg.querySelectorAll('.pt.fixed').forEach((c, i) => {{
            const [px, py] = toPx(points[i][0], points[i][1]);
            c.setAttribute('cx', px); c.setAttribute('cy', py);
          }});
          extraGroup.innerHTML = '';
          extraPoints.forEach((p, i) => {{
            const [px, py] = toPx(p[0], p[1]);
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('class', 'pt extra');
            circle.setAttribute('data-idx', i);
            circle.setAttribute('cx', px); circle.setAttribute('cy', py); circle.setAttribute('r', 7);
            extraGroup.appendChild(circle);
          }});
        }}
        function commit() {{
          try {{
            const payload = JSON.stringify({{
              curve_black: points[0][1], curve_shadows: points[1][1], curve_mid: points[2][1],
              curve_highlights: points[3][1], curve_white: points[4][1], curve_enabled: true,
              curve_extra_points: extraPoints,
            }});
            const container = window.parent.document.querySelector('.st-key-curve-hidden');
            if (!container) return;
            const textarea = container.querySelector('textarea');
            if (!textarea) return;
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            textarea.focus();
            setter.call(textarea, payload);
            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keyup', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.blur();
          }} catch (e) {{}}
        }}

        // Fixed points: Y-only drag within their own [min, max] -- unchanged from before extra
        // points existed. Their count never changes, so listeners attach once, directly.
        let dragIdx = null;
        svg.querySelectorAll('.pt.fixed').forEach((circle, i) => {{
          circle.addEventListener('pointerdown', (e) => {{
            dragIdx = i;
            circle.setPointerCapture(e.pointerId);
            e.preventDefault();
          }});
        }});
        svg.addEventListener('pointermove', (e) => {{
          if (dragIdx === null) return;
          const rect = svg.getBoundingClientRect();
          const scale = svgW / rect.width;
          const py = (e.clientY - rect.top) * scale;
          let value = Math.round(fromPy(py));
          const [lo, hi] = bounds[dragIdx];
          value = Math.max(lo, Math.min(hi, value));
          points[dragIdx][1] = value;
          redraw();
        }});
        function endDrag() {{
          if (dragIdx === null) return;
          dragIdx = null;
          commit();
        }}
        svg.addEventListener('pointerup', endDrag);
        svg.addEventListener('pointerleave', endDrag);

        // Extra points: added/removed dynamically, so handled via event delegation on `svg`
        // (checking e.target) instead of per-circle listeners that would need re-attaching every
        // time redraw() rebuilds the #extra-points group's children.
        let dragExtraIdx = null;
        svg.addEventListener('pointerdown', (e) => {{
          const el = e.target.closest('.pt.extra');
          if (!el) return;
          dragExtraIdx = Number(el.dataset.idx);
          svg.setPointerCapture(e.pointerId);
          e.preventDefault();
          e.stopPropagation();
        }});
        svg.addEventListener('pointermove', (e) => {{
          if (dragExtraIdx === null) return;
          const rect = svg.getBoundingClientRect();
          const scale = svgW / rect.width;
          const px = (e.clientX - rect.left) * scale;
          const py = (e.clientY - rect.top) * scale;
          extraPoints[dragExtraIdx] = [Math.round(fromPx(px)), Math.round(fromPy(py))];
          redraw();
        }});
        function endExtraDrag() {{
          if (dragExtraIdx === null) return;
          dragExtraIdx = null;
          commit();
        }}
        svg.addEventListener('pointerup', endExtraDrag);
        svg.addEventListener('pointerleave', endExtraDrag);
        svg.addEventListener('dblclick', (e) => {{
          const el = e.target.closest('.pt.extra');
          if (!el) return;
          extraPoints.splice(Number(el.dataset.idx), 1);
          redraw();
          commit();
        }});

        // Click empty background (not a drag-release from a point, which targets a circle
        // element instead of the rect) adds a new extra point at the clicked position.
        bg.addEventListener('click', (e) => {{
          const rect = svg.getBoundingClientRect();
          const scale = svgW / rect.width;
          const px = (e.clientX - rect.left) * scale;
          const py = (e.clientY - rect.top) * scale;
          extraPoints.push([Math.round(fromPx(px)), Math.round(fromPy(py))]);
          redraw();
          commit();
        }});

        redraw();
      }})();
    </script>
    """


def _paint_editor_html(background: Image.Image, key_prefix: str, preload_mask_b64: str | None = None, commit_label: str = "➕ Add to selections") -> tuple[str, int]:
    """An on-canvas selection editor: brush (paint/erase freehand), radial (drag out a circle),
    and gradient (drag out a direction), all painted onto one HTML canvas as solid white at
    varying alpha -- alpha *is* the mask value, so committing just base64-encodes the canvas as a
    PNG and that becomes a MaskLayer's `params["mask_b64"]` (decoded by `_decode_painted_mask`).
    Committing writes into the hidden `st.text_area` inside `st.container(key=f"{key_prefix}-
    paint-hidden")` (CSS-hidden -- see the app's <style> block) via the same native-value-setter
    trick as `_curve_editor_html`, so Python notices it on the next rerun (see the "Painted"
    branch in _render_selection_stack for a new selection, or the per-layer "Edit Painting"
    section for an existing one) -- there's no separate native Streamlit "Add" button to click
    after drawing, the in-canvas button *is* the commit. No component library dependency, same
    approach as the other editors in this file; avoids a repeat of the streamlit-drawable-canvas
    dead-weight episode.

    `preload_mask_b64`, when given (re-editing an existing "Painted" selection instead of
    starting a fresh one), is a base64 PNG at this same canvas resolution -- the exact bytes
    `_decode_painted_mask` would read back out -- drawn onto the canvas before any user input, so
    painting continues from what was already there instead of a blank slate. `commit_label` lets
    the re-edit path read "💾 Update Painting" instead of "➕ Add to selections" without a second,
    near-duplicate template.
    """
    width, height = background.size
    display_w = min(480, width)
    display_h = int(display_w * height / width) if width else 320

    buffer = io.BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    bg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    hidden_selector = f".st-key-{key_prefix}-paint-hidden"
    preload_js = (
        f"""const preload = new Image();
          preload.onload = () => ctx.drawImage(preload, 0, 0, W, H);
          preload.src = 'data:image/png;base64,{preload_mask_b64}';"""
        if preload_mask_b64
        else ""
    )

    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:8px;">
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px;">
        <button type="button" class="tool-btn active" data-tool="brush">🖌 Brush</button>
        <button type="button" class="tool-btn" data-tool="radial">⭕ Radial</button>
        <button type="button" class="tool-btn" data-tool="gradient">📐 Gradient</button>
        <button type="button" id="erase-btn" class="tool-btn">🧹 Erase</button>
        <button type="button" id="clear-btn" class="tool-btn">✖ Clear</button>
        <label style="display:flex;align-items:center;gap:4px;color:#cbd5e1;font:12px system-ui;margin-left:4px;">
          Size <input id="brush-size" type="range" min="4" max="80" value="18" style="width:80px;">
        </label>
      </div>
      <div style="position:relative;width:100%;max-width:{display_w}px;line-height:0;touch-action:none;">
        <img src="data:image/png;base64,{bg_b64}" style="display:block;width:100%;height:auto;border-radius:4px;">
        <canvas id="mask-canvas" width="{display_w}" height="{display_h}"
                style="position:absolute;inset:0;width:100%;height:100%;opacity:0.6;cursor:crosshair;"></canvas>
        <canvas id="preview-canvas" width="{display_w}" height="{display_h}"
                style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;"></canvas>
      </div>
      <button type="button" id="commit-btn" style="margin-top:8px;width:100%;padding:8px;border-radius:6px;
              border:none;background:#fbbf24;color:#1e293b;font:600 13px system-ui;cursor:pointer;">
        {commit_label}
      </button>
    </div>
    <style>
      .tool-btn {{ background:rgba(255,255,255,0.08); color:#cbd5e1; border:1px solid rgba(255,255,255,0.15);
                   border-radius:5px; padding:5px 9px; font:12px system-ui; cursor:pointer; }}
      .tool-btn.active {{ background:#fbbf24; color:#1e293b; border-color:#fbbf24; }}
      #erase-btn.active {{ background:#f87171; color:#1e293b; border-color:#f87171; }}
    </style>
    <script>
      (function() {{
        const maskCanvas = document.getElementById('mask-canvas');
        const previewCanvas = document.getElementById('preview-canvas');
        const ctx = maskCanvas.getContext('2d');
        const pctx = previewCanvas.getContext('2d');
        const W = maskCanvas.width, H = maskCanvas.height;
        let tool = 'brush', erasing = false, brushSize = 18;
        let drawing = false, lastX = 0, lastY = 0, startX = 0, startY = 0;
        {preload_js}

        document.querySelectorAll('.tool-btn[data-tool]').forEach((btn) => {{
          btn.addEventListener('click', () => {{
            tool = btn.dataset.tool;
            document.querySelectorAll('.tool-btn[data-tool]').forEach((b) => b.classList.toggle('active', b === btn));
          }});
        }});
        document.getElementById('erase-btn').addEventListener('click', (e) => {{
          erasing = !erasing;
          e.currentTarget.classList.toggle('active', erasing);
        }});
        document.getElementById('clear-btn').addEventListener('click', () => ctx.clearRect(0, 0, W, H));
        document.getElementById('brush-size').addEventListener('input', (e) => {{ brushSize = Number(e.target.value); }});

        function toCanvasXY(e) {{
          const rect = maskCanvas.getBoundingClientRect();
          return [(e.clientX - rect.left) * (W / rect.width), (e.clientY - rect.top) * (H / rect.height)];
        }}

        function paintDab(x, y) {{
          ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
          ctx.strokeStyle = 'white'; ctx.fillStyle = 'white';
          ctx.lineWidth = brushSize * 2; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
          ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(x, y); ctx.stroke();
          ctx.beginPath(); ctx.arc(x, y, brushSize, 0, Math.PI * 2); ctx.fill();
        }}

        function paintRadial(cx, cy, radius) {{
          ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
          const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(1, radius));
          grad.addColorStop(0, 'rgba(255,255,255,1)');
          grad.addColorStop(0.7, 'rgba(255,255,255,1)');
          grad.addColorStop(1, 'rgba(255,255,255,0)');
          ctx.fillStyle = grad;
          ctx.beginPath(); ctx.arc(cx, cy, Math.max(1, radius), 0, Math.PI * 2); ctx.fill();
        }}

        function paintGradient(x0, y0, x1, y1) {{
          ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
          const grad = ctx.createLinearGradient(x0, y0, x1, y1);
          grad.addColorStop(0, 'rgba(255,255,255,1)');
          grad.addColorStop(1, 'rgba(255,255,255,0)');
          ctx.fillStyle = grad;
          ctx.fillRect(0, 0, W, H);
        }}

        maskCanvas.addEventListener('pointerdown', (e) => {{
          drawing = true;
          maskCanvas.setPointerCapture(e.pointerId);
          [startX, startY] = toCanvasXY(e);
          lastX = startX; lastY = startY;
          if (tool === 'brush') paintDab(startX, startY);
          e.preventDefault();
        }});
        maskCanvas.addEventListener('pointermove', (e) => {{
          if (!drawing) return;
          const [x, y] = toCanvasXY(e);
          if (tool === 'brush') {{
            paintDab(x, y);
            lastX = x; lastY = y;
          }} else {{
            pctx.clearRect(0, 0, W, H);
            pctx.strokeStyle = 'rgba(255,255,255,0.9)'; pctx.lineWidth = 1.5; pctx.setLineDash([4, 3]);
            if (tool === 'radial') {{
              const radius = Math.hypot(x - startX, y - startY);
              pctx.beginPath(); pctx.arc(startX, startY, radius, 0, Math.PI * 2); pctx.stroke();
            }} else {{
              pctx.beginPath(); pctx.moveTo(startX, startY); pctx.lineTo(x, y); pctx.stroke();
            }}
          }}
        }});
        function endStroke(e) {{
          if (!drawing) return;
          drawing = false;
          pctx.clearRect(0, 0, W, H);
          if (tool === 'radial') {{
            const [x, y] = toCanvasXY(e);
            paintRadial(startX, startY, Math.hypot(x - startX, y - startY));
          }} else if (tool === 'gradient') {{
            const [x, y] = toCanvasXY(e);
            paintGradient(startX, startY, x, y);
          }}
        }}
        maskCanvas.addEventListener('pointerup', endStroke);
        maskCanvas.addEventListener('pointerleave', endStroke);

        document.getElementById('commit-btn').addEventListener('click', () => {{
          const base64 = maskCanvas.toDataURL('image/png').split(',')[1];
          try {{
            const container = window.parent.document.querySelector('{hidden_selector}');
            if (!container) return;
            const textarea = container.querySelector('textarea');
            if (!textarea) return;
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            textarea.focus();
            setter.call(textarea, base64);
            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            // Confirmed live: unlike a slider/checkbox (which commit on a plain 'change'),
            // st.text_area only commits -- and reruns the script -- on Ctrl+Enter or on losing
            // focus, and a synthetic .blur() alone wasn't reliably enough to trigger that in
            // this Streamlit version. Ctrl+Enter is Streamlit's own documented text_area
            // shortcut, so dispatch that directly rather than depend on blur semantics.
            textarea.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keyup', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.blur();
          }} catch (e) {{}}
        }});
      }})();
    </script>
    """, display_h + 130


def _wand_picker_html(background: Image.Image, key_prefix: str) -> tuple[str, int]:
    """Magic Wand's on-canvas "click to sample a color" picker: a single click on the preview
    reports its fractional (x, y) position back through the same hidden-`st.text_area` commit
    pattern `_paint_editor_html`/`_curve_editor_html` use (see `_curve_editor_html`'s docstring for
    why a script-dispatched event can't just touch a native widget directly). One click, no
    drawing state to track, so this is its own small component rather than another branch bolted
    onto `_paint_editor_html`.
    """
    width, height = background.size
    display_w = min(480, width)
    display_h = int(display_w * height / width) if width else 320

    buffer = io.BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    bg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    hidden_selector = f".st-key-{key_prefix}-wand-hidden"

    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:8px;">
      <div style="position:relative;width:100%;max-width:{display_w}px;line-height:0;">
        <img id="wand-img" src="data:image/png;base64,{bg_b64}"
             style="display:block;width:100%;height:auto;border-radius:4px;cursor:crosshair;">
        <div id="wand-marker" style="position:absolute;width:16px;height:16px;margin:-8px 0 0 -8px;
             border:2px solid #fbbf24;border-radius:50%;box-shadow:0 0 0 1px rgba(0,0,0,.6);
             display:none;pointer-events:none;"></div>
      </div>
      <div style="color:#94a3b8;font:12px system-ui;margin-top:6px;">Click anywhere on the photo to sample that color.</div>
    </div>
    <script>
      (function() {{
        const img = document.getElementById('wand-img');
        const marker = document.getElementById('wand-marker');
        img.addEventListener('click', function(e) {{
          const rect = img.getBoundingClientRect();
          const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
          const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
          marker.style.left = (fx * rect.width) + 'px';
          marker.style.top = (fy * rect.height) + 'px';
          marker.style.display = 'block';
          try {{
            const payload = JSON.stringify({{x: fx, y: fy}});
            const container = window.parent.document.querySelector('{hidden_selector}');
            if (!container) return;
            const textarea = container.querySelector('textarea');
            if (!textarea) return;
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            textarea.focus();
            setter.call(textarea, payload);
            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keyup', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.blur();
          }} catch (err) {{}}
        }});
      }})();
    </script>
    """, display_h + 60


def _crop_editor_html(
    background: Image.Image, key_prefix: str, box: tuple[float, float, float, float], full_size: tuple[int, int],
    snap_x: list[float] | None = None, snap_y: list[float] | None = None,
) -> tuple[str, int]:
    """A draggable free-form crop rectangle directly on the photo -- replaces the old pair of
    plain X/Y range sliders (see "Known Issues" in docs/PROJECT_STATUS.md, which explicitly called
    that a deliberate stand-in to avoid reintroducing a third-party UI component dependency after
    streamlit-drawable-canvas was removed as dead weight). No new dependency here either: two
    round corner-drag handles plus a move-by-dragging-the-body rectangle, all plain absolutely-
    positioned `<div>`s over the preview image, following the same "commit into a hidden
    st.text_area via the native-value-setter trick" pattern as `_paint_editor_html`/
    `_wand_picker_html`/`_curve_editor_html`.

    `box` is `(x0, y0, x1, y1)` as fractions of the image (0..1 each) -- deliberately not pixel
    coordinates, so the stored crop survives switching to a differently-sized photo (the exact gap
    the old range-slider version's Known Issues entry called out: its pixel-bounded slider values
    reset to the full frame on reload because they couldn't safely be interpreted against a new
    image's dimensions). `full_size` is the *source* image's real (width, height) -- distinct from
    `background`'s own (possibly downscaled-for-display) size -- purely so the on-canvas dimensions
    label reads real pixel counts rather than the preview's.

    `snap_x`/`snap_y` are the caller's active grid-line/guide fractions (see the Guides & Grid panel
    in canvas_col) -- only the two corner handles snap to them, not a body drag/move; snapping a
    translate against two independent edges per axis is a meaningfully bigger piece of geometry for
    comparatively little extra value over "frame this edge exactly on a guide," which the corners
    already cover, so it's left as a known, deliberate gap (see "Known Issues").
    """
    width, height = background.size
    display_w = min(480, width)
    display_h = int(display_w * height / width) if width else 320
    full_w, full_h = full_size
    x0, y0, x1, y1 = box

    buffer = io.BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    bg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    hidden_selector = f".st-key-{key_prefix}-hidden"

    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:8px;">
      <div id="crop-wrap" style="position:relative;width:100%;max-width:{display_w}px;line-height:0;touch-action:none;">
        <img src="data:image/png;base64,{bg_b64}" style="display:block;width:100%;height:auto;border-radius:4px;">
        <div id="crop-rect" style="position:absolute;border:2px solid #fbbf24;background:rgba(251,191,36,0.15);cursor:move;box-sizing:border-box;"></div>
        <div id="crop-tl" class="crop-handle" style="cursor:nwse-resize;"></div>
        <div id="crop-br" class="crop-handle" style="cursor:nwse-resize;"></div>
      </div>
      <div id="crop-dims" style="color:#94a3b8;font:12px system-ui;margin-top:6px;"></div>
    </div>
    <style>
      .crop-handle {{ position:absolute; width:16px; height:16px; margin:-8px 0 0 -8px;
                      background:#fbbf24; border:2px solid #1e293b; border-radius:50%; }}
    </style>
    <script>
      (function() {{
        const wrap = document.getElementById('crop-wrap');
        const rectEl = document.getElementById('crop-rect');
        const tl = document.getElementById('crop-tl');
        const br = document.getElementById('crop-br');
        const dimsLabel = document.getElementById('crop-dims');
        const box = {{x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}}};
        const snapX = {json.dumps(list(snap_x or []))};
        const snapY = {json.dumps(list(snap_y or []))};
        const SNAP_THRESHOLD = 0.012;
        let dragMode = null, moveStart = null;

        function snapValue(value, candidates) {{
          for (const candidate of candidates) {{
            if (Math.abs(value - candidate) < SNAP_THRESHOLD) return candidate;
          }}
          return value;
        }}

        function redraw() {{
          rectEl.style.left = (box.x0 * 100) + '%';
          rectEl.style.top = (box.y0 * 100) + '%';
          rectEl.style.width = ((box.x1 - box.x0) * 100) + '%';
          rectEl.style.height = ((box.y1 - box.y0) * 100) + '%';
          tl.style.left = (box.x0 * 100) + '%';
          tl.style.top = (box.y0 * 100) + '%';
          br.style.left = (box.x1 * 100) + '%';
          br.style.top = (box.y1 * 100) + '%';
          const pxW = Math.round((box.x1 - box.x0) * {full_w});
          const pxH = Math.round((box.y1 - box.y0) * {full_h});
          dimsLabel.textContent = 'Crop: ' + pxW + '×' + pxH + 'px -- drag a corner to resize, drag inside to move.';
        }}
        redraw();

        function fractionalPos(e) {{
          const rect = wrap.getBoundingClientRect();
          const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
          const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
          return [fx, fy];
        }}

        function startDrag(mode) {{
          return function(e) {{
            dragMode = mode;
            wrap.setPointerCapture(e.pointerId);
            if (mode === 'move') {{
              const [fx, fy] = fractionalPos(e);
              moveStart = {{fx: fx, fy: fy, x0: box.x0, y0: box.y0, x1: box.x1, y1: box.y1}};
            }}
            e.preventDefault();
            e.stopPropagation();
          }};
        }}
        tl.addEventListener('pointerdown', startDrag('tl'));
        br.addEventListener('pointerdown', startDrag('br'));
        rectEl.addEventListener('pointerdown', startDrag('move'));

        const MIN_SIZE = 0.02;
        wrap.addEventListener('pointermove', function(e) {{
          if (!dragMode) return;
          let [fx, fy] = fractionalPos(e);
          if (dragMode === 'tl' || dragMode === 'br') {{
            // Snap only the corner handles -- not a body drag/move, see the docstring above.
            fx = snapValue(fx, snapX);
            fy = snapValue(fy, snapY);
          }}
          if (dragMode === 'tl') {{
            box.x0 = Math.min(fx, box.x1 - MIN_SIZE);
            box.y0 = Math.min(fy, box.y1 - MIN_SIZE);
          }} else if (dragMode === 'br') {{
            box.x1 = Math.max(fx, box.x0 + MIN_SIZE);
            box.y1 = Math.max(fy, box.y0 + MIN_SIZE);
          }} else if (dragMode === 'move') {{
            const w = moveStart.x1 - moveStart.x0, h = moveStart.y1 - moveStart.y0;
            let nx0 = Math.max(0, Math.min(1 - w, moveStart.x0 + (fx - moveStart.fx)));
            let ny0 = Math.max(0, Math.min(1 - h, moveStart.y0 + (fy - moveStart.fy)));
            box.x0 = nx0; box.y0 = ny0; box.x1 = nx0 + w; box.y1 = ny0 + h;
          }}
          redraw();
        }});

        function commit() {{
          try {{
            const payload = JSON.stringify(box);
            const container = window.parent.document.querySelector('{hidden_selector}');
            if (!container) return;
            const textarea = container.querySelector('textarea');
            if (!textarea) return;
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            textarea.focus();
            setter.call(textarea, payload);
            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keyup', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.blur();
          }} catch (err) {{}}
        }}
        function endDrag() {{
          if (!dragMode) return;
          dragMode = null;
          commit();
        }}
        wrap.addEventListener('pointerup', endDrag);
        wrap.addEventListener('pointerleave', endDrag);
      }})();
    </script>
    """, display_h + 40


def _pen_editor_html(background: Image.Image, key_prefix: str, points: list[dict], closed: bool) -> tuple[str, int]:
    """The Pen tool: click to add anchor points on the photo, drag a point to move it, drag its
    amber handle to curve the path through that point (a symmetric "smooth point" handle -- see
    app.vector_shapes's module docstring for why it's not independent in/out handles), double-click
    a point to remove it. Same "commit into a hidden st.text_area via the native-value-setter
    trick" pattern as every other on-canvas editor in this file (_paint_editor_html/
    _wand_picker_html/_curve_editor_html/_crop_editor_html) -- confirmed the only reliable
    cross-iframe commit mechanism in this Streamlit version.

    Anchor/handle dots are plain absolutely-positioned `<div>`s (percentage left/top, exactly like
    _crop_editor_html's corner handles) rather than SVG circles inside a percentage-viewBox SVG --
    a 100x100 viewBox stretched with `preserveAspectRatio="none"` onto a non-square display box
    would visually squash circles into ellipses. The connecting path/handle-guide-lines don't need
    to be perfect circles, so they stay as an overlaid SVG using that same simple 0-100 coordinate
    space, avoiding any pixel/fraction conversion math for either layer.

    The "add a point" click listener is attached to the background `<img>` specifically, not the
    whole wrapper div -- anchor/handle dots are siblings layered on top of it (not descendants), so
    a click landing on one of them never bubbles into "add a new point at this spot" the way
    attaching the listener to a shared ancestor would.
    """
    width, height = background.size
    display_w = min(480, width)
    display_h = int(display_w * height / width) if width else 320

    buffer = io.BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    bg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    hidden_selector = f".st-key-{key_prefix}-hidden"
    points_json = json.dumps(points)
    closed_json = json.dumps(bool(closed))

    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:8px;">
      <div id="pen-wrap" style="position:relative;width:100%;max-width:{display_w}px;line-height:0;touch-action:none;">
        <img id="pen-bg" src="data:image/png;base64,{bg_b64}" style="display:block;width:100%;height:auto;border-radius:4px;">
        <svg id="pen-svg" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;" viewBox="0 0 100 100" preserveAspectRatio="none">
          <path id="pen-path" d="" fill="none" stroke="#60a5fa" stroke-width="0.6"></path>
          <g id="pen-handle-lines" stroke="rgba(251,191,36,0.7)" stroke-width="0.3"></g>
        </svg>
        <div id="pen-anchors"></div>
        <div id="pen-handles"></div>
      </div>
      <div style="color:#94a3b8;font:12px system-ui;margin-top:6px;">Click the photo to add a point. Drag a point to move it, drag its amber handle to curve the path. Double-click a point to remove it.</div>
    </div>
    <style>
      #pen-wrap .pen-anchor {{ position:absolute; width:14px; height:14px; margin:-7px 0 0 -7px;
                      background:#60a5fa; border:2px solid #1e293b; border-radius:50%; cursor:move; }}
      #pen-wrap .pen-handle {{ position:absolute; width:10px; height:10px; margin:-5px 0 0 -5px;
                      background:#fbbf24; border:2px solid #1e293b; border-radius:50%; cursor:crosshair; }}
    </style>
    <script>
      (function() {{
        const wrap = document.getElementById('pen-wrap');
        const bg = document.getElementById('pen-bg');
        const pathEl = document.getElementById('pen-path');
        const handleLines = document.getElementById('pen-handle-lines');
        const anchorLayer = document.getElementById('pen-anchors');
        const handleLayer = document.getElementById('pen-handles');
        let points = {points_json};
        const closed = {closed_json};
        let dragMode = null;  // {{type: 'anchor'|'handle', idx: number}}

        function fractionalPos(e) {{
          const rect = wrap.getBoundingClientRect();
          const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
          const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
          return [fx * 100, fy * 100];
        }}

        // Mirrors app.vector_shapes.path_outline_points()/_cubic_bezier_points() exactly -- the
        // preview drawn while editing and the layer actually rendered by the backend must trace
        // the same curve, or a point could look right here and land somewhere else once applied.
        function cubicPoints(p0, c1, c2, p1, steps) {{
          const pts = [];
          for (let i = 0; i <= steps; i++) {{
            const t = i / steps, mt = 1 - t;
            const x = mt*mt*mt*p0[0] + 3*mt*mt*t*c1[0] + 3*mt*t*t*c2[0] + t*t*t*p1[0];
            const y = mt*mt*mt*p0[1] + 3*mt*mt*t*c1[1] + 3*mt*t*t*c2[1] + t*t*t*p1[1];
            pts.push([x, y]);
          }}
          return pts;
        }}

        function outline() {{
          if (points.length < 2) return points.map(p => [p.x_pct, p.y_pct]);
          const anchors = points.map(p => [p.x_pct, p.y_pct]);
          const handles = points.map(p => [p.handle_dx_pct || 0, p.handle_dy_pct || 0]);
          const n = anchors.length;
          const segCount = closed ? n : n - 1;
          let out = [anchors[0]];
          for (let i = 0; i < segCount; i++) {{
            const p0 = anchors[i], p1 = anchors[(i + 1) % n];
            const h0 = handles[i], h1 = handles[(i + 1) % n];
            const c1 = [p0[0] + h0[0], p0[1] + h0[1]];
            const c2 = [p1[0] - h1[0], p1[1] - h1[1]];
            out = out.concat(cubicPoints(p0, c1, c2, p1, 20).slice(1));
          }}
          return out;
        }}

        function redraw() {{
          const out = outline();
          const d = out.length ? ('M ' + out.map(p => p[0].toFixed(2) + ' ' + p[1].toFixed(2)).join(' L ')) + (closed ? ' Z' : '') : '';
          pathEl.setAttribute('d', d);
          pathEl.setAttribute('fill', closed ? 'rgba(96,165,250,0.25)' : 'none');

          anchorLayer.innerHTML = '';
          handleLayer.innerHTML = '';
          handleLines.innerHTML = '';
          points.forEach((p, i) => {{
            const anchor = document.createElement('div');
            anchor.className = 'pen-anchor';
            anchor.style.left = p.x_pct + '%';
            anchor.style.top = p.y_pct + '%';
            anchor.dataset.idx = i;
            anchorLayer.appendChild(anchor);

            const hx = p.x_pct + (p.handle_dx_pct || 0);
            const hy = p.y_pct + (p.handle_dy_pct || 0);
            const handle = document.createElement('div');
            handle.className = 'pen-handle';
            handle.style.left = hx + '%';
            handle.style.top = hy + '%';
            handle.dataset.idx = i;
            handleLayer.appendChild(handle);

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', p.x_pct); line.setAttribute('y1', p.y_pct);
            line.setAttribute('x2', hx); line.setAttribute('y2', hy);
            handleLines.appendChild(line);
          }});
        }}
        redraw();

        function commit() {{
          try {{
            const payload = JSON.stringify(points);
            const container = window.parent.document.querySelector('{hidden_selector}');
            if (!container) return;
            const textarea = container.querySelector('textarea');
            if (!textarea) return;
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            textarea.focus();
            setter.call(textarea, payload);
            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keyup', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.blur();
          }} catch (err) {{}}
        }}

        anchorLayer.addEventListener('pointerdown', (e) => {{
          const el = e.target.closest('.pen-anchor');
          if (!el) return;
          dragMode = {{type: 'anchor', idx: Number(el.dataset.idx)}};
          wrap.setPointerCapture(e.pointerId);
          e.preventDefault();
          e.stopPropagation();
        }});
        handleLayer.addEventListener('pointerdown', (e) => {{
          const el = e.target.closest('.pen-handle');
          if (!el) return;
          dragMode = {{type: 'handle', idx: Number(el.dataset.idx)}};
          wrap.setPointerCapture(e.pointerId);
          e.preventDefault();
          e.stopPropagation();
        }});
        wrap.addEventListener('pointermove', (e) => {{
          if (!dragMode) return;
          const [fx, fy] = fractionalPos(e);
          const p = points[dragMode.idx];
          if (dragMode.type === 'anchor') {{
            p.x_pct = fx; p.y_pct = fy;
          }} else {{
            p.handle_dx_pct = fx - p.x_pct;
            p.handle_dy_pct = fy - p.y_pct;
          }}
          redraw();
        }});
        function endDrag() {{
          if (!dragMode) return;
          dragMode = null;
          commit();
        }}
        wrap.addEventListener('pointerup', endDrag);
        wrap.addEventListener('pointerleave', endDrag);

        anchorLayer.addEventListener('dblclick', (e) => {{
          const el = e.target.closest('.pen-anchor');
          if (!el) return;
          points.splice(Number(el.dataset.idx), 1);
          redraw();
          commit();
        }});

        // Attached to the background image itself, not `wrap` -- see the docstring above for why.
        bg.addEventListener('click', (e) => {{
          const [fx, fy] = fractionalPos(e);
          points.push({{x_pct: fx, y_pct: fy, handle_dx_pct: 0, handle_dy_pct: 0}});
          redraw();
          commit();
        }});
      }})();
    </script>
    """, display_h + 60


@st.cache_data(show_spinner=False)
def _preset_thumbnails(image_bytes: bytes, filename: str) -> dict[str, bytes]:
    """Small local preview renders for the Preset Look dropdown, so a user can see what each
    preset looks like before picking one instead of guessing from its name. Decorative only --
    generated in-process from a small thumbnail, separate from the real edit pipeline, which
    always goes through app/backend.py regardless of what's previewed here."""
    base = PhotoProcessor.from_bytes(image_bytes, filename).image
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


@st.cache_data(show_spinner=False)
def _custom_preset_thumbnail(image_bytes: bytes, filename: str, recipe: dict) -> bytes:
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
    base = PhotoProcessor.from_bytes(image_bytes, filename).image
    base.thumbnail((96, 96), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    base.save(buffer, format="PNG")
    try:
        preview = process_image(FileLike(filename, buffer.getvalue()), recipe_to_options(recipe))
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

# Global Ctrl+Z (undo) / Ctrl+Shift+Z / Ctrl+Y (redo) shortcuts for the canvas header's Undo/Redo
# buttons -- unlike the Library shortcuts above, canvas_col (and these buttons) render on every
# rerun regardless of which tab is selected, so this listener needs no "is my panel visible" guard
# and can be mounted once, unconditionally. Targets `.st-key-hist-undo-btn button` /
# `.st-key-hist-redo-btn button` -- confirmed live via the DOM that st.button(key=...) puts an
# "st-key-<key>" class on the enclosing stElementContainer div, the same convention already relied
# on for st.container(key=...) elsewhere in this file (e.g. "st-key-workspace-split"). Clicking a
# disabled <button> is a browser no-op, so this naturally does nothing when Undo/Redo has nothing
# to do -- no extra can_undo/can_redo check needed here. Skips real typing targets (text/number
# inputs, textareas) so Ctrl+Z still does normal text-undo there (e.g. the preset name field)
# instead of undoing a photo edit -- but NOT a slider: a real bug caught live here, not by the test
# suite (AppTest can't exercise a keydown at all -- see Known Issues) -- every st.slider is also an
# <input>, and it's the single most common place focus sits right when someone wants to hit Ctrl+Z
# (just after dragging one), so treating every INPUT as a typing target made the shortcut a no-op
# in the one moment it matters most.
_UNDO_REDO_KEYBOARD_SHORTCUTS_HTML = """
<script>
  (function() {
    const TEXT_ENTRY_INPUT_TYPES = new Set(['text', 'search', 'number', 'email', 'url', 'tel', 'password']);
    function handler(e) {
      const active = window.parent.document.activeElement;
      const tag = active && active.tagName;
      if (tag === 'TEXTAREA') return;
      if (tag === 'INPUT' && TEXT_ENTRY_INPUT_TYPES.has(active.type)) return;
      if (!(e.ctrlKey || e.metaKey)) return;
      let selector;
      if (e.key === 'z' || e.key === 'Z') {
        selector = e.shiftKey ? '.st-key-hist-redo-btn button' : '.st-key-hist-undo-btn button';
      } else if (e.key === 'y' || e.key === 'Y') {
        selector = '.st-key-hist-redo-btn button';
      } else {
        return;
      }
      const target = window.parent.document.querySelector(selector);
      if (!target) return;
      target.click();
      e.preventDefault();
    }
    if (!window.parent._photoStudioUndoRedoShortcutsBound) {
      window.parent.document.addEventListener('keydown', handler);
      window.parent._photoStudioUndoRedoShortcutsBound = true;
    }
  })();
</script>
"""


# Keys belonging to app/backend.py::process_image()'s Lens/Tone/Color/Detail stage (the "through
# Detail" half `_cached_process_through_detail` caches) -- every other key the `options = {...}`
# dict below builds belongs to the Repair-onward stage (`_cached_process_from_repair_onward`)
# instead. Kept as one explicit, auditable list rather than inferring the split some other way, so
# it's easy to check against process_image()'s own step-order comments when either changes.
_DETAIL_STAGE_OPTION_KEYS = (
    "lens_distortion", "lens_vignette_amount", "lens_vignette_midpoint", "lens_chromatic_aberration",
    "lens_perspective_v", "lens_perspective_h",
    "exposure", "contrast", "highlights", "shadows", "preset",
    "temperature", "tint", "saturation", "vibrance", "hsl_bands",
    "cg_shadows_hue", "cg_shadows_intensity", "cg_midtones_hue", "cg_midtones_intensity",
    "cg_highlights_hue", "cg_highlights_intensity", "cg_balance", "curve",
    "texture", "clarity", "dehaze", "denoise", "denoise_color", "sharpen",
)


@st.cache_data(show_spinner="🎨 Applying edits (tone/color/detail)…")
def _cached_process_through_detail(image_bytes: bytes, filename: str, detail_options: dict) -> bytes:
    """Stage 1 of 2 of the edit pipeline (Lens -> Tone -> Color -> Detail -- see
    app.backend.process_image()'s own docstring for the full order and the `stop_after`/
    `resume_from_detail` flags this two-stage split is built on). Cached independently of every
    Repair/Layers/Masks/Portrait/Crop/Effects option: resolves the Known Issue that the old single
    combined cache (keyed on the *entire* options dict) forced a full recompute -- including the
    genuinely expensive Non-Local Means denoise step -- every time *any* option changed, even a
    Mask slider that Denoise's own output doesn't depend on at all. Returns PNG bytes (not a
    decoded Image) so the cached value itself is what feeds `_cached_process_from_repair_onward`
    as its own `image_bytes` argument, unchanged."""
    options = dict(detail_options)
    options["stop_after"] = "detail"
    result = process_via_backend(image_bytes, filename, options)
    buffer = io.BytesIO()
    save_kwargs = {"format": "PNG"}
    if result.info.get("icc_profile"):
        save_kwargs["icc_profile"] = result.info["icc_profile"]
    result.save(buffer, **save_kwargs)
    return buffer.getvalue()


@st.cache_data(show_spinner="🎨 Applying edits…")
def _cached_process_from_repair_onward(
    stage_a_png_bytes: bytes, remaining_options: dict,
    local_mask_bytes: bytes | None, layers_bytes: bytes | None, removal_mask_bytes: bytes | None = None,
) -> Image.Image:
    """Stage 2 of 2 -- Repair through Effects, resuming from `stage_a_png_bytes`
    (`_cached_process_through_detail`'s own output, already processed through Detail) instead of
    the original upload, via `options["resume_from_detail"]`. Cached independently of every Lens/
    Tone/Color/Detail option, so nudging e.g. Exposure doesn't force this cheaper stage to redo
    Repair/Masks/Portrait/Crop/Effects either -- each stage only recomputes when its own inputs
    actually change. The spinner message is the only feedback the user gets that an edit is being
    sent over the network to app/backend.py -- without it, a slider drag on a large image can look
    like the app hung."""
    options = dict(remaining_options)
    options["resume_from_detail"] = True
    return process_via_backend(stage_a_png_bytes, "stage_a.png", options, local_mask_bytes, layers_bytes, removal_mask_bytes)


@st.cache_data(show_spinner=False)
def _lens_corrected_source(image_bytes: bytes, filename: str, distortion: float, vignette_amount: float, vignette_midpoint: float, chromatic_aberration: float, perspective_vertical: float = 0.0, perspective_horizontal: float = 0.0) -> Image.Image:
    """The base image the Crop preview and the Masks/Repair on-canvas selection tools paint
    against. app/backend.py::process_image() applies Lens corrections (a real cv2.undistort/remap
    geometric warp for distortion/chromatic aberration/perspective) before Crop or any mask, so a
    crop box or mask painted against the raw, uncorrected source would land on the wrong pixels
    once the backend actually warps the image -- worst near edges/corners, where distortion and
    perspective move content the most. Cached (and a no-op fast path when every lens value is
    still at its default) since this recomputes on every Streamlit rerun otherwise, not just when
    a Lens slider changes."""
    image = PhotoProcessor.from_bytes(image_bytes, filename).image
    if not (distortion or vignette_amount or chromatic_aberration or perspective_vertical or perspective_horizontal):
        return image
    return apply_lens_corrections(image, distortion, vignette_amount, vignette_midpoint, chromatic_aberration, perspective_vertical, perspective_horizontal)


@st.cache_data(show_spinner=False)
def _cached_gamut_warning_overlay(image_bytes: bytes, filename: str, icc_bytes: bytes):
    """gamut_warning_overlay() runs two full-image littlecms conversions -- cached so it only
    recomputes when the photo or its embedded profile actually changes, not on every Streamlit
    rerun triggered by an unrelated control while the "Show out-of-gamut pixels" checkbox stays
    on."""
    image = PhotoProcessor.from_bytes(image_bytes, filename).image
    return gamut_warning_overlay(image, icc_bytes)


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
.st-key-curve-hidden, .st-key-crop-editor-hidden,
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

# Apply any pending Undo/Redo or Load-Project restoration. This MUST run before any tab widget
# below is created (see _restore_recipe_to_state's docstring) -- it only ever fires on the rerun
# immediately triggered by clicking Undo, Redo, or loading a project.
if "_pending_restore" in st.session_state:
    _restore_recipe_to_state(st.session_state.pop("_pending_restore"))

# File loading section
upload_col1, upload_col2 = st.columns([2.5, 1], gap="small")
with upload_col1:
    uploaded = st.file_uploader(
        "Open Image",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff", "cr2", "cr3", "nef", "arw", "dng"],
        help="Open image (JPG, PNG, WEBP, TIFF, or Camera RAW). Original is never overwritten.",
        label_visibility="collapsed",
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
        # must not crash the whole app with a raw traceback -- every other user-input path in
        # this file (RAW load, backend calls, curve JSON) already degrades to a friendly error.
        try:
            loaded_project = ProjectBundle.load(project_upload.getvalue())
            pending_recipe = dict(loaded_project.recipe)
            workspace = loaded_project.layer_workspace_object()
            restored_layers = []
            if workspace:
                # workspace.layers[0] is always the LayerWorkspace-internal "Background" (the
                # source image itself, already captured separately as restored_source_bytes) --
                # everything from index 1 on is a real overlay layer the Layers tab added. A
                # project saved before this multi-layer panel existed only ever had exactly one
                # (workspace.layers[1]), so this naturally still restores those old files correctly
                # as a one-item list.
                for overlay_layer in workspace.layers[1:]:
                    overlay_buffer = io.BytesIO()
                    overlay_layer.image.save(overlay_buffer, format="PNG")
                    restored_layers.append(
                        {
                            "id": uuid.uuid4().hex,
                            "kind": "image",  # a saved project only ever stores flattened pixels, even for what was a Text layer -- see the Save Project handler
                            "name": overlay_layer.name,
                            "image_b64": base64.b64encode(overlay_buffer.getvalue()).decode("ascii"),
                            "opacity": overlay_layer.opacity,
                            "blend_mode": overlay_layer.blend_mode,
                            "visible": overlay_layer.visible,
                        }
                    )
        except (ValueError, KeyError, OSError) as error:
            st.error(f"⚠️ Could not open this project file: {error}")
        else:
            st.session_state["restored_source_bytes"] = loaded_project.source_bytes
            st.session_state["restored_source_name"] = loaded_project.source_name
            pending_recipe["layers"] = restored_layers
            st.session_state["_pending_restore"] = pending_recipe
            st.rerun()

if not uploaded and "restored_source_bytes" not in st.session_state:
    st.info("👆 Upload an image above, load a saved project, or click \"Try a sample photo\" to open the editing workspace.")
    st.stop()

# A freshly uploaded file always takes priority; otherwise fall back to the image bundled inside a
# loaded .photoproject so re-opening a project doesn't also require re-uploading its source image.
upload_name = uploaded.name if uploaded else st.session_state["restored_source_name"]
upload_bytes = uploaded.getvalue() if uploaded else st.session_state["restored_source_bytes"]

# Load image safely. PhotoProcessor.from_bytes() routes camera RAW files (.cr2/.cr3/.nef/.arw/.dng)
# through rawpy via app.raw_import, and falls back to a plain PIL open for every other format --
# using it here (instead of Image.open() directly) is what makes the RAW types advertised in the
# uploader above actually work instead of failing with an unreadable-file error.
try:
    processor = PhotoProcessor.from_bytes(upload_bytes, upload_name)
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
source = processor.image

# The actual edit pipeline runs on app/backend.py, not in this process (see docs/PROJECT_STATUS.md);
# `processor`/`source` above are only used locally for decoding, mask geometry, and previews.
try:
    ensure_backend_running()
except RuntimeError as error:
    st.error(f"⚠️ {error}")
    st.stop()

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
        tab_tone, tab_lens, tab_crop, tab_color, tab_detail, tab_repair, tab_layers, tab_masks, tab_portrait, tab_presets, tab_more = st.tabs([
            "☼ Tone", "◎ Lens", "⛶ Crop", "🎨 Color", "🔬 Detail", "🩹 Repair", "◈ Layers", "◉ Masks", "◌ Portrait", "✦ Presets", "☰ Batch & Library"
        ])

        show_original = False
        compare_mode = False
        zoom_level = 1.0
        pan_x = 0.5
        pan_y = 0.5
        edited_image = source
        layers_bytes = None
        aspect_code = "Original"
        crop_box = None

        # TAB 1: TONE & EXPOSURE
        with tab_tone:
            # Mirrors PhotoProcessor.auto_enhance() (app/image_processor.py) exactly -- that method
            # is otherwise only reachable from the CLI's --auto-enhance flag. Writing straight into
            # _pending_restore reuses the same "one rerun to apply" mechanism as Undo/Redo/Reset,
            # and only touches these five recipe fields, leaving everything else (crop, masks,
            # curve, temperature, ...) exactly as the user left it.
            if st.button("✨ Auto Enhance", key="auto-enhance-btn", help="Apply a quick, balanced exposure/contrast/saturation/denoise/sharpen pass", use_container_width=True):
                st.session_state["_pending_restore"] = {
                    "exposure": 1.12, "contrast": 1.15, "saturation": 1.08, "denoise": 3, "sharpen": True,
                }
                _request_rerun()
            preset_names = ["None", *sorted(PRESETS)]
            preset_thumbs = _preset_thumbnails(upload_bytes, upload_name)
            thumb_cols = st.columns(len(preset_names))
            for thumb_col, name in zip(thumb_cols, preset_names):
                with thumb_col:
                    st.image(preset_thumbs[name], use_container_width=True)
                    st.caption(name)
            preset_choice = st.selectbox("Preset Look", preset_names, key="tone-preset")
            exp_val = st.slider("Exposure", 0.5, 1.5, 1.0, 0.01, key="t-exp", help="Overall brightness multiplier -- below 1.0 darkens the image, above 1.0 brightens it.")
            cont_val = st.slider("Contrast", 0.5, 1.5, 1.0, 0.01, key="t-cont", help="Spreads tones apart around midtone gray -- higher makes shadows darker and highlights brighter.")
            high_val = st.slider("Highlights", -100, 100, 0, key="t-high", help="Recovers or boosts the brightest parts of the image -- negative pulls blown highlights back, positive brightens them further.")
            shad_val = st.slider("Shadows", -100, 100, 0, key="t-shad", help="Lifts or deepens the darkest parts of the image -- positive opens up shadow detail, negative deepens shadows.")

        # TAB: LENS CORRECTIONS
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

        # The Crop preview and the Masks/Repair selection tools below all paint against this,
        # not the raw `source` -- see _lens_corrected_source()'s docstring for why: the backend
        # applies Lens corrections before Crop/Masks/Repair, so those tools need to agree with it
        # on which pixel is where.
        mask_source = _lens_corrected_source(
            upload_bytes, upload_name, lens_distortion, lens_vignette_amount, lens_vignette_midpoint, lens_chromatic_aberration,
            lens_perspective_v, lens_perspective_h,
        )

        # TAB: CROP & ASPECT RATIO -- placed right after Tone for workflow convenience (frame the
        # shot before fine-tuning tone/color); still applied last in the actual pipeline, see the
        # note above the tabs() call.
        with tab_crop:
            st.markdown("**Crop / Aspect Ratio**")
            crop_choice = st.selectbox("Aspect Ratio", ["Original", "Free-form", *ASPECT_RATIOS], key="crop-aspect")
            if crop_choice == "Free-form":
                # Fractional (0..1) storage, not pixel coordinates -- unlike the old pair of plain
                # X/Y range sliders this replaced (bounded 0..src_w/src_h, so the stored box reset
                # to the full frame whenever a differently-sized photo loaded -- see "Known Issues"
                # in docs/PROJECT_STATUS.md), a fraction of the frame means the same box scales
                # correctly to any image size, so it now genuinely persists through Undo/Redo and
                # Save/Load Project via RECIPE_WIDGET_KEYS/RECIPE_DEFAULTS like every other control.
                crop_box_fractional = (
                    st.session_state.get("crop-free-x0", 0.0), st.session_state.get("crop-free-y0", 0.0),
                    st.session_state.get("crop-free-x1", 1.0), st.session_state.get("crop-free-y1", 1.0),
                )
                # Read (not written) here -- the Guides & Grid panel lives in canvas_col, which
                # renders later in the script; session_state still carries last rerun's value, and
                # on the very first-ever run these default to "no grid, no guides" (see the comment
                # above that panel). Drawn onto the crop preview's own background and fed to the
                # editor as snap targets, so the corner handles below can snap to the same lines the
                # user sees drawn on the photo.
                crop_grid_type = st.session_state.get("grid-type", "None")
                crop_guides = st.session_state.get("guides", [])
                snap_x = list(grid_line_fractions(crop_grid_type)) + [g["position_pct"] / 100.0 for g in crop_guides if g.get("orientation") == "vertical"]
                snap_y = list(grid_line_fractions(crop_grid_type)) + [g["position_pct"] / 100.0 for g in crop_guides if g.get("orientation") == "horizontal"]
                crop_preview_bg = preview_image(mask_source, (480, 480))
                if crop_grid_type != "None" or crop_guides:
                    crop_preview_bg = draw_guides_and_grid(crop_preview_bg, crop_grid_type, crop_guides)
                editor_html, editor_height = _crop_editor_html(
                    crop_preview_bg, "crop-editor", crop_box_fractional, mask_source.size, snap_x=snap_x, snap_y=snap_y,
                )
                components.html(editor_html, height=editor_height)
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
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    crop_box = None
                    st.warning("Select a non-empty region.")
            elif crop_choice != "Original":
                crop_w_ratio, crop_h_ratio = ASPECT_RATIOS[crop_choice]
                st.caption(f"Center-cropping to {crop_choice} ({crop_w_ratio}:{crop_h_ratio}).")
                aspect_code = f"{crop_w_ratio}:{crop_h_ratio}"
            else:
                st.caption("No crop applied — showing the full frame.")

        # TAB 2: COLOR & CURVES
        with tab_color:
            temp_val = st.slider("Temperature (Cool/Warm)", -100, 100, 0, key="c-temp", help="Shifts the overall white balance -- negative cools the image toward blue, positive warms it toward orange.")
            tint_val = st.slider("Tint (Green/Magenta)", -100, 100, 0, key="c-tint", help="Corrects a color cast along the green/magenta axis, independent of Temperature.")
            sat_val = st.slider("Saturation", 0.0, 2.0, 1.0, 0.01, key="c-sat", help="Uniformly boosts or reduces color intensity across the whole image -- 0 is grayscale, 1.0 is unchanged.")
            vib_val = st.slider(
                "Vibrance", -100, 100, 0, key="c-vib",
                help="A smarter saturation boost -- protects skin tones and already-saturated colors, pushing washed-out ones harder instead.",
            )

            with st.expander("🎯 Pick White Balance", expanded=False):
                st.caption("Click a spot in the photo that should be neutral gray or white -- Temperature and Tint above are set to whatever would neutralize it. A starting point to fine-tune from, not a guaranteed perfect result.")
                wb_html, wb_height = _wand_picker_html(preview_image(mask_source, (480, 480)), "wb")
                components.html(wb_html, height=wb_height)
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
                components.html(
                    _curve_editor_html(preview_points, curve_bounds, st.session_state["curve_extra_points"]), height=270
                )
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
                                _cached_gamut_warning_overlay(upload_bytes, upload_name, source_icc_bytes),
                                caption="Magenta = clipped when converted to this app's sRGB working space",
                                use_container_width=True,
                            )
                        except ICC_TRANSFORM_ERRORS:
                            st.caption("Could not read this photo's embedded color profile.")

        # TAB 3: DETAIL & SHARPNESS
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

        # TAB: REPAIR (OBJECT REMOVAL)
        with tab_repair:
            st.markdown("**Remove Objects**")
            st.caption("Select the blemish or object to remove with the same selection tools as Masks (below). The selected area is painted over by blending in nearby texture.")

            removal_stack = _render_selection_stack("removal_mask_layers", "rm", mask_source, include_opacity=False)
            removal_combined_mask = removal_stack.combined()
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
                st.image(apply_mask_overlay(mask_source, removal_combined_mask, color=(255, 200, 0)), caption="Yellow = will be removed", use_container_width=True)

        # TAB 4: LAYERS & COMPOSITING
        with tab_layers:
            st.markdown("**Layers**")
            st.caption("Add any number of image layers, each with its own opacity and blend mode -- composited on the processing backend when the image renders.")
            layers_bytes = _render_layer_stack(source)

        # TAB 5: MASKS & LOCAL ADJUSTMENTS
        with tab_masks:
            st.markdown("**Mask Layers**")
            st.caption("Build several named selections and combine them into one mask (union). The target adjustments below apply only where the combined mask is active.")

            mask_stack = _render_selection_stack("mask_layers", "m", mask_source)
            # Union of every enabled layer (or an all-zero, no-op mask when there are none).
            combined_mask = mask_stack.combined()

            if not st.session_state.mask_layers:
                st.caption("No mask layers yet — the sliders below have no effect until you add one.")
            elif not any(layer["enabled"] for layer in st.session_state.mask_layers):
                st.caption("All mask layers are disabled — the sliders below have no effect.")

            if st.checkbox("👁️ Preview combined mask overlay", key="m-preview-overlay") and st.session_state.mask_layers:
                # Overlaid on mask_source (lens-corrected, but otherwise untouched) -- the
                # backend-processed preview isn't available locally, so this debug view shows the
                # selection rather than the live edited result.
                st.image(apply_mask_overlay(mask_source, combined_mask), caption="Red = selected by the combined mask", use_container_width=True)

            mask_inert = not any(layer["enabled"] for layer in st.session_state.mask_layers)
            m_exp = st.slider("Target Exposure", 0.5, 1.5, 1.0, 0.01, key="m-texp", disabled=mask_inert, help="Exposure adjustment applied only inside the combined mask's selected area.")
            m_cont = st.slider("Target Contrast", 0.5, 1.5, 1.0, 0.01, key="m-tcont", disabled=mask_inert, help="Contrast adjustment applied only inside the combined mask's selected area.")
            m_sat = st.slider("Target Saturation", 0.0, 2.0, 1.0, 0.01, key="m-tsat", disabled=mask_inert, help="Saturation adjustment applied only inside the combined mask's selected area.")

        # TAB 6: PORTRAIT RETOUCHING
        with tab_portrait:
            st.markdown("**Face & Skin Retouching**")
            # Defaults to 0 (no change) so skin smoothing is opt-in, not silently forced onto every
            # export just because this tab exists in the workspace.
            p_smooth = st.slider(
                "Skin Smoothing", 0.0, 1.0, 0.0, 0.01, key="port-sm",
                help="0 leaves skin untouched. Increase to smooth detected skin tones.",
            )
            detected_faces = detect_faces(source)
            st.caption(f"👤 Detected faces: {len(detected_faces)}")

        # TAB 7: PRESETS
        with tab_presets:
            st.markdown("**Preset Library**")
            p_store = PresetStore()
            p_names = p_store.names()
            st.caption(f"Saved: {', '.join(p_names) if p_names else 'None'}")

            # Load/apply/delete a saved custom preset -- p_store.get()/.delete() already existed
            # and were tested, but nothing in the UI ever called them: a preset could be saved,
            # exported, and imported, but never actually reapplied to a photo. Routed through the
            # same "_pending_restore" + _request_rerun() mechanism as Undo/Redo/Load Project, since
            # a preset is just a recipe like any other.
            if p_names:
                st.markdown("**Load a Preset**")
                p_load_name = st.selectbox("Preset to load", p_names, key="p-load-sel", label_visibility="collapsed")
                st.image(
                    _custom_preset_thumbnail(upload_bytes, upload_name, p_store.get(p_load_name)),
                    width=96, caption=p_load_name,
                )
                p_load_col, p_delete_col = st.columns(2)
                if p_load_col.button("📂 Apply", key="p-load-btn", use_container_width=True):
                    st.session_state["_pending_restore"] = p_store.get(p_load_name)
                    st.toast(f"Applied preset '{p_load_name}'", icon="📂")
                    _request_rerun()
                if p_delete_col.button("🗑️ Delete", key="p-delete-btn", use_container_width=True):
                    p_store.delete(p_load_name)
                    st.toast(f"Deleted preset '{p_load_name}'", icon="🗑️")
                    _request_rerun()

            p_new_name = st.text_input("New Preset Name", key="p-name-in")
            if st.button("Save Current Look as Preset", key="p-save-btn"):
                if p_new_name.strip():
                    # The full recipe (every Tone/Lens/Color/Detail/Masks/... control, plus mask
                    # layers), not just 5 basic fields -- a preset saved after building a mask or a
                    # lens correction used to silently drop all of that. Deliberately excludes the
                    # layer overlay image, same as Undo/Redo history (see _build_recipe's
                    # docstring): a preset is a reusable style, not tied to one photo's overlay.
                    p_store.save(p_new_name.strip(), _build_recipe())
                    st.toast(f"Preset '{p_new_name.strip()}' saved", icon="💾")
                    _request_rerun()
                else:
                    st.toast("Type a name before saving a preset", icon="⚠️")

            st.markdown("---")
            st.markdown("**Export / Import Presets**")
            if p_store.names():
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
            if st.button("💾 Save Project", key="proj-save-btn"):
                # Layer images are already decoded and validated when added in the Layers tab (see
                # _render_layer_stack) -- a corrupted upload never makes it into image_layers, so
                # there's nothing left to fail here the way the old single-uploader version could.
                # Text layers are flattened to a plain image here too, via the same
                # _layer_dict_to_image() _render_layer_stack() itself uses -- a saved project keeps
                # the text's appearance, not its still-editable text/font/color/position (those live
                # only in this session's image_layers); reopening a project shows it as an Image
                # layer instead of a Text one. See docs/PROJECT_STATUS.md's "Known Issues".
                layer_workspace_json = None
                if st.session_state.get("image_layers"):
                    workspace = LayerWorkspace(source)
                    for layer in st.session_state.image_layers:
                        overlay = _layer_dict_to_image(layer, source.size)
                        workspace.add(WorkspaceLayer(layer["name"], overlay, layer["opacity"], layer["blend_mode"], layer["visible"]))
                    layer_workspace_json = workspace.to_json()
                project_recipe = _build_recipe()
                project = ProjectBundle(source_name=upload_name, source_bytes=upload_bytes, recipe=project_recipe, layer_workspace=layer_workspace_json)
                proj_buf = io.BytesIO()
                project.save(proj_buf)
                st.toast("Project packaged — click Download to save it", icon="💾")
                st.download_button(
                    "⬇️ Download .photoproject",
                    proj_buf.getvalue(),
                    file_name=f"{Path(upload_name).stem}.photoproject",
                    mime="application/zip",
                    key="proj-dl-btn",
                )

        # TAB 8: MORE (Catalog, Batch, System)
        with tab_more:
            more_mode = st.radio("Tool", ["Library Catalog", "Batch Processor", "System Health"], horizontal=True, label_visibility="collapsed", key="more-mode")
            if more_mode == "Library Catalog":
                lib_folder = st.text_input("Folder", str(Path.home() / "Pictures"), key="lib-f-in")
                lib_db = Path.home() / ".photo-studio" / "catalog.sqlite"
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
                rec = st.session_state.get("records", [])
                q = st.text_input("Search", key="lib-q-in")
                r_min = st.slider("Min Stars", 0, 5, 0, key="lib-r-in")
                search_results = PhotoCatalog(lib_db).search(q, r_min) if rec else []
                st.dataframe(search_results, use_container_width=True)
                if rec and not search_results:
                    st.caption("No photos match this search / star filter.")

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
                        st.image(current_path, width=240)
                    except Exception:
                        st.caption("Preview unavailable.")

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
                b_uploaded = st.file_uploader("Batch Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="b-up")
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
                        _custom_preset_thumbnail(upload_bytes, upload_name, p_store.get(b_preset_choice[: -len(" (custom)")])),
                        width=96,
                    )
                if b_uploaded and st.button("Run Batch", key="b-run-btn"):
                    b_root = Path.home() / ".photo-studio" / "batch-input"
                    b_root.mkdir(parents=True, exist_ok=True)
                    b_paths = []
                    for f in b_uploaded:
                        t = b_root / f.name
                        t.write_bytes(f.getvalue())
                        b_paths.append(t)
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

                    outs, rep = process_files(b_paths, b_preset, b_prof, progress=_update_batch_progress, recipe=b_recipe)
                    b_progress_bar.progress(1.0)
                    b_progress_status.caption(f"Done — {sum(1 for x in rep if x.status == 'ok')}/{len(rep)} succeeded")
                    st.dataframe([{"source": x.source.name, "status": x.status, "error": x.error or ""} for x in rep], use_container_width=True)
                    st.download_button("Download ZIP", make_zip(outs), "batch.zip", "application/zip", key="b-dl")
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
            "lens_distortion": lens_distortion,
            "lens_vignette_amount": lens_vignette_amount,
            "lens_vignette_midpoint": lens_vignette_midpoint,
            "lens_chromatic_aberration": lens_chromatic_aberration,
            "lens_perspective_v": lens_perspective_v,
            "lens_perspective_h": lens_perspective_h,
            "exposure": exp_val,
            "contrast": cont_val,
            "highlights": high_val,
            "shadows": shad_val,
            "preset": preset_choice,
            "temperature": temp_val,
            "tint": tint_val,
            "saturation": sat_val,
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
            "mask_exposure": m_exp,
            "mask_contrast": m_cont,
            "mask_saturation": m_sat,
            "portrait_smoothing": p_smooth,
            "aspect": aspect_code,
            "crop_box": list(crop_box) if crop_box else None,
        }

        local_mask_bytes = None
        if st.session_state.mask_layers:
            mask_buffer = io.BytesIO()
            combined_mask.save(mask_buffer, format="PNG")
            local_mask_bytes = mask_buffer.getvalue()

        removal_mask_bytes = None
        if removal_active:
            removal_buffer = io.BytesIO()
            removal_combined_mask.save(removal_buffer, format="PNG")
            removal_mask_bytes = removal_buffer.getvalue()

        try:
            detail_options = {key: options[key] for key in _DETAIL_STAGE_OPTION_KEYS}
            remaining_options = {key: value for key, value in options.items() if key not in _DETAIL_STAGE_OPTION_KEYS}
            stage_a_png_bytes = _cached_process_through_detail(upload_bytes, upload_name, detail_options)
            edited_image = _cached_process_from_repair_onward(
                stage_a_png_bytes, remaining_options, local_mask_bytes, layers_bytes, removal_mask_bytes
            )
        except RuntimeError as error:
            st.error(f"⚠️ {error}")
            st.stop()

    with canvas_col:
        # Canvas Header Bar
        top_c1, top_undo, top_redo, top_reset, top_snap, top_c2, top_c3 = st.columns(
            [0.85, 0.18, 0.18, 0.18, 0.22, 0.65, 0.95], gap="small"
        )
        with top_c1:
            # html.escape() here: upload_name is the uploaded file's own filename (user-controlled),
            # interpolated into both an HTML attribute (title=) and element text below via
            # unsafe_allow_html -- unescaped, a filename containing a quote or angle bracket could
            # break out of the attribute or inject markup.
            live_view_label = html.escape(f"{upload_name} ({edited_image.width}×{edited_image.height})")
            st.markdown(
                f'<div class="panel-header-title canvas-live-view" title="{live_view_label}">🖼️ Live View: <span>{live_view_label}</span></div>',
                unsafe_allow_html=True,
            )
        with top_undo:
            if st.button("↶", key="hist-undo-btn", disabled=not history.can_undo, help="Undo (Ctrl+Z)", use_container_width=True):
                st.session_state["_pending_restore"] = history.undo() or {}
                _request_rerun()
        with top_redo:
            if st.button("↷", key="hist-redo-btn", disabled=not history.can_redo, help="Redo (Ctrl+Shift+Z or Ctrl+Y)", use_container_width=True):
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
        components.html(_UNDO_REDO_KEYBOARD_SHORTCUTS_HTML, height=0)
        with top_c2:
            zoom_level = st.selectbox("Zoom View", [1.0, 0.5, 1.5, 2.0, 4.0], format_func=lambda z: f"🔍 {int(z*100)}%" if z>=1 else f"🔍 {int(z*100)}% (Fit)", key="cv-zoom", label_visibility="collapsed")
        with top_c3:
            compare_mode = st.checkbox("⇄ Split View", value=False, key="cv-split", help="Drag to compare before/after")
            show_original = st.checkbox("👁️ Original", value=False, key="cv-orig", disabled=compare_mode)

        if zoom_level > 1.0:
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
            before_preview = zoom_preview(source, zoom_level, (pan_x, pan_y))
            after_preview = zoom_preview(edited_image, zoom_level, (pan_x, pan_y))
            if before_preview.size != after_preview.size:
                before_preview = before_preview.resize(after_preview.size, Image.Resampling.LANCZOS)
            compare_html, compare_height = _before_after_html(before_preview, after_preview)
            components.html(compare_html, height=compare_height + 10)
        else:
            st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
            if show_original:
                canvas_display, canvas_caption = preview_image(source), "Original Untouched"
            else:
                canvas_display, canvas_caption = zoom_preview(edited_image, zoom_level, (pan_x, pan_y)), "Current Edited Image"
            if grid_type != "None" or guides:
                canvas_display = draw_guides_and_grid(canvas_display, grid_type, guides)
            st.image(canvas_display, caption=canvas_caption, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Histogram -- always reflects edited_image (the backend-processed result), not whichever
        # view (Original / Split) happens to be toggled above, since that's the reference exposure
        # decisions are actually made against.
        histogram_bins = 64
        histogram = compute_histogram(edited_image, bins=histogram_bins)
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
                compute_waveform(edited_image, mode=waveform_mode),
                caption=f"Waveform ({waveform_mode})",
                use_container_width=True,
            )

        # Clipping warning -- the same idea as Lightroom's blown-highlight/blocked-shadow
        # triangles on the histogram, shown as a full preview overlay instead (matching this app's
        # existing overlay-checkbox pattern -- the Masks/Repair preview checkboxes, and the Soft
        # Proof gamut warning). Reflects edited_image, same as the histogram right above it.
        if st.checkbox("⚠️ Show clipping", key="clip-warning-toggle", help="Highlights blocked shadows (blue) and blown highlights (red)."):
            st.image(
                clipping_warning_overlay(edited_image),
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

        # Export Footer Bar
        st.markdown('<div class="export-panel">', unsafe_allow_html=True)
        exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 1.2], gap="small")
        with exp_c1:
            out_format = st.selectbox("Export Format", ["JPEG", "PNG", "WEBP", "TIFF"], key="exp-fmt", label_visibility="collapsed")
        with exp_c2:
            out_quality = st.slider("Quality", 50, 100, 92, key="exp-qual", disabled=out_format in ["PNG", "TIFF"], label_visibility="collapsed")
        with exp_c3:
            buf = io.BytesIO()
            f_name = {"JPEG": "JPEG", "PNG": "PNG", "WEBP": "WEBP", "TIFF": "TIFF"}[out_format]
            export_source = apply_watermark(edited_image, wm_text, wm_position, wm_opacity, wm_font_size) if wm_text.strip() else edited_image
            out_img = export_source.convert("RGB") if f_name in ["JPEG", "WEBP"] else export_source
            opts = {"format": f_name}
            if f_name in ["JPEG", "WEBP"]:
                opts["quality"] = out_quality
            out_img.save(buf, **opts)
            ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif"}[out_format]
            mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "TIFF": "image/tiff"}[out_format]
            st.download_button(
                label=f"💾 Save {out_format}",
                data=buf.getvalue(),
                file_name=f"{Path(upload_name).stem}-edited{ext}",
                mime=mime,
                use_container_width=True,
                key="exp-download-btn",
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
            if st.button("Prepare ZIP", key="exp-multi-btn", disabled=not selected_profiles):
                zip_bytes = export_multiple(
                    edited_image, selected_profiles, Path(upload_name).stem,
                    watermark_text=wm_text if wm_text.strip() else None,
                    watermark_position=wm_position,
                    watermark_opacity=wm_opacity,
                    watermark_font_size=wm_font_size,
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
