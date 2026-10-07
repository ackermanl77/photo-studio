"""Local tab: Masks (local adjustments), Repair (object removal) and Portrait (skin smoothing) --
the adjustments that apply to part of the photo."""
from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from app.portrait_tools import face_detection_backend
from app.professional_tools import OVERLAY_STYLES, apply_mask_overlay, preview_image
from app.ui.processing import face_count
from app.ui.selections import combined_mask_png, own_adjustment_layers, render_selection_stack, selections_signature
from app.ui.state import hex_to_rgba, section_header, section_reset_button, tab_visible
from app.ui.widgets import render_control


def masks_section(ctx) -> None:
    section_header("Mask Layers", "mask_targets", extra_modified=bool(st.session_state.get("mask_layers")))
    st.caption("Build several named selections and combine them into one mask. The target adjustments below apply only where the combined mask is active; a selection with \"Own adjustments\" gets its own set instead.")
    mask_source, mask_source_key = ctx.mask_source, ctx.mask_source_key
    ctx.mask_signature = render_selection_stack("mask_layers", "m", mask_source)
    mask_layers = st.session_state.mask_layers
    if not mask_layers:
        st.caption("No mask layers yet — the sliders below have no effect until you add one.")
    elif not any(layer["enabled"] for layer in mask_layers):
        st.caption("All mask layers are disabled — the sliders below have no effect.")

    if st.checkbox("👁️ Preview combined mask overlay", key="m-preview-overlay") and mask_layers:
        # Overlaid on the geometry-corrected source: shows the selection, not the edited result.
        mask_preview_base = preview_image(mask_source, (1024, 1024))
        mask_preview = Image.open(io.BytesIO(combined_mask_png(mask_source_key, mask_source, ctx.mask_signature, mask_preview_base.size)))
        mask_style_col, mask_color_col = st.columns([3, 1])
        mask_style = mask_style_col.radio("Overlay", OVERLAY_STYLES, horizontal=True, key="m-overlay-style")
        mask_color = mask_color_col.color_picker("Color", "#DC3723", key="m-overlay-color")
        st.image(apply_mask_overlay(mask_preview_base, mask_preview, color=hex_to_rgba(mask_color)[:3], style=mask_style), caption="Highlighted = selected by the combined mask", use_container_width=True)

    ctx.shared_mask_layers = [layer for layer in mask_layers if not layer.get("adjust")]
    ctx.own_mask_layers = own_adjustment_layers(mask_layers)
    ctx.own_signatures = [selections_signature([layer]) for layer in ctx.own_mask_layers]
    mask_inert = not any(layer["enabled"] for layer in ctx.shared_mask_layers)
    for key in ("mask_target_exposure_ev", "mask_target_contrast_amount", "mask_target_saturation_amount"):
        ctx.values[key] = render_control(key, disabled=mask_inert)
    section_reset_button("mask_targets", "Mask targets")


def repair_section(ctx) -> None:
    section_header("Remove Objects", "repair", extra_modified=bool(st.session_state.get("removal_mask_layers")))
    st.caption("Select the blemish or object to remove with the same selection tools as Masks. Fill paints the area over from its edges; Heal copies a matching nearby patch over it.")
    mask_source, mask_source_key = ctx.mask_source, ctx.mask_source_key
    ctx.removal_signature = render_selection_stack("removal_mask_layers", "rm", mask_source, include_opacity=False)
    removal_layers = st.session_state.removal_mask_layers
    ctx.removal_active = any(layer["enabled"] for layer in removal_layers)
    if not removal_layers:
        st.caption("No selections yet — nothing will be removed.")
    elif not ctx.removal_active:
        st.caption("All selections are disabled — nothing will be removed.")
    ctx.values["removal_method"] = render_control("removal_method", disabled=not ctx.removal_active)
    ctx.values["removal_radius"] = render_control("removal_radius", disabled=not ctx.removal_active)
    if st.checkbox("👁️ Preview removal selection", key="rm-preview-overlay") and removal_layers:
        # Yellow (vs. Masks' red) so the two overlays are never confused.
        removal_preview_base = preview_image(mask_source, (1024, 1024))
        removal_preview_mask = Image.open(io.BytesIO(combined_mask_png(mask_source_key, mask_source, ctx.removal_signature, removal_preview_base.size)))
        removal_style = st.radio("Overlay", OVERLAY_STYLES, horizontal=True, key="rm-overlay-style")
        st.image(apply_mask_overlay(removal_preview_base, removal_preview_mask, color=(255, 200, 0), style=removal_style), caption="Highlighted = will be removed", use_container_width=True)


def portrait_section(ctx) -> None:
    section_header("Face & Skin Retouching", "portrait")
    # 0 by default: skin smoothing is opt-in.
    ctx.values["portrait_smoothing"] = render_control("portrait_smoothing")
    if face_detection_backend() == "none":
        st.caption("👤 Face detection is unavailable in this OpenCV build (see System Health).")
    elif tab_visible():
        st.caption(f"👤 Detected faces: {face_count(ctx.source_digest, ctx.source)}")
