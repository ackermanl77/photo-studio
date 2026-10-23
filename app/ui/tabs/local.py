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
from app.ui.state import CANVAS_MODE_KEY, CANVAS_PHOTO, editor_box, hex_to_rgba, section_header, section_reset_button, show_component, tab_visible
from app.ui_editors import _image_view_html
from app.ui.widgets import render_control
from app.i18n import t, t_option


# The overlay previews are drawn over the photo, like the editors (show_component(canvas_label=...)).
MASK_PREVIEW, REMOVAL_PREVIEW = "🟥 Mask preview", "🟨 Removal preview"


def _open_preview_on_photo(checkbox_key: str, label: str) -> None:
    """Ticking a preview shows it over the photo straight away; unticking goes back to the photo."""
    st.session_state[CANVAS_MODE_KEY] = label if st.session_state.get(checkbox_key) else CANVAS_PHOTO


def _overlay_on_photo(ctx, signature: str, label: str, slot_id: str, color: tuple[int, int, int], style: str) -> None:
    """The combined selection highlighted on the geometry-corrected source (the selection, not the
    edited result), offered over the photo; built only when it is shown there."""
    def build():
        base = preview_image(ctx.mask_source, editor_box())
        mask = Image.open(io.BytesIO(combined_mask_png(ctx.mask_source_key, ctx.mask_source, signature, base.size)))
        return _image_view_html(apply_mask_overlay(base, mask, color=color, style=style), label)

    show_component(build, canvas_label=label, slot_id=slot_id)


def masks_section(ctx) -> None:
    section_header("Mask Layers", "mask_targets", extra_modified=bool(st.session_state.get("mask_layers")))
    st.caption(t("Build several named selections and combine them into one mask. The target adjustments below apply only where the combined mask is active; a selection with \"Own adjustments\" gets its own set instead."))
    mask_source = ctx.mask_source
    ctx.mask_signature = render_selection_stack("mask_layers", "m", mask_source)
    mask_layers = st.session_state.mask_layers
    if not mask_layers:
        st.caption(t("No mask layers yet — the sliders below have no effect until you add one."))
    elif not any(layer["enabled"] for layer in mask_layers):
        st.caption(t("All mask layers are disabled — the sliders below have no effect."))

    if st.checkbox(t("👁️ Preview combined mask overlay"), key="m-preview-overlay", on_change=_open_preview_on_photo,
                   args=("m-preview-overlay", MASK_PREVIEW)) and mask_layers:
        mask_style_col, mask_color_col = st.columns([3, 1])
        mask_style = mask_style_col.radio(t("Overlay"), OVERLAY_STYLES, horizontal=True, key="m-overlay-style", format_func=t_option)
        mask_color = mask_color_col.color_picker(t("Color"), "#DC3723", key="m-overlay-color")
        _overlay_on_photo(ctx, ctx.mask_signature, MASK_PREVIEW, "mask-overlay", hex_to_rgba(mask_color)[:3], mask_style)

    ctx.shared_mask_layers = [layer for layer in mask_layers if not layer.get("adjust")]
    ctx.own_mask_layers = own_adjustment_layers(mask_layers)
    ctx.own_signatures = [selections_signature([layer]) for layer in ctx.own_mask_layers]
    mask_inert = not any(layer["enabled"] for layer in ctx.shared_mask_layers)
    for key in ("mask_target_exposure_ev", "mask_target_contrast_amount", "mask_target_saturation_amount"):
        ctx.values[key] = render_control(key, disabled=mask_inert)
    section_reset_button("mask_targets", "Mask targets")


def repair_section(ctx) -> None:
    section_header("Remove Objects", "repair", extra_modified=bool(st.session_state.get("removal_mask_layers")))
    st.caption(t("Select the blemish or object to remove with the same selection tools as Masks. Fill paints the area over from its edges; Heal copies a matching nearby patch over it."))
    mask_source = ctx.mask_source
    ctx.removal_signature = render_selection_stack(
        "removal_mask_layers", "rm", mask_source, include_opacity=False,
        default_kind="Painted", paint_label="🩹 Repair brush", paint_commit_label="🩹 Remove", paint_back_to_photo=True,
    )
    removal_layers = st.session_state.removal_mask_layers
    ctx.removal_active = any(layer["enabled"] for layer in removal_layers)
    if not removal_layers:
        st.caption(t("No selections yet — nothing will be removed."))
    elif not ctx.removal_active:
        st.caption(t("All selections are disabled — nothing will be removed."))
    ctx.values["removal_method"] = render_control("removal_method", disabled=not ctx.removal_active)
    ctx.values["removal_radius"] = render_control("removal_radius", disabled=not ctx.removal_active)
    if st.checkbox(t("👁️ Preview removal selection"), key="rm-preview-overlay", on_change=_open_preview_on_photo,
                   args=("rm-preview-overlay", REMOVAL_PREVIEW)) and removal_layers:
        removal_style = st.radio(t("Overlay"), OVERLAY_STYLES, horizontal=True, key="rm-overlay-style", format_func=t_option)
        # Yellow (vs. Masks' red) so the two overlays are never confused.
        _overlay_on_photo(ctx, ctx.removal_signature, REMOVAL_PREVIEW, "rm-overlay", (255, 200, 0), removal_style)


def portrait_section(ctx) -> None:
    section_header("Face & Skin Retouching", "portrait")
    # 0 by default: skin smoothing is opt-in.
    ctx.values["portrait_smoothing"] = render_control("portrait_smoothing")
    ctx.values["portrait_faces_only"] = render_control(
        "portrait_faces_only", disabled=not ctx.values["portrait_smoothing"] or face_detection_backend() == "none"
    )
    if face_detection_backend() == "none":
        st.caption(t("👤 Face detection is unavailable in this OpenCV build (see System Health)."))
    elif tab_visible():
        st.caption(t("👤 Detected faces: {0}").format(t(face_count(ctx.source_digest, ctx.source))))
