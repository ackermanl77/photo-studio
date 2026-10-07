"""Curves & Color tab: the RGB tone curve, the HSL color mixer, color grading and the soft-proof
gamut warning."""
from __future__ import annotations

import json

import streamlit as st

from app.controls import CONTROLS_BY_KEY
from app.professional_color import HSL_BAND_ORDER, ICC_TRANSFORM_ERRORS
from app.ui.processing import cached_gamut_warning_overlay
from app.ui.state import consume_new_commit, is_modified, request_rerun, section_header, section_label, section_reset_button, show_component
from app.ui.widgets import render_control
from app.ui_editors import _curve_editor_html

CURVE_X = (0, 64, 128, 192, 255)
CURVE_KEYS = ("curve_black", "curve_shadows", "curve_mid", "curve_highlights", "curve_white")


def curve_section(ctx) -> None:
    st.session_state.setdefault("curve_extra_points", [])
    values = ctx.values
    label = section_label("📈 RGB Tone Curve", "curve", extra_modified=bool(st.session_state["curve_extra_points"]))
    with st.expander(label, expanded=False, key="sec-curve"):
        curve_bounds = [(0, 64), (0, 128), (64, 192), (128, 255), (192, 255)]
        preview_points = [(x, st.session_state.get(CONTROLS_BY_KEY[key].widget_key, CONTROLS_BY_KEY[key].default)) for x, key in zip(CURVE_X, CURVE_KEYS)]
        show_component(lambda: (_curve_editor_html(preview_points, curve_bounds, st.session_state["curve_extra_points"]), 270))
        with st.container(key="curve-hidden"):
            curve_commit = st.text_area("Curve commit data", key="curve-commit-data", label_visibility="collapsed")
        # A dragged point commits its recipe values here (a script-dispatched slider event doesn't
        # reliably reach the server), applied through _pending_restore like Undo/Redo.
        if consume_new_commit(curve_commit, "curve-committed"):
            try:
                parsed = json.loads(curve_commit)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                st.session_state["_pending_restore"] = parsed
                request_rerun()
        with st.container(key="curve-points"):
            for key in ("curve_enabled", *CURVE_KEYS):
                values[key] = render_control(key)
        st.caption("Drag a point on the curve, or use the sliders below for exact values.")


def hsl_section(ctx) -> None:
    section_header("Color Mixer (HSL)", "hsl")
    st.caption("Adjust Hue, Saturation, and Luminance independently for each color band, like a professional editor's Color Mixer.")
    # Every band's sliders render on every rerun (collapsed expanders, not a band picker): a widget
    # that isn't rendered in a run loses its state. persist_state was evaluated for a picker and not
    # adopted (see "Known Issues" in docs/PROJECT_STATUS.md).
    values = ctx.values
    hsl_bands = {}
    for band in HSL_BAND_ORDER:
        keys = [f"hsl_{band.lower()}_{channel}" for channel in ("hue", "sat", "lum")]
        with st.expander(f"{'🟢' if is_modified(keys) else '⚪'} {band}", expanded=False, key=f"sec-hsl-{band.lower()}"):
            band_values = [render_control(key) for key in keys]
        values.update(zip(keys, band_values))
        hsl_bands[band] = dict(zip(("hue", "sat", "lum"), band_values))
    ctx.hsl_bands = hsl_bands


def black_white_section(ctx) -> None:
    values = ctx.values
    with st.expander(section_label("⚫ Black & White", "bw"), expanded=False, key="sec-bw"):
        values["black_white"] = render_control("black_white")
        st.caption("Brighten or darken the gray each original colour turns into -- e.g. darken Blue for a dramatic sky. Applies while the conversion is on.")
        # Always rendered (disabled while the conversion is off): a widget skipped in a run loses its state.
        enabled = bool(values["black_white"])
        for column, bands in zip(st.columns(2), (HSL_BAND_ORDER[:4], HSL_BAND_ORDER[4:])):
            with column:
                for band in bands:
                    values[f"bw_{band.lower()}"] = render_control(f"bw_{band.lower()}", disabled=not enabled)
        section_reset_button("bw", "Black & White")


def grading_section(ctx) -> None:
    values = ctx.values
    with st.expander(section_label("🎨 Color Grading (Split Toning)", "grading"), expanded=False, key="sec-grading"):
        st.caption("Tint shadows, midtones, and highlights toward independent colors — the same three-range color grading every professional editor offers, separate from the global Temperature/Tint.")
        for column, (title, tone_range) in zip(st.columns(3), (("Shadows", "shadows"), ("Midtones", "midtones"), ("Highlights", "highlights"))):
            with column:
                st.markdown(f"**{title}**")
                for key in (f"cg_{tone_range}_hue", f"cg_{tone_range}_intensity"):
                    values[key] = render_control(key)
        values["cg_balance"] = render_control("cg_balance")


def soft_proof_section(ctx) -> None:
    with st.expander("🎯 Soft Proof: Gamut Warning", expanded=False):
        source_icc_bytes = ctx.source.info.get("icc_profile")
        if not source_icc_bytes:
            st.caption("This photo has no embedded color profile to check against, so there's nothing to proof.")
            return
        st.caption(
            "Highlights, in magenta, pixels that get clipped when this app's own sRGB pipeline converts "
            "the photo's embedded color profile. Not a simulation of a specific printer/press profile -- "
            "there's no profile database bundled -- just where this app's own color management can't "
            "fully represent the source."
        )
        if st.checkbox("Show out-of-gamut pixels", key="gamut-warning-toggle"):
            try:
                st.image(
                    cached_gamut_warning_overlay(ctx.source_digest, ctx.source, source_icc_bytes),
                    caption="Magenta = clipped when converted to this app's sRGB working space",
                    use_container_width=True,
                )
            except ICC_TRANSFORM_ERRORS:
                st.caption("Could not read this photo's embedded color profile.")

