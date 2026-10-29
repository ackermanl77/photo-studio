"""Curves & Color tab: the RGB tone curve, the HSL color mixer, color grading and the soft-proof
gamut warning."""
from __future__ import annotations

import json

import streamlit as st

from app.controls import CONTROLS_BY_KEY
from app.professional_color import HSL_BAND_ORDER, ICC_TRANSFORM_ERRORS
from app.ui.processing import cached_gamut_warning_overlay
from app.ui.state import consume_new_commit, is_modified, request_rerun, section_header, section_label, section_reset_button, show_component, tab_open
from app.ui.widgets import render_control
from app.ui_editors import _curve_editor_html
from app.i18n import t

CURVE_X = (0, 64, 128, 192, 255)
CURVE_KEYS = ("curve_black", "curve_shadows", "curve_mid", "curve_highlights", "curve_white")


# The curve channels (F13): tab label, recipe-key prefix, editor colour.
CURVE_CHANNEL_TABS = (("RGB", "curve", "var(--ps-accent)"), ("Red", "curve_r", "#e5534b"), ("Green", "curve_g", "#57ab5a"), ("Blue", "curve_b", "#539bf5"))
CURVE_POINT_NAMES = ("black", "shadows", "mid", "highlights", "white")


def _channel_curve_keys(prefix: str) -> tuple[str, ...]:
    return CURVE_KEYS if prefix == "curve" else tuple(f"{prefix}_{name}" for name in CURVE_POINT_NAMES)


def curve_section(ctx) -> None:
    st.session_state.setdefault("curve_extra_points", [])
    values = ctx.values
    label = section_label("📈 RGB Tone Curve", "curve", extra_modified=bool(st.session_state["curve_extra_points"]))
    with st.expander(t(label), expanded=False, key="sec-curve"):
        curve_bounds = [(0, 64), (0, 128), (64, 192), (128, 255), (192, 255)]
        values["curve_enabled"] = render_control("curve_enabled")
        # One tab per channel (F13); every tab's sliders render on every run (a widget skipped in a
        # run loses its state), only the open tab draws its editor.
        channel_tabs = st.tabs([t(_label) for _label in [name for name, _prefix, _color in CURVE_CHANNEL_TABS]], key="curve-channel-tab", on_change="rerun")
        for tab, (name, prefix, color) in zip(channel_tabs, CURVE_CHANNEL_TABS):
            keys = _channel_curve_keys(prefix)
            with tab:
                if tab_open(tab):
                    preview_points = [(x, st.session_state.get(CONTROLS_BY_KEY[key].widget_key, CONTROLS_BY_KEY[key].default)) for x, key in zip(CURVE_X, keys)]
                    extra = st.session_state["curve_extra_points"] if prefix == "curve" else None
                    show_component(lambda: (_curve_editor_html(preview_points, curve_bounds, extra, keys=keys, color=color), 300))
                with st.container(key="curve-points" if prefix == "curve" else f"curve-points-{prefix}"):
                    for key in keys:
                        values[key] = render_control(key)
        with st.container(key="curve-hidden"):
            curve_commit = st.text_area(t("Curve commit data"), key="curve-commit-data", label_visibility="collapsed")
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
        st.caption(t("Drag a point on the curve, or use the sliders for exact values. Red/Green/Blue curves shift one colour channel -- e.g. lift Blue in the shadows for a cool cast."))


def hsl_section(ctx) -> None:
    section_header("Color Mixer (HSL)", "hsl")
    st.caption(t("Adjust Hue, Saturation, and Luminance independently for each color band, like a professional editor's Color Mixer."))
    # Every band's sliders render on every rerun (collapsed expanders, not a band picker): a widget
    # that isn't rendered in a run loses its state. persist_state was evaluated for a picker and not
    # adopted (see "Known Issues" in docs/PROJECT_STATUS.md).
    values = ctx.values
    hsl_bands = {}
    for band in HSL_BAND_ORDER:
        keys = [f"hsl_{band.lower()}_{channel}" for channel in ("hue", "sat", "lum")]
        with st.expander(t("{0} {1}").format(t('🟢' if is_modified(keys) else '⚪'), t(band)), expanded=False, key=f"sec-hsl-{band.lower()}"):
            band_values = [render_control(key) for key in keys]
        values.update(zip(keys, band_values))
        hsl_bands[band] = dict(zip(("hue", "sat", "lum"), band_values))
    ctx.hsl_bands = hsl_bands


def black_white_section(ctx) -> None:
    values = ctx.values
    with st.expander(t(section_label("⚫ Black & White", "bw")), expanded=False, key="sec-bw"):
        values["black_white"] = render_control("black_white")
        st.caption(t("Brighten or darken the gray each original colour turns into -- e.g. darken Blue for a dramatic sky. Applies while the conversion is on."))
        # Always rendered (disabled while the conversion is off): a widget skipped in a run loses its state.
        enabled = bool(values["black_white"])
        for column, bands in zip(st.columns(2), (HSL_BAND_ORDER[:4], HSL_BAND_ORDER[4:])):
            with column:
                for band in bands:
                    values[f"bw_{band.lower()}"] = render_control(f"bw_{band.lower()}", disabled=not enabled)
        section_reset_button("bw", "Black & White")


def grading_section(ctx) -> None:
    values = ctx.values
    with st.expander(t(section_label("🎨 Color Grading (Split Toning)", "grading")), expanded=False, key="sec-grading"):
        st.caption(t("Tint shadows, midtones, and highlights toward independent colors — the same three-range color grading every professional editor offers, separate from the global Temperature/Tint."))
        for column, (title, tone_range) in zip(st.columns(3), (("Shadows", "shadows"), ("Midtones", "midtones"), ("Highlights", "highlights"))):
            with column:
                st.markdown(t("**{0}**").format(t(title)))
                for key in (f"cg_{tone_range}_hue", f"cg_{tone_range}_intensity"):
                    values[key] = render_control(key)
        values["cg_balance"] = render_control("cg_balance")


def lut_section(ctx) -> None:
    """LUT (F17): a .cube look from the library, its strength, and importing new ones."""
    from app import lut as lut_module

    values = ctx.values
    with st.expander(t(section_label("🎞️ LUT", "lut")), expanded=False, key="sec-lut"):
        st.caption(t("Apply a 3D LUT (.cube) -- a film emulation or creative look from Resolve, Premiere or a LUT pack. It is applied after the other colour adjustments."))
        values["lut"] = render_control("lut")
        values["lut_amount"] = render_control("lut_amount", disabled=not values["lut"])
        upload = st.file_uploader(t("Import a LUT (.cube)"), type=["cube"], key="lut-upload",
                                  help=t("The LUT is copied into the app's data folder and listed in the LUT choice."))
        if upload is not None:
            imported = st.session_state.setdefault("_luts_imported", set())
            if upload.file_id not in imported:  # a file_uploader keeps its file across reruns
                imported.add(upload.file_id)
                try:
                    name = lut_module.import_lut(upload.getvalue(), upload.name)
                except ValueError as error:
                    st.error(t("⚠️ {0}").format(t(str(error))))
                else:
                    st.toast(t("LUT '{0}' imported").format(name), icon="🎞️")
                    st.session_state["_pending_restore"] = {"lut": name}
                    request_rerun()
        section_reset_button("lut", "LUT")
    # The Adjustments run right after the LUT in the pipeline, so they sit right after it here too
    # (and called from here keeps app/frontend_unified_app.py under its size limit).
    adjustments_section(ctx)


def adjustments_section(ctx) -> None:
    """🧪 Adjustments (PS1, app.adjustments): Photoshop's Image > Adjustments not covered by the other
    sections. Every control renders on every run (disabled while its adjustment is off), so none
    loses its state."""
    values = ctx.values
    with st.expander(t(section_label("🧪 Adjustments", "adjust")), expanded=False, key="sec-adjust"):
        st.caption(t("Photoshop's Image › Adjustments that the other sections don't cover, applied after the LUT and in this order."))
        exposure_col, gamma_col = st.columns(2)
        values["exposure_offset"] = render_control("exposure_offset", exposure_col)
        values["exposure_gamma"] = render_control("exposure_gamma", gamma_col)

        values["levels_enabled"] = render_control("levels_enabled")
        off = not values["levels_enabled"]
        in_col, out_col = st.columns(2)
        with in_col:
            for key in ("levels_in_black", "levels_in_white", "levels_gamma"):
                values[key] = render_control(key, disabled=off)
        with out_col:
            for key in ("levels_out_black", "levels_out_white"):
                values[key] = render_control(key, disabled=off)

        values["mixer_enabled"] = render_control("mixer_enabled")
        off = not values["mixer_enabled"]
        for column, out in zip(st.columns(3), ("r", "g", "b")):
            with column:
                for src in ("r", "g", "b"):
                    values[f"mixer_{out}_{src}"] = render_control(f"mixer_{out}_{src}", disabled=off)

        filter_col, density_col = st.columns(2)
        values["photo_filter"] = render_control("photo_filter", filter_col)
        off = values["photo_filter"] == "None"
        values["photo_filter_density"] = render_control("photo_filter_density", density_col, disabled=off)
        values["photo_filter_luminosity"] = render_control("photo_filter_luminosity", disabled=off)

        values["colorize_enabled"] = render_control("colorize_enabled")
        off = not values["colorize_enabled"]
        for column, key in zip(st.columns(3), ("colorize_hue", "colorize_saturation", "colorize_lightness")):
            values[key] = render_control(key, column, disabled=off)

        values["gradient_map_amount"] = render_control("gradient_map_amount")
        off = not values["gradient_map_amount"]
        shadow_col, highlight_col = st.columns(2)
        values["gradient_map_shadow"] = render_control("gradient_map_shadow", shadow_col, disabled=off)
        values["gradient_map_highlight"] = render_control("gradient_map_highlight", highlight_col, disabled=off)

        values["posterize_levels"] = render_control("posterize_levels")
        threshold_col, level_col = st.columns(2)
        values["threshold_enabled"] = render_control("threshold_enabled", threshold_col)
        values["threshold_level"] = render_control("threshold_level", level_col, disabled=not values["threshold_enabled"])
        values["equalize"] = render_control("equalize")
        section_reset_button("adjust", "Adjustments")


def soft_proof_section(ctx) -> None:
    with st.expander(t("🎯 Soft Proof: Gamut Warning"), expanded=False, key="sec-soft-proof"):
        source_icc_bytes = ctx.source.info.get("icc_profile")
        if not source_icc_bytes:
            st.caption(t("This photo has no embedded color profile to check against, so there's nothing to proof."))
            return
        st.caption(
            t("Highlights, in magenta, pixels that get clipped when this app's own sRGB pipeline converts "
            "the photo's embedded color profile. Not a simulation of a specific printer/press profile -- "
            "there's no profile database bundled -- just where this app's own color management can't "
            "fully represent the source.")
        )
        if st.checkbox(t("Show out-of-gamut pixels"), key="gamut-warning-toggle"):
            try:
                st.image(
                    cached_gamut_warning_overlay(ctx.source_digest, ctx.source, source_icc_bytes),
                    caption="Magenta = clipped when converted to this app's sRGB working space",
                    use_container_width=True,
                )
            except ICC_TRANSFORM_ERRORS:
                st.caption(t("Could not read this photo's embedded color profile."))

