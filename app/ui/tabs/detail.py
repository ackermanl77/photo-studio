"""Detail tab (noise reduction and sharpening) and Effects tab (post-crop vignette, film grain)."""
from __future__ import annotations

import streamlit as st

from app.ui.state import section_header, section_reset_button
from app.ui.widgets import render_control
from app.i18n import t


def detail_section(ctx) -> None:
    section_header("Noise Reduction & Sharpening", "detail")
    values = ctx.values
    values["denoise"] = render_control("denoise")
    values["denoise_color"] = render_control("denoise_color", disabled=values["denoise"] == 0)
    values["sharpen"] = render_control("sharpen")
    section_reset_button("detail", "Detail")
    st.caption(t("Texture, Clarity and Dehaze are in **☼ Basic → Presence**; output sharpening for a delivery size is in **⬇ Export**."))


def effects_section(ctx) -> None:
    section_header("Effects", "effects")
    st.caption(t("Creative finishing touches applied to the final, already-cropped frame -- separate from the Lens corrections' vignette, which runs before Crop."))
    values = ctx.values
    fx_v_col, fx_g_col = st.columns(2)
    with fx_v_col:
        st.markdown(t("**Post-Crop Vignette**"))
        values["effects_vignette_amount"] = render_control("effects_vignette_amount")
        values["effects_vignette_midpoint"] = render_control("effects_vignette_midpoint")
    with fx_g_col:
        st.markdown(t("**Film Grain**"))
        values["effects_grain_amount"] = render_control("effects_grain_amount")
    st.markdown(t("**Creative Filters**"))
    blur_col, motion_col = st.columns(2)
    with blur_col:
        values["effects_lens_blur"] = render_control("effects_lens_blur")
        values["effects_lens_blur_focus"] = render_control("effects_lens_blur_focus")
    with motion_col:
        values["effects_motion_blur"] = render_control("effects_motion_blur")
        values["effects_motion_angle"] = render_control("effects_motion_angle")
    bloom_col, halftone_col = st.columns(2)
    with bloom_col:
        values["effects_bloom"] = render_control("effects_bloom")
        values["effects_bloom_threshold"] = render_control("effects_bloom_threshold")
    with halftone_col:
        values["effects_halftone"] = render_control("effects_halftone")
        values["effects_halftone_mode"] = render_control("effects_halftone_mode")
    values["effects_glitch"] = render_control("effects_glitch")
    section_reset_button("effects", "Effects")
    relight_section(ctx)
    background_section(ctx)


def relight_section(ctx) -> None:
    """F32: one movable light over a classical (shape-from-shading) normal map (app.relight)."""
    section_header("💡 Relight", "relight")
    st.caption(t("Adds a light from a new direction: slopes facing it brighten, the others darken. Brighter parts of the photo are read as nearer, so it suits portraits, products and landscapes better than flat graphics."))
    values = ctx.values
    values["relight_amount"] = render_control("relight_amount")
    inactive = values["relight_amount"] == 0
    dir_col, height_col = st.columns(2)
    values["relight_azimuth"] = render_control("relight_azimuth", dir_col, disabled=inactive)
    values["relight_elevation"] = render_control("relight_elevation", height_col, disabled=inactive)
    ambient_col, warmth_col = st.columns(2)
    values["relight_ambient"] = render_control("relight_ambient", ambient_col, disabled=inactive)
    values["relight_warmth"] = render_control("relight_warmth", warmth_col, disabled=inactive)
    section_reset_button("relight", "Relight")


def background_section(ctx) -> None:
    """F31: one step to blur, remove or recolour the background (app.background)."""
    section_header("Background", "background")
    values = ctx.values
    mode_col, subject_col = st.columns(2)
    values["background_mode"] = render_control("background_mode", mode_col)
    values["background_subject"] = render_control("background_subject", subject_col, disabled=values["background_mode"] == "None")
    amount_col, color_col = st.columns(2)
    values["background_amount"] = render_control("background_amount", amount_col, disabled=values["background_mode"] != "Blur")
    values["background_color"] = render_control("background_color", color_col, disabled=values["background_mode"] != "Color")
    if values["background_mode"] == "Remove":
        st.caption(t("The background becomes transparent: export as PNG or TIFF to keep it (JPEG and WebP exports have no transparency)."))
    section_reset_button("background", "Background")
