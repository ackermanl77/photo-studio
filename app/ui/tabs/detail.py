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
    section_reset_button("effects", "Effects")
