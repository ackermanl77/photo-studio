"""Basic tab: Auto Enhance, preset look, Tone, Color (white balance, saturation, vibrance) and
Presence (texture, clarity, dehaze) -- the everyday adjustments, Lightroom's "Basic" panel."""
from __future__ import annotations

import json

import streamlit as st

from app.controls import CONTROLS_BY_KEY
from app.image_processor import auto_enhance_recipe
from app.professional_color import estimate_white_balance
from app.professional_tools import preview_image
from app.ui.processing import preset_thumbnails
from app.ui.state import consume_new_commit, request_rerun, section_header, section_label, section_reset_button, show_component, tab_visible
from app.ui.widgets import render_control
from app.ui_editors import _wand_picker_html


def auto_enhance_button(ctx) -> None:
    """Image-adaptive (app.image_processor.auto_enhance_recipe, measured on a small proxy):
    exposure, black/white points, white balance, vibrance and -- only for a noisy photo -- denoise,
    applied through _pending_restore; everything else stays as the user left it."""
    if st.button("✨ Auto Enhance", key="auto-enhance-btn", help="Measure this photo and set exposure, black/white points, white balance and vibrance to suit it", use_container_width=True):
        auto_settings = auto_enhance_recipe(ctx.source)
        st.session_state["_pending_restore"] = auto_settings
        changed = [
            f"exposure {auto_settings['exposure_ev']:+.2f} EV" if auto_settings["exposure_ev"] else None,
            f"black/white points {auto_settings['curve_black']}/{auto_settings['curve_white']}" if auto_settings["curve_enabled"] else None,
            f"temperature {auto_settings['temperature']:+d}, tint {auto_settings['tint']:+d}" if auto_settings["temperature"] or auto_settings["tint"] else None,
            f"vibrance +{auto_settings['vibrance']}" if auto_settings["vibrance"] else None,
            "denoise" if auto_settings["denoise"] else None,
        ]
        st.toast("Auto Enhance: " + (", ".join(item for item in changed if item) or "this photo already looks balanced"), icon="✨")
        request_rerun()


def tone_section(ctx) -> None:
    values = ctx.values
    preset_modified = st.session_state.get("tone-preset", "None") != "None"
    with st.expander(section_label("🎞️ Preset Look", extra_modified=preset_modified), key="sec-preset-look"):
        preset_names = list(CONTROLS_BY_KEY["preset"].choices)
        if tab_visible():
            thumbs = preset_thumbnails(ctx.source_digest, ctx.source)
            for thumb_col, name in zip(st.columns(len(preset_names)), preset_names):
                with thumb_col:
                    st.image(thumbs[name], use_container_width=True)
                    st.caption(name)
        values["preset"] = render_control("preset")
    section_header("Tone", "tone")
    for key in ("exposure_ev", "contrast_amount", "highlights", "shadows", "whites", "blacks"):
        values[key] = render_control(key)
    section_reset_button("tone", "Tone")


def color_section(ctx) -> None:
    section_header("Color", "color")
    values = ctx.values
    for key in ("temperature", "tint", "saturation_amount", "vibrance"):
        values[key] = render_control(key)
    section_reset_button("color", "Color")
    with st.expander("🎯 Pick White Balance", expanded=False):
        st.caption("Click a spot in the photo that should be neutral gray or white -- Temperature and Tint above are set to whatever would neutralize it. A starting point to fine-tune from, not a guaranteed perfect result.")
        show_component(lambda: _wand_picker_html(preview_image(ctx.mask_source, (480, 480)), "wb"))
        with st.container(key="wb-wand-hidden"):
            wb_click_raw = st.text_area("White balance click", key="wb-wand-data", label_visibility="collapsed")
        if consume_new_commit(wb_click_raw, "wb-wand-committed"):
            try:
                wb_click = json.loads(wb_click_raw)
                wb_x, wb_y = float(wb_click["x"]), float(wb_click["y"])
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                pass
            else:
                new_temp, new_tint = estimate_white_balance(ctx.mask_source, wb_x, wb_y)
                st.session_state["_pending_restore"] = {"temperature": new_temp, "tint": new_tint}
                request_rerun()


def presence_section(ctx) -> None:
    section_header("Presence", "presence")
    for key in ("texture", "clarity", "dehaze"):
        ctx.values[key] = render_control(key)
    section_reset_button("presence", "Presence")
