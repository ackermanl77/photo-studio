"""The workspace layout, remembered between launches (Photoshop's workspaces, in small): the tool
tab and sub-tabs that were open, and the Essentials/Pro mode. (The sidebar's width Streamlit already
keeps itself, in the browser's localStorage, `sidebarWidth`.)

The tabs go to settings.json (`"layout"`, app.settings) by their English labels -- their values in
session_state are the labels in the interface language -- and come back once per session, before
the tabs are created.
"""
from __future__ import annotations

import streamlit as st

from app.i18n import t
from app.settings import read_settings, save_setting

_RESTORED_KEY, _SAVED_KEY = "_layout_restored", "_layout_saved"


def restore_tabs(tab_lists: dict[str, list[str]]) -> None:
    """Open the tabs that were open last time: once per session, before the tabs exist. Skips a
    tab whose state this session already has (a test or a jump set it)."""
    if st.session_state.get(_RESTORED_KEY):
        return
    st.session_state[_RESTORED_KEY] = True
    saved = read_settings().get("layout")
    if not isinstance(saved, dict):
        return
    for key, labels in tab_lists.items():
        if saved.get(key) in labels and key not in st.session_state:
            st.session_state[key] = t(saved[key])
    st.session_state[_SAVED_KEY] = {key: saved.get(key) for key in tab_lists}


def remember_tabs(tab_lists: dict[str, list[str]]) -> None:
    """After the tabs rendered: save which ones are open, when that changed."""
    current = {}
    for key, labels in tab_lists.items():
        value = st.session_state.get(key)
        current[key] = next((label for label in labels if value in (label, t(label))), None)
    if current != st.session_state.get(_SAVED_KEY):
        st.session_state[_SAVED_KEY] = current
        save_setting("layout", current)


# --- Essentials / Pro (UI8, Photoshop's Assisted and Pro workspaces) ---------------------------------
# Essentials shows the everyday tool tabs only. The others still run on every rerun (their widgets
# keep their state, like every tab's); their icons are only hidden, and Find a tool switches back to
# Pro to reach one.
MODE_KEY = "ui-mode"
MODES = ("Pro", "Essentials")
MODE_LABELS = {"Pro": "🧰 Pro", "Essentials": "🌱 Essentials"}
ESSENTIALS_HIDDEN_TABS = ("🎨 Curves & Color", "🔬 Detail", "◈ Layers")


def _mode_changed(tool_tabs: list[str]) -> None:
    mode = st.session_state.get(MODE_KEY) or "Pro"
    st.session_state[MODE_KEY] = mode  # a segmented control can be deselected: that's Pro
    save_setting("mode", mode)
    if mode == "Essentials" and st.session_state.get("tool-tab") in {*ESSENTIALS_HIDDEN_TABS, *map(t, ESSENTIALS_HIDDEN_TABS)}:
        st.session_state["tool-tab"] = t(tool_tabs[0])


def show_everything() -> None:
    """Back to Pro (from a callback: Find a tool jumping to a tab Essentials hides)."""
    if st.session_state.get(MODE_KEY) == "Essentials":
        st.session_state[MODE_KEY] = "Pro"
        save_setting("mode", "Pro")


def mode_switch(tool_tabs: list[str]) -> None:
    """The Essentials/Pro switch (remembered in settings.json, `"mode"`), and Essentials' tab hiding."""
    if MODE_KEY not in st.session_state:
        saved = read_settings().get("mode")
        st.session_state[MODE_KEY] = saved if saved in MODES else "Pro"
    mode = st.segmented_control(
        t("Workspace"), MODES, key=MODE_KEY, format_func=lambda name: t(MODE_LABELS[name]), label_visibility="collapsed",
        on_change=_mode_changed, args=(tool_tabs,),
        help=t("Essentials shows the everyday tools only (Basic, Geometry, Local, Effects, Export, Presets); Pro shows every tool. Nothing you set is lost when switching."),
    )
    if mode == "Essentials":
        row = '.st-key-tool-tab > div > [role="tablist"] > [data-testid="stTab"]'
        hidden = ", ".join(f'{row}[data-key="{tool_tabs.index(tab)}"]' for tab in ESSENTIALS_HIDDEN_TABS)
        st.markdown(f"<style>{hidden} {{ display: none !important; }}</style>", unsafe_allow_html=True)
