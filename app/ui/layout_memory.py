"""The workspace layout, remembered between launches (Photoshop's workspaces, in small): the tool
tab and sub-tabs that were open. (The sidebar's width Streamlit already keeps itself, in the
browser's localStorage, `sidebarWidth`.)

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
