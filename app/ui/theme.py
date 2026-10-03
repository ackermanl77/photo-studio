"""The app's colours, defined once as CSS variables (F12).

THEME is the palette. The page stylesheet (static/theme.css) and the on-canvas editors
(app/ui_editors.py, rendered in components.html iframes, which don't inherit the page's CSS) both
read it as `var(--ps-<name>)`: page_style() prepends the :root block to theme.css, and
editor_style() gives each editor iframe the same block.

The app pins Streamlit's dark theme (.streamlit/config.toml and the desktop launcher's
--theme.base flag), and this palette is the dark one that goes with it.

Streamlit-free at import (app/ui_editors.py imports it); only inject_page_style() touches st.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"

THEME = {
    "font": "'Segoe UI', system-ui, -apple-system, sans-serif",
    "canvas": "#090d16",          # behind the photo
    "panel": "#0f172a",           # editor panels
    "border": "#1e293b",
    "border-soft": "rgba(255, 255, 255, 0.12)",
    "surface-soft": "rgba(255, 255, 255, 0.06)",
    "surface-faint": "rgba(255, 255, 255, 0.04)",
    "text-strong": "#f1f5f9",
    "text": "#cbd5e1",
    "text-muted": "#94a3b8",
    "brand": "#ea580c",
    "brand-dark": "#c2410c",
    "on-brand": "#ffffff",
    "accent": "#fbbf24",          # active tool, handles, the Add/Update buttons
    "accent-soft": "#fde68a",
    "on-accent": "#1e293b",
    "info": "#60a5fa",
    "info-soft": "#93c5fd",
    "danger": "#f87171",
}


def css_variables() -> str:
    return ":root {" + " ".join(f"--ps-{name}: {value};" for name, value in THEME.items()) + "}"


def static_text(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def page_style() -> str:
    return f"<style>\n{css_variables()}\n{static_text('theme.css')}\n</style>"


def editor_style() -> str:
    """The palette for an editor iframe (prepended to each editor's markup)."""
    return f"<style>{css_variables()}</style>"


def inject_page_style() -> None:
    import streamlit as st

    st.markdown(page_style(), unsafe_allow_html=True)
