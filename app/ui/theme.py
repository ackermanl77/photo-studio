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


def _css_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def icon_tabs_style(tabs_key: str, tabs: list[tuple[str, str]]) -> str:
    """Streamlit DOM: a tab row shown as icons, like Photoshop's toolbar -- `tabs` are (icon, name)
    in order; each tab shows its icon, hovering one shows its name, and the open tab's name is
    written under the row. The labels stay the tabs' values (saved state) and their accessible
    names; only their text is hidden. Child combinators keep tab rows nested inside it untouched."""
    row = f'.st-key-{tabs_key} > div > [role="tablist"]'
    tab = f'{row} > [data-testid="stTab"]'
    rules = [
        f"{row} {{ flex-wrap: nowrap !important; justify-content: space-between; gap: 0.1rem; margin-bottom: 1.3rem; position: relative; overflow: visible !important; }}",
        f"{tab} {{ position: relative; padding: 0.3rem 0.5rem !important; }}",
        f"{tab} p {{ font-size: 0 !important; }}",
        f"{tab} p::before {{ font-size: 1.15rem; }}",
        f"{tab}:hover::after {{ position: absolute; top: 100%; left: 50%; transform: translateX(-50%); z-index: 10;"
        " pointer-events: none; white-space: nowrap; padding: 0.1rem 0.5rem; border-radius: 6px; font-size: 0.75rem;"
        " background: var(--ps-panel); color: var(--ps-text-strong); border: 1px solid var(--ps-border-soft); }",
        f"{row}::after {{ position: absolute; top: calc(100% + 0.15rem); left: 0.2rem; font-size: 0.78rem;"
        " font-weight: 700; color: var(--ps-text-muted); }",
    ]
    for index, (icon, name) in enumerate(tabs):
        rules.append(f'{tab}[data-key="{index}"] p::before {{ content: {_css_string(icon)}; }}')
        rules.append(f'{tab}[data-key="{index}"]:hover::after {{ content: {_css_string(name)}; }}')
        rules.append(f'{row}:has(> [data-key="{index}"][aria-selected="true"])::after {{ content: {_css_string(name)}; }}')
    return "<style>\n" + "\n".join(rules) + "\n</style>"


def inject_page_style() -> None:
    import streamlit as st

    st.markdown(page_style(), unsafe_allow_html=True)
