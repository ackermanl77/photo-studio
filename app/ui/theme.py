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

import json
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


def _slider_tracks() -> dict[str, str]:
    """Sliders whose track shows what the value means (UI4, like Photoshop's Adjustments): widget
    key -> CSS gradient. Data colours (a colour temperature, a hue), not interface colours, so they
    are plain hsl() here rather than palette variables."""
    from app.professional_color import HSL_BAND_HUES

    def hue(degrees: float, saturation: int = 75, lightness: int = 55) -> str:
        return f"hsl({degrees % 360:.0f} {saturation}% {lightness}%)"

    tracks = {
        "c-temp": f"linear-gradient(to right, {hue(212)}, hsl(0 0% 80%), {hue(46, 90)})",
        "c-tint": f"linear-gradient(to right, {hue(120, 60, 45)}, hsl(0 0% 80%), {hue(305, 70)})",
        "c-saturation": f"linear-gradient(to right, hsl(0 0% 55%), {hue(10, 85)})",
        "c-vib": f"linear-gradient(to right, hsl(0 0% 55%), {hue(10, 85)})",
        "t-ev": "linear-gradient(to right, hsl(0 0% 8%), hsl(0 0% 96%))",
        "t-whites": "linear-gradient(to right, hsl(0 0% 55%), hsl(0 0% 98%))",
        "t-blacks": "linear-gradient(to right, hsl(0 0% 2%), hsl(0 0% 45%))",
        "rl-warmth": f"linear-gradient(to right, {hue(212)}, hsl(0 0% 80%), {hue(32, 90)})",
    }
    rainbow = ", ".join(hue(degrees) for degrees in range(0, 361, 60))
    for prefix in ("cg-sh", "cg-mid", "cg-hi"):
        tracks[f"{prefix}-hue"] = f"linear-gradient(to right, {rainbow})"
    for band, center in HSL_BAND_HUES.items():
        key = f"hsl-{band.lower()}"
        # The hue slider turns the band up to 30 degrees either way (app.professional_color).
        tracks[f"{key}-h"] = f"linear-gradient(to right, {hue(center - 30)}, {hue(center)}, {hue(center + 30)})"
        tracks[f"{key}-s"] = f"linear-gradient(to right, hsl(0 0% 55%), {hue(center, 90)})"
        tracks[f"{key}-l"] = f"linear-gradient(to right, {hue(center, 70, 12)}, {hue(center)}, {hue(center, 70, 88)})"
    return tracks


def slider_tracks_style() -> str:
    """Streamlit DOM: the track is the first child of the slider's react-aria group. `--ps-track`
    tells static/sliders.js to leave these tracks alone (no value fill over the meaning)."""
    track = '[data-testid="stSlider"] [role="group"] > div > div:first-child'
    return "\n".join(f".st-key-{key} {track} {{ background-image: {gradient} !important; --ps-track: 1; }}"
                     for key, gradient in _slider_tracks().items())


@lru_cache(maxsize=None)
def page_style() -> str:
    return f"<style>\n{css_variables()}\n{static_text('theme.css')}\n{slider_tracks_style()}\n</style>"


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
    # A sidebar widened to 520 px or more has room for the names: each tab shows its own label
    # (icon and name, small, wrapping) instead of the bare icon (UI11), and the hover and open-tab
    # names are dropped as redundant.
    rules += [
        f".st-key-{tabs_key} {{ container-type: inline-size; }}",
        f"@container (min-width: 520px) {{ {tab} p {{ font-size: 0.68rem !important; line-height: 1.2;"
        " white-space: normal; text-align: center; }"
        f" {tab} p::before, {tab}:hover::after, {row}::after {{ content: none !important; }} }}",
    ]
    for index, (icon, name) in enumerate(tabs):
        rules.append(f'{tab}[data-key="{index}"] p::before {{ content: {_css_string(icon)}; }}')
        rules.append(f'{tab}[data-key="{index}"]:hover::after {{ content: {_css_string(name)}; }}')
        rules.append(f'{row}:has(> [data-key="{index}"][aria-selected="true"])::after {{ content: {_css_string(name)}; }}')
    return "<style>\n" + "\n".join(rules) + "\n</style>"


# Scripts that work on Streamlit's own markup, in the page itself: accessibility fixes (UI3) and
# slider fills from zero (UI4). Each guards itself, so installing one twice is a no-op.
PAGE_SCRIPTS = ("a11y.js", "sliders.js")


@lru_cache(maxsize=None)
def page_scripts_loader() -> str:
    """A one-shot same-origin script that installs PAGE_SCRIPTS in the page (the parent document),
    where they outlive this iframe -- once per page load (window.__psPageScripts)."""
    code = json.dumps("\n".join(static_text(name) for name in PAGE_SCRIPTS)).replace("</", "<\\/")  # never closes the <script> early
    return ("<script>(function(){const w = window.parent; if (w.__psPageScripts) return; w.__psPageScripts = true;"
            f"const s = w.document.createElement('script'); s.textContent = {code}; w.document.head.appendChild(s);}})();</script>")


def inject_page_style() -> None:
    """The stylesheet, and the scripts for Streamlit's own markup (PAGE_SCRIPTS)."""
    import streamlit as st
    import streamlit.components.v1 as components

    st.markdown(page_style(), unsafe_allow_html=True)
    with st.container(key="a11y-fixes"):
        components.html(page_scripts_loader(), height=0)
