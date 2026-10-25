"""Find a tool (Photoshop's "Search"): a box at the top of the sidebar that jumps to any tool in
the nine tabs -- it opens the tab (and sub-tab, and the section's expander, when it has one) and
scrolls the sidebar to the tool, which flashes briefly.

The index is the curated sections below plus every registry control (app.controls) whose label
is unique. The tabs (and radios) are switched through session_state from the box's on_change
callback, which runs before the script; the rest is a one-shot same-origin script that opens the
expander (the app's expanders don't track state, so it clicks its summary) and scrolls to the
place's widget by its `.st-key-<key>` class.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import streamlit as st
import streamlit.components.v1 as components

from app.controls import CONTROLS
from app.i18n import t
from app.ui.layout_memory import ESSENTIALS_HIDDEN_TABS, show_everything

FIND_KEY = "find-tool"
_SCROLL_KEY = "_find_scroll"

BASIC, COLOR, DETAIL, GEOMETRY, LOCAL = "☼ Basic", "🎨 Curves & Color", "🔬 Detail", "⛶ Geometry", "◉ Local"
LAYERS, EFFECTS, EXPORT, LIBRARY = "◈ Layers", "✨ Effects", "⬇ Export", "✦ Presets & Library"
MASKS, REPAIR, PORTRAIT = ("local-tab", "◉ Masks"), ("local-tab", "🩹 Repair"), ("local-tab", "◌ Portrait")
PRESETS, LIBRARY_BATCH = ("library-tab", "✦ Presets"), ("library-tab", "☰ Library & Batch")


@dataclass(frozen=True)
class Place:
    tab: str
    anchor: str | None = None  # the widget key scrolled to (its `.st-key-<anchor>` element)
    sub_tab: tuple[str, str] | None = None  # (the sub-tabs' key, the sub-tab's label)
    expander: str | None = None  # a keyed expander to open (in the page, by the scroll script)
    state: dict = field(default_factory=dict)  # other widgets to set (e.g. the Library's tool radio)


# Where each registry section's controls live.
SECTION_PLACES = {
    "tone": Place(BASIC), "color": Place(BASIC), "presence": Place(BASIC),
    "curve": Place(COLOR, expander="sec-curve"), "hsl": Place(COLOR), "bw": Place(COLOR, expander="sec-bw"),
    "grading": Place(COLOR, expander="sec-grading"), "lut": Place(COLOR, expander="sec-lut"),
    "detail": Place(DETAIL), "effects": Place(EFFECTS), "relight": Place(EFFECTS), "background": Place(EFFECTS),
    "portrait": Place(LOCAL, sub_tab=PORTRAIT), "mask_targets": Place(LOCAL, sub_tab=MASKS),
    "repair": Place(LOCAL, sub_tab=REPAIR),
    "crop": Place(GEOMETRY), "geometry": Place(GEOMETRY), "lens": Place(GEOMETRY),
}
# Controls inside an expander of their own (not their whole section's).
_CONTROL_EXPANDERS = {"tone-preset": "sec-preset-look", "tone-preset-amount": "sec-preset-look"}

# The sections and tools that aren't one registry control.
SECTIONS = {
    "✨ Auto Enhance": Place(BASIC, "auto-enhance-btn"),
    "🎞️ Preset Look": Place(BASIC, "sec-preset-look", expander="sec-preset-look"),
    "Tone": Place(BASIC, "t-ev"),
    "Color": Place(BASIC, "c-temp"),
    "🎯 Pick White Balance": Place(BASIC, "sec-pick-wb", expander="sec-pick-wb"),
    "Presence": Place(BASIC, "d-text"),
    "RGB Tone Curve": Place(COLOR, "sec-curve", expander="sec-curve"),
    "Color Mixer (HSL)": Place(COLOR, "sec-hsl-red"),
    "⚫ Black & White": Place(COLOR, "sec-bw", expander="sec-bw"),
    "🎨 Color Grading (Split Toning)": Place(COLOR, "sec-grading", expander="sec-grading"),
    "🎞️ LUT": Place(COLOR, "sec-lut", expander="sec-lut"),
    "🎯 Soft Proof: Gamut Warning": Place(COLOR, "sec-soft-proof", expander="sec-soft-proof"),
    "Noise Reduction & Sharpening": Place(DETAIL, "d-den"),
    "Rotate & Straighten": Place(GEOMETRY, "geo-rotate"),
    "Lens Corrections": Place(GEOMETRY, "lens-profile"),
    "Perspective (Keystone)": Place(GEOMETRY, "lens-persp-v"),
    "🌀 Liquify": Place(GEOMETRY, "liq-tool"),
    "Crop / Aspect Ratio": Place(GEOMETRY, "crop-aspect"),
    "Mask Layers": Place(LOCAL, "m-new-kind", sub_tab=MASKS),
    "🤖 AI Mask": Place(LOCAL, "m-new-kind", sub_tab=MASKS, state={"m-new-kind": "AI Mask"}),
    "Remove Objects": Place(LOCAL, "rm-new-kind", sub_tab=REPAIR),
    "Face & Skin Retouching": Place(LOCAL, "port-sm", sub_tab=PORTRAIT),
    "Layers": Place(LAYERS, "l-new-kind"),
    "📐 Align & distribute": Place(LAYERS, "sec-l-align", expander="sec-l-align"),
    "🔤 Fonts": Place(LAYERS, "sec-fonts", expander="sec-fonts"),
    "Post-Crop Vignette": Place(EFFECTS, "fx-vignette-amt"),
    "Film Grain": Place(EFFECTS, "fx-grain-amt"),
    "Creative Filters": Place(EFFECTS, "fx-lens-blur"),
    "💡 Relight": Place(EFFECTS, "rl-amount"),
    "Background": Place(EFFECTS, "bg-mode"),
    "Export settings": Place(EXPORT, "exp-fmt"),
    "🏷️ Watermark": Place(EXPORT, "sec-watermark", expander="sec-watermark"),
    "AI Upscale": Place(EXPORT, "exp-ai-upscale"),
    "📦 Export Multiple Sizes": Place(EXPORT, "sec-export-multi", expander="sec-export-multi"),
    "Layered PSD": Place(EXPORT, "exp-psd-btn"),
    "SVG (vector layers)": Place(EXPORT, "exp-svg-btn"),
    "Project File": Place(EXPORT, "proj-dl-btn"),
    "Preset Library": Place(LIBRARY, sub_tab=PRESETS),
    "Library Catalog": Place(LIBRARY, "more-mode", sub_tab=LIBRARY_BATCH, state={"more-mode": "Library Catalog"}),
    "Batch Processor": Place(LIBRARY, "more-mode", sub_tab=LIBRARY_BATCH, state={"more-mode": "Batch Processor"}),
    "HDR & Panorama": Place(LIBRARY, "more-mode", sub_tab=LIBRARY_BATCH, state={"more-mode": "HDR & Panorama"}),
    "System Health": Place(LIBRARY, "more-mode", sub_tab=LIBRARY_BATCH, state={"more-mode": "System Health"}),
}


def _control_places() -> dict[str, Place]:
    """Every registry control with a widget and a label no other control shares ("Amount", "Hue"
    and the like are reached through their section instead)."""
    counts = Counter(control.label for control in CONTROLS)
    places = {}
    for control in CONTROLS:
        section = SECTION_PLACES.get(control.section)
        if section is None or control.widget_key is None or counts[control.label] > 1 or control.label in SECTIONS:
            continue
        places[control.label] = Place(section.tab, control.widget_key, section.sub_tab,
                                      _CONTROL_EXPANDERS.get(control.widget_key, section.expander))
    return places


PLACES = {**SECTIONS, **_control_places()}


def jump_to(name: str) -> None:
    """Open the place called `name` (a key of PLACES) and scroll to it on the next run. Only from a
    callback: it sets the state of widgets that already exist."""
    place = PLACES[name]
    if place.tab in ESSENTIALS_HIDDEN_TABS:
        show_everything()
    st.session_state["tool-tab"] = t(place.tab)
    if place.sub_tab:
        st.session_state[place.sub_tab[0]] = t(place.sub_tab[1])
    st.session_state.update(place.state)
    st.session_state[_SCROLL_KEY] = {"anchor": place.anchor, "expander": place.expander}


def _on_find() -> None:
    name = st.session_state.get(FIND_KEY)
    if name in PLACES:
        jump_to(name)
    st.session_state[FIND_KEY] = None  # ready for the next search


def find_tool_box() -> None:
    """The search box, and the scroll a jump asked for."""
    st.selectbox(
        t("Find a tool"), list(PLACES), index=None, key=FIND_KEY, on_change=_on_find, label_visibility="collapsed",
        placeholder=t("🔎 Find a tool…"), format_func=lambda name: f"{t(name)} · {t(PLACES[name].tab)}",
        help=t("Type a tool's name: it opens its tab and scrolls to it."),
    )
    target = st.session_state.pop(_SCROLL_KEY, None)
    if target and (target["anchor"] or target["expander"]):
        with st.container(key="find-scroll"):
            components.html(_SCROLL_SCRIPT.replace("__TARGET__", json.dumps(target)), height=0)


# Same-origin (srcdoc), so it reaches the page: wait for the tab's content, open the expander, scroll
# the place into the middle of the sidebar and flash it (.ps-find-flash in theme.css).
_SCROLL_SCRIPT = """<script>
(function() {
  const doc = window.parent.document, target = __TARGET__;
  const inSidebar = (key) => key && doc.querySelector('[data-testid="stSidebar"] .st-key-' + key);
  let tries = 0;
  (function find() {
    const details = inSidebar(target.expander) && inSidebar(target.expander).querySelector('details');
    if (details && !details.open && details.offsetParent !== null) details.querySelector('summary').click();
    const el = inSidebar(target.anchor || target.expander);
    if (el && el.offsetParent !== null && (!details || details.open)) {
      el.scrollIntoView({block: 'center', behavior: 'smooth'});
      el.classList.remove('ps-find-flash'); void el.offsetWidth; el.classList.add('ps-find-flash');
    } else if (tries++ < 25) {
      setTimeout(find, 120);
    }
  })();
})();
</script>"""
