"""Session-state conventions every panel shares.

The rules these helpers encode (see CLAUDE.md, "Architecture"):

- Every tab's body runs on every rerun and every widget is instantiated each pass -- Streamlit
  drops the state of a keyed widget that isn't rendered in a full run. Only display-only work
  (editor iframes, preview images, face detection) is skipped for hidden tabs, via tab_visible().
- A widget's value is written from outside the widget (Undo/Redo, Reset, presets, Load Project,
  Auto Enhance, the on-canvas editors' commits) only through st.session_state["_pending_restore"],
  applied by restore_recipe_to_state() at the very top of the next run, before any widget exists.
- A rerun is requested with request_rerun() and happens once, at the end of the script
  (consume_rerun_request()), after every widget has rendered -- an immediate st.rerun() before the
  last widget would drop the state of every widget not yet reached.
"""
from __future__ import annotations

import threading

import streamlit as st
import streamlit.components.v1 as components

from app.controls import CONTROLS_BY_KEY, RECIPE_WIDGET_KEYS, section_keys
from app.edit_schema import recipe_defaults, sanitize_recipe
from app.painted_masks import intern_recipe_painted_masks
from app.i18n import t

# Defaults (and every control's valid range/type) come from the app.controls registry, through
# app.edit_schema.RECIPE_CONTROLS -- the table sanitize_recipe() clamps restored values against.
RECIPE_DEFAULTS = recipe_defaults()
# The recipe's list entries, kept in plain session_state lists (not widgets).
RECIPE_LIST_STATE = {"mask_layers": "mask_layers", "removal_mask_layers": "removal_mask_layers", "curve_extra_points": "curve_extra_points",
                     "liquify_strokes": "liquify_strokes"}


def build_recipe() -> dict:
    """Every editable control's current value as one JSON-serializable dict -- Undo/Redo snapshots,
    presets and the recipe inside a .photoproject. The layer stack's images are deliberately not
    included (saved with the project file, kept out of per-step history)."""
    recipe = {key: st.session_state.get(widget_key, RECIPE_DEFAULTS[key]) for key, widget_key in RECIPE_WIDGET_KEYS.items()}
    for recipe_key, state_key in RECIPE_LIST_STATE.items():
        recipe[recipe_key] = st.session_state.get(state_key, [])
    return recipe


def restore_recipe_to_state(recipe: dict) -> None:
    """Write a saved recipe into the session_state keys the widgets read. Only valid before any
    widget is created in this run (Streamlit forbids writing an instantiated widget's key), so it
    runs from the "_pending_restore" block at the top of the script. Every restore path funnels
    through here, so the recipe is sanitized once (app.edit_schema.sanitize_recipe): out-of-range
    values are clamped, wrong types fall back to the default, unknown keys are dropped."""
    # Painted masks from a file arrive inline; inside the session they're stored once and
    # referenced by hash (app.painted_masks, P5).
    recipe = intern_recipe_painted_masks(sanitize_recipe(recipe))
    for key, widget_key in RECIPE_WIDGET_KEYS.items():
        if key in recipe:
            st.session_state[widget_key] = recipe[key]
    for recipe_key, state_key in RECIPE_LIST_STATE.items():
        if recipe_key in recipe:
            st.session_state[state_key] = recipe[recipe_key]
    if "layers" in recipe:
        st.session_state.image_layers = recipe["layers"]


def request_rerun() -> None:
    """Use instead of st.rerun() anywhere in a panel: the rerun happens at the end of the script
    (consume_rerun_request) once every widget has rendered. In a fragment-only run (see
    run_fragment) the fragment's wrapper performs it when the fragment body ends."""
    st.session_state["_rerun_pending"] = True


def consume_rerun_request() -> None:
    """The very last statement of the script: perform a deferred rerun, if one was requested."""
    if st.session_state.pop("_rerun_pending", False):
        st.rerun()


def in_fragment_rerun() -> bool:
    """True during a fragment-only rerun (not a full script run)."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx()
    return bool(ctx is not None and ctx.fragment_ids_this_run)


def run_fragment(body, *args, run_every=None) -> None:
    """Run `body(*args)` as an st.fragment (F1): a widget inside it reruns only the fragment --
    nothing else on the page runs, and nothing outside it loses state (Streamlit doesn't clean up
    widgets a fragment-only run didn't reach). A request_rerun() made inside a fragment-only run is
    turned into a full-app rerun when the body ends; in a full run the usual end-of-script rerun
    handles it. The arguments are the ones from the last full run, so a fragment must only depend
    on them and on its own widgets."""

    def wrapper(*inner):
        body(*inner)
        if in_fragment_rerun() and st.session_state.pop("_rerun_pending", False):
            st.rerun(scope="app")

    # Streamlit identifies a fragment by its function's module + name + container: two fragments
    # in the same column need distinct names.
    wrapper.__module__ = body.__module__
    wrapper.__name__ = wrapper.__qualname__ = f"fragment_{body.__qualname__}"
    st.fragment(wrapper, run_every=run_every)(*args)


def consume_new_commit(pending_value: str, committed_key: str) -> bool:
    """True exactly once per genuinely new value an on-canvas editor wrote into its hidden
    `st.text_area` (the editors' cross-iframe commit channel, see app/ui_editors.py). Marks
    `committed_key` first, so the same string arriving again on an unrelated rerun -- or a value
    that turns out not to parse -- is a no-op instead of a duplicate."""
    if not pending_value or pending_value == st.session_state.get(committed_key):
        return False
    st.session_state[committed_key] = pending_value
    return True


# Whether the tab being rendered is the one on screen. Every tab body still runs on every rerun,
# but display-only work is skipped in hidden tabs. Thread-local: each session's script run (and
# fragment run) has its own thread.
_visibility = threading.local()


def set_tab_visible(visible: bool) -> None:
    _visibility.visible = bool(visible)


def tab_visible() -> bool:
    return getattr(_visibility, "visible", True)


def tab_open(tab) -> bool:
    """`.open` of a tab from st.tabs(..., on_change="rerun") (True when Streamlit doesn't say)."""
    is_open = getattr(tab, "open", None)
    return True if is_open is None else bool(is_open)


def show_component(build) -> None:
    """components.html() for an on-canvas editor, only when its tab is visible -- `build()` returns
    (markup, height) and isn't even called otherwise (it base64-encodes a preview). The editor's
    hidden commit text_area is rendered by the caller regardless, so its state survives."""
    if tab_visible():
        markup, height = build()
        components.html(markup, height=height)


def hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), 255)


# --- Sections: reset buttons and "modified" indicators (F2, F9) ----------------------------------

MODIFIED_MARK = "●"


def is_modified(keys) -> bool:
    """Whether any of these recipe controls differs from its default right now."""
    for key in keys:
        control = CONTROLS_BY_KEY[key]
        value = st.session_state.get(control.widget_key, control.default)
        if value != control.default:
            return True
    return False


def sections_modified(*sections: str) -> bool:
    return any(is_modified(section_keys(section)) for section in sections)


def section_label(title: str, *sections: str, extra_modified: bool = False) -> str:
    """`title` plus a "●" when any control of these sections has been changed -- the label of a
    section's expander or header. Expanders that carry it are keyed, so the label change doesn't
    collapse them."""
    title = t(title)
    return f"{title} {MODIFIED_MARK}" if extra_modified or sections_modified(*sections) else title


def section_header(title: str, *sections: str, extra_modified: bool = False) -> None:
    """A bold section heading with the "modified" mark (a coloured dot plus the word, so it doesn't
    rely on colour alone)."""
    modified = extra_modified or sections_modified(*sections)
    st.markdown(f"**{t(title)}**" + (f" :orange[{MODIFIED_MARK}] :gray[{t('modified')}]" if modified else ""))


def section_reset_button(section: str, label: str, *more_sections: str) -> None:
    """A "↺ Reset <label>" button that puts one or more sections' controls back to their defaults,
    through the same _pending_restore path as Undo/Reset."""
    if st.button(t("↺ Reset {0}").format(t(label)), key=f"reset-{section}-btn", help=t("Reset the {0} controls to their defaults").format(t(label))):
        keys = [key for name in (section, *more_sections) for key in section_keys(name)]
        st.session_state["_pending_restore"] = {key: RECIPE_DEFAULTS[key] for key in keys}
        request_rerun()
