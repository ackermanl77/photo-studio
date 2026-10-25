"""The start screen's hands-on tutorials (UI7, like Photoshop's Home "Learn" tasks): each opens the
bundled sample photo at the tool it teaches (through Find a tool's jump_to) and shows its steps in
a card at the top of the sidebar until it's closed. Also the start screen's extras around the
uploader: these tutorials and Recent files."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.i18n import t
from app.ui.project_io import recent_files_section
from app.ui.tool_search import jump_to

# A small bundled placeholder image (generated, not a real photo) so a first-time visitor can
# explore every tool; loaded through the same restored-source path a reopened project uses.
SAMPLE_PHOTO_PATH = Path(__file__).resolve().parents[1] / "assets" / "sample_photo.jpg"
SAMPLE_PHOTO_NAME = "sample_photo.jpg"

_TUTORIAL_KEY = "_tutorial"

# id -> (title, the Find a tool place it opens, extra session state (a widget's value only through
# "_pending_restore"), steps)
TUTORIALS = {
    "enhance": ("✨ Fix a flat photo in one click", "✨ Auto Enhance", {}, (
        "Click ✨ Auto Enhance: it measures the photo and sets exposure, black and white points, white balance and vibrance.",
        "Press B (or 👁 Before / After above the photo) to compare with the original.",
        "Fine-tune with the Tone sliders below; every change is one Undo step (Ctrl+Z).",
        "Ctrl+E saves the result.",
    )),
    "remove": ("🩹 Remove something from a photo", "Remove Objects", {}, (
        "Pick 🩹 Repair brush above the photo.",
        "Paint over the dark square — the ring shows the brush size, [ and ] change it.",
        "Click 🩹 Remove: the photo comes back with the square filled in from its surroundings.",
        "Try Repair Method: Heal for textured areas; ↶ Undo stroke or Ctrl+Z take a stroke back.",
    )),
    "crop": ("✂ Crop for a 4:5 post", "Crop / Aspect Ratio", {"_pending_restore": {"crop_aspect": "4:5 Portrait"}}, (
        "Aspect Ratio is set to 4:5 Portrait — the photo is cropped from its centre.",
        "Choose Free-form instead to drag the crop box over the photo yourself.",
        "Rotate & Straighten, above, levels a tilted horizon first.",
        "⬇ Export picks the format and size; Ctrl+E saves.",
    )),
    "looks": ("🎞️ Try a look", "Preset Library", {}, (
        "Point at a look: the photo shows it right away, without applying it.",
        "Click one to apply it — one Undo step, so Ctrl+Z goes back.",
        "Save Current Look as Preset, below, keeps your own edit as a new card.",
    )),
}


def _start(tutorial_id: str) -> None:
    """A tutorial's button (a callback, before the script): open the sample at the tool."""
    title, place, state, _steps = TUTORIALS[tutorial_id]
    st.session_state["restored_source_bytes"] = SAMPLE_PHOTO_PATH.read_bytes()
    st.session_state["restored_source_name"] = SAMPLE_PHOTO_NAME
    st.session_state.pop("_restored_source_path", None)
    st.session_state.update(state)
    jump_to(place)
    st.session_state[_TUTORIAL_KEY] = tutorial_id


def start_screen_extras() -> None:
    """Below the start screen's uploader: Recent files, then the tutorials."""
    recent_files_section(compact=False)
    st.markdown(t("**🎓 Learn by doing** — on the sample photo"))
    for column, (tutorial_id, (title, _place, _state, _steps)) in zip(st.columns(len(TUTORIALS)), TUTORIALS.items()):
        column.button(t(title), key=f"tutorial-{tutorial_id}", on_click=_start, args=(tutorial_id,), use_container_width=True)


def _close() -> None:
    st.session_state.pop(_TUTORIAL_KEY, None)


def tutorial_panel() -> None:
    """The open tutorial's steps, at the top of the sidebar, until closed."""
    tutorial_id = st.session_state.get(_TUTORIAL_KEY)
    if tutorial_id not in TUTORIALS:
        return
    title, _place, _state, steps = TUTORIALS[tutorial_id]
    with st.container(key="tutorial-card", border=True):
        st.markdown(f"**{t(title)}**")
        st.markdown("\n".join(f"{number}. {t(step)}" for number, step in enumerate(steps, 1)))
        st.button(t("✓ Got it"), key="tutorial-close-btn", on_click=_close)
