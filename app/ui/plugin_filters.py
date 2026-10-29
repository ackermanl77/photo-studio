"""✨ Effects → 🧩 Photoshop plug-in filters (PS20): pick a third-party .8bf filter, set it up in its own
dialog on the current photo (through the backend, app.filter_plugins), and keep it as a step the
pipeline replays on every render and export. The steps live in st.session_state["plugin_filters"]
(a recipe list, like Liquify's strokes); the chosen plug-ins folder is remembered per user."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import streamlit as st

from app.config import user_data_file
from app.i18n import t
from app.ui.state import request_rerun, section_header

SETTINGS_FILE = "plugin_filters.json"
STEPS_KEY = "plugin_filters"


def _saved_folder() -> str:
    try:
        return str(json.loads(user_data_file(SETTINGS_FILE).read_text(encoding="utf-8")).get("folder") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def _save_folder(folder: str) -> None:
    path = user_data_file(SETTINGS_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"folder": folder}), encoding="utf-8")
    except OSError:
        pass  # only a convenience: the folder is asked for again next time


@st.cache_data(show_spinner=False, max_entries=4)
def _filters_in(folder: str, _modified: float) -> list[dict]:
    """The folder's filters (cached until the folder changes: `_modified` is its newest mtime)."""
    from app.filter_plugins import list_filters

    return list_filters(folder)


def _newest_change(folder: Path) -> float:
    try:
        return max([folder.stat().st_mtime] + [path.stat().st_mtime for path in folder.rglob("*.8bf")])
    except OSError:
        return 0.0


def pipeline_steps() -> list[dict] | None:
    """The plugin_filters option for /process: the steps without their UI-only ids."""
    steps = [{key: value for key, value in step.items() if key != "id"} for step in st.session_state.get(STEPS_KEY, [])]
    return steps or None


def plugin_filters_section(ctx) -> None:
    steps = st.session_state.setdefault(STEPS_KEY, [])
    for step in steps:
        step.setdefault("id", uuid.uuid4().hex)
    section_header("Photoshop plug-in filters", extra_modified=bool(steps))
    from app.filter_plugins import available

    if not available():
        st.caption(t("Runs third-party Photoshop filter plug-ins (.8bf) on the photo. Needs Windows and the filter host, "
                     "which the desktop app includes (from source: scripts/build_filter_host.ps1)."))
        return
    st.caption(t("Third-party Photoshop filter plug-ins (.8bf, 64-bit). Apply opens the filter's own window on the photo; "
                 "the settings you choose there are kept and replayed on every render and export. Adobe's own filters only "
                 "run inside Photoshop and aren't listed."))
    folder = st.text_input(t("Plug-ins folder"), value=st.session_state.get("pf-folder-default") or _saved_folder(),
                           key="pf-folder", placeholder=r"C:\Program Files\Common Files\Adobe\Plug-ins")
    filters: list[dict] = []
    if folder.strip():
        from app.filter_plugins import FilterPluginError

        try:
            filters = _filters_in(folder.strip(), _newest_change(Path(folder.strip())))
        except FilterPluginError as error:
            st.warning(t("⚠️ {0}").format(t(str(error))))
        else:
            if folder.strip() != _saved_folder():
                _save_folder(folder.strip())
            if not filters:
                st.info(t("No 64-bit filter plug-ins (.8bf) were found in this folder or its subfolders."))
    if filters:
        labels = [f"{item['category']} › {item['title']}" if item["category"] else item["title"] for item in filters]
        choice = st.selectbox(t("Filter"), range(len(filters)), format_func=lambda index: labels[index], key="pf-choice")
        if st.button(t("🧩 Apply… (opens the filter's window)"), key="pf-apply"):
            _configure(ctx, filters[choice])
    message = st.session_state.pop("_plugin_filter_message", None)
    if message:
        st.info(message)
    for index, step in enumerate(steps):
        name_col, remove_col = st.columns([5, 1])
        name_col.markdown(f"{index + 1}. **{step['title']}**  \n`{Path(step['file']).name}`")
        if remove_col.button("✖", key=f"pf-remove-{step['id']}", help=t("Remove this filter")):
            steps.pop(index)
            request_rerun()
            break


def _configure(ctx, item: dict) -> None:
    from app.backend_client import configure_plugin_filter_via_backend
    from app.ui.processing import png_bytes

    image = ctx.edited_image if ctx.edited_image is not None else ctx.source
    step = {"file": item["file"], "entry": item["entry"], "title": item["title"]}
    with st.spinner(t("Waiting for {0}'s window…").format(item["title"])):
        try:
            configured = configure_plugin_filter_via_backend(png_bytes(image.convert("RGB")), step)
        except RuntimeError as error:
            st.error(t("⚠️ {0}").format(t(str(error))))
            return
    if configured is None:
        st.session_state["_plugin_filter_message"] = t("{0} was cancelled; nothing changed.").format(item["title"])
    else:
        st.session_state[STEPS_KEY].append({**configured, "id": uuid.uuid4().hex})
        st.session_state["_plugin_filter_message"] = t("{0} added: it now runs after Film Grain, on every render and export.").format(item["title"])
    request_rerun()
