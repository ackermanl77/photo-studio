"""Photo Studio -- the only Streamlit entry point (`streamlit run app/frontend_unified_app.py`).

This script sets the page up, opens the photo, and composes the panels in app/ui/ (see that
package's docstring). It never edits pixels: the panels collect widget values into an
EditorContext, app.ui.processing turns them into pipeline options and renders the edit on the
backend (app/backend.py -> app.pipeline.run_pipeline, the one place the pipeline order lives).

Run order matters for Streamlit's state rules (see CLAUDE.md):
1. "_pending_restore" is applied before any widget exists;
2. every tab body runs on every rerun (hidden ones skip display-only work via tab_visible());
3. a requested rerun happens only at the very end, after the canvas column's widgets.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from app import backend_client
from app.config import MAX_UPLOAD_BYTES
from app.edit_history import EditHistory
from app.geometry import orient
from app.observability import configure_logging
from app.raw_import import uploader_types
from app.ui import canvas, export_panel, processing
from app.ui.context import EditorContext
from app.ui.project_io import open_launch_file, open_project_bytes
from app.ui.state import build_recipe, consume_rerun_request, restore_recipe_to_state, set_tab_visible, tab_open
from app.ui.tabs import basic, color, detail, geometry, layers_tab, library, local
from app.ui.theme import inject_page_style
from app.i18n import LANGUAGES, save_language, saved_language, set_language, t

# A small bundled placeholder image (generated, not a real photo) so a first-time visitor can
# explore every tool; loaded through the same restored-source path a reopened project uses.
SAMPLE_PHOTO_PATH = Path(__file__).resolve().parent / "assets" / "sample_photo.jpg"
SAMPLE_PHOTO_NAME = "sample_photo.jpg"

# The tool tabs, grouped by workflow (F9). The labels are the "tool-tab" widget's values (tests
# and saved session state select tabs by them); sub-tabs have their own keys ("local-tab",
# "library-tab"). Their order is a workflow choice, independent of the pipeline order.
TOOL_TABS = ["☼ Basic", "🎨 Curves & Color", "🔬 Detail", "⛶ Geometry", "◉ Local", "◈ Layers", "✨ Effects", "⬇ Export", "✦ Presets & Library"]
LOCAL_TABS = ["◉ Masks", "🩹 Repair", "◌ Portrait"]
LIBRARY_TABS = ["✦ Presets", "☰ Library & Batch"]

st.set_page_config(page_title="Photo Studio Pro", page_icon="🎨", layout="wide", initial_sidebar_state="collapsed")
# The Streamlit process's own log (R2: modules never configure logging on import). Idempotent.
configure_logging("frontend")
inject_page_style()

# Interface language (app/i18n.py): the selector's value from the previous run (or the saved
# preference on the first run) is this run's language, set before any text is rendered.
st.session_state.setdefault("ui-language", saved_language())
set_language(st.session_state["ui-language"])

st.markdown(f"""
<div class="top-header">
    <div class="brand-title">
        <span>🎨 Photo Studio</span>
        <span class="brand-badge">CPU Pro</span>
    </div>
    <div class="header-info">{t("⚡ Professional Photo Workstation")}</div>
</div>
""", unsafe_allow_html=True)

open_launch_file()
if st.session_state.get("_launch_file_error"):
    st.error(t("⚠️ {0}").format(t(st.session_state.pop('_launch_file_error'))))

# Undo/Redo, Reset, presets, Load Project, Auto Enhance, the editors' commits: applied here, before
# any widget below is created (see app.ui.state.restore_recipe_to_state).
if "_pending_restore" in st.session_state:
    restore_recipe_to_state(st.session_state.pop("_pending_restore"))

# --- Open a photo ---------------------------------------------------------------------------------
upload_col1, upload_col2, language_col = st.columns([2.5, 1, 0.55], gap="small")
with language_col:
    st.selectbox(
        t("🌐 Language"), list(LANGUAGES), format_func=LANGUAGES.get, key="ui-language",
        help=t("Interface language. Remembered for the next launch."),
        on_change=lambda: save_language(st.session_state["ui-language"]),
    )
with upload_col1:
    # Keyed by a generation counter: a file_uploader can't be cleared through session_state, so
    # opening a project/Library photo empties it by re-keying it (the opened file must win).
    uploaded = st.file_uploader(
        t("Open Image"),
        type=uploader_types(),  # app.raw_import.SUPPORTED_EXTENSIONS, the one accepted-format list
        help=t("Open image (JPG, PNG, WEBP, TIFF, or Camera RAW). Original is never overwritten."),
        label_visibility="collapsed",
        key=f"main-upload-{st.session_state.get('_main_upload_generation', 0)}",
    )
with upload_col2:
    st.caption(t("📂 Open any JPG, PNG, WEBP, TIFF, or camera RAW file."))
    if not uploaded and "restored_source_bytes" not in st.session_state:
        if st.button(
            t("🖼️ Try a sample photo"),
            key="sample-photo-btn",
            use_container_width=True,
            help=t("No photo of your own handy? Load a small bundled placeholder image so you can explore every tool right away."),
        ):
            st.session_state["restored_source_bytes"] = SAMPLE_PHOTO_PATH.read_bytes()
            st.session_state["restored_source_name"] = SAMPLE_PHOTO_NAME
            st.rerun()

with st.expander(t("📁 Open a saved project (.photoproject)")):
    project_upload = st.file_uploader(t("Project file"), type=["photoproject", "zip"], key="proj-upload")
    if project_upload and st.button(t("Load Project"), key="proj-load-btn"):
        try:
            open_project_bytes(project_upload.getvalue())
        except (ValueError, KeyError, OSError) as error:
            st.error(t("⚠️ Could not open this project file: {0}").format(t(error)))
        else:
            st.rerun()

if not uploaded and "restored_source_bytes" not in st.session_state:
    st.info(t("👆 Upload an image above, load a saved project, or click \"Try a sample photo\" to open the editing workspace."))
    st.stop()

# A fresh upload wins; otherwise the photo a project/Library/launch file opened.
upload_name = uploaded.name if uploaded else st.session_state["restored_source_name"]
upload_bytes = uploaded.getvalue() if uploaded else st.session_state["restored_source_bytes"]
if len(upload_bytes) > MAX_UPLOAD_BYTES:
    st.error(t("⚠️ '{0}' is {1:.0f} MB -- over this app's {2} MB limit per photo.").format(t(upload_name), t(len(upload_bytes) / 1024 ** 2), t(MAX_UPLOAD_BYTES // 1024 ** 2)))
    st.stop()

# Decoded once per upload (RAW included), keyed by a digest hashed once per upload.
source_digest = processing.source_digest(getattr(uploaded, "file_id", None) if uploaded else ("restored", id(upload_bytes)), upload_bytes)
try:
    source = processing.decoded_source(source_digest, upload_bytes, upload_name)
except RuntimeError as error:
    st.error(t("⚠️ {0}").format(t(error)))
    st.stop()
except OSError:
    # PIL's own message for an unreadable file includes a memory address -- not worth showing.
    st.error(t("⚠️ Couldn't open '{0}' — the file may be corrupted, incomplete, or not a supported image format.").format(t(upload_name)))
    st.stop()

try:
    backend_client.ensure_backend_running()
except RuntimeError as error:
    st.error(t("⚠️ {0}").format(t(error)))
    st.stop()
# Tags this session's backend requests, so a newer render supersedes an older one ("latest wins").
st.session_state.setdefault("_edit_session", uuid.uuid4().hex)
backend_client.bind_edit_session(st.session_state["_edit_session"])

ctx = EditorContext(
    source=source, source_digest=source_digest, upload_name=upload_name, upload_bytes=upload_bytes,
    # The source's own EXIF (Orientation already applied), attached to every export.
    source_exif=source.getexif().tobytes() if len(source.getexif()) else None,
)

# --- The workspace: tool tabs | canvas -------------------------------------------------------------
with st.container(key="workspace-split"):
    tools_col, canvas_col = st.columns([1.1, 1.4], gap="large")

    with tools_col:
        st.markdown(t('<div class="panel-header-title">🛠️ Tool Workspace</div>'), unsafe_allow_html=True)
        # on_change="rerun" + key: Streamlit reports which tab is on screen (`.open`), so hidden
        # tabs skip their display-only work. Switching tabs is a (cheap, cached) rerun.
        tabs = dict(zip(TOOL_TABS, st.tabs([t(_label) for _label in TOOL_TABS], key="tool-tab", on_change="rerun")))

        def enter(tab_label: str, sub_tab=None):
            set_tab_visible(tab_open(tabs[tab_label]) and (sub_tab is None or tab_open(sub_tab)))

        # Geometry first: Crop and the Masks/Repair/White Balance tools draw against its frame.
        enter("⛶ Geometry")
        with tabs["⛶ Geometry"]:
            geometry.rotate_section(ctx)
            st.divider()
            geometry.lens_section(ctx)
            geometry.compute_geometry(ctx)
            st.divider()
            geometry.liquify_section(ctx)
            st.divider()
            geometry.crop_section(ctx)

        enter("☼ Basic")
        with tabs["☼ Basic"]:
            basic.auto_enhance_button(ctx)
            basic.tone_section(ctx)
            st.divider()
            basic.color_section(ctx)
            st.divider()
            basic.presence_section(ctx)

        enter("🎨 Curves & Color")
        with tabs["🎨 Curves & Color"]:
            color.curve_section(ctx)
            color.hsl_section(ctx)
            color.black_white_section(ctx)
            color.grading_section(ctx)
            color.lut_section(ctx)
            color.soft_proof_section(ctx)

        enter("🔬 Detail")
        with tabs["🔬 Detail"]:
            detail.detail_section(ctx)

        with tabs["◉ Local"]:
            local_tabs = dict(zip(LOCAL_TABS, st.tabs([t(_label) for _label in LOCAL_TABS], key="local-tab", on_change="rerun")))
            enter("◉ Local", local_tabs["◉ Masks"])
            with local_tabs["◉ Masks"]:
                local.masks_section(ctx)
            enter("◉ Local", local_tabs["🩹 Repair"])
            with local_tabs["🩹 Repair"]:
                local.repair_section(ctx)
            enter("◉ Local", local_tabs["◌ Portrait"])
            with local_tabs["◌ Portrait"]:
                local.portrait_section(ctx)

        enter("◈ Layers")
        with tabs["◈ Layers"]:
            layers_tab.layers_section(ctx)

        enter("✨ Effects")
        with tabs["✨ Effects"]:
            detail.effects_section(ctx)

        with tabs["✦ Presets & Library"]:
            library_tabs = dict(zip(LIBRARY_TABS, st.tabs([t(_label) for _label in LIBRARY_TABS], key="library-tab", on_change="rerun")))
            enter("✦ Presets & Library", library_tabs["✦ Presets"])
            with library_tabs["✦ Presets"]:
                library.presets_section(ctx)
            enter("✦ Presets & Library", library_tabs["☰ Library & Batch"])
            with library_tabs["☰ Library & Batch"]:
                library.library_section(ctx)

        # Undo/Redo history: one step per distinct recipe (every control has rendered by now).
        if "edit_history" not in st.session_state:
            st.session_state.edit_history = EditHistory()
        ctx.history = st.session_state.edit_history
        ctx.history.record(build_recipe())

        # Export settings first (the Save footer needs them), then the edit itself.
        enter("⬇ Export")
        with tabs["⬇ Export"]:
            export_panel.export_settings_section(ctx)
            export_panel.watermark_section(ctx)

        processing.render_edit(ctx)

        enter("⬇ Export")
        with tabs["⬇ Export"]:
            export_panel.export_multiple_section(ctx)
            export_panel.layered_psd_section(ctx)
            st.divider()
            export_panel.project_file_section(ctx)

    set_tab_visible(True)  # the canvas column is always on screen
    with canvas_col:
        # "Original" and Split View compare against the source in the edit's orientation.
        oriented_source = orient(source, ctx.values["rotate"], ctx.values["flip_horizontal"], ctx.values["flip_vertical"])
        canvas.render_canvas_column(ctx, oriented_source)
        export_panel.save_footer(ctx)

# Any panel's request_rerun() happens here, after every widget (tabs and canvas) has rendered.
consume_rerun_request()
