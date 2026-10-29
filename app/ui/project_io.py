"""Save/Open Project, and opening a photo from the Library or the desktop launcher's file argument.

Opening anything puts its bytes in st.session_state["restored_source_bytes"/"restored_source_name"]
and bumps "_main_upload_generation" (which empties the main uploader by re-keying it), so the
opened file, not a previously uploaded one, is what gets edited. A project's recipe goes through
"_pending_restore" like every other outside write into the widgets. A file opened from a path also
sets "_restored_source_path", so Recent files can list it by that path instead of a copy.

Recent files (app.recent_files) are recorded by remember_opened_file() once the opened photo has
decoded, and listed by recent_files_section() on the start screen and in the open bar.
"""
from __future__ import annotations

import base64
import io
import os
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

from app import recent_files
from app.edit_schema import migrate_legacy_keys
from app.i18n import t
from app.major_features import BLEND_MODE_LABELS
from app.project_bundle import ProjectBundle
from app.raw_import import SUPPORTED_EXTENSIONS
from app.ui.state import RECIPE_DEFAULTS

# What a saved project may expand to and still reopen (app.reliability.validate_project_archive).
PROJECT_MAX_BYTES = 250 * 1024 * 1024


def project_archive(source_name: str, source_bytes: bytes, recipe: dict, layers: list[dict]) -> bytes:
    """The .photoproject bytes for Save Project -- built only when the download button is clicked
    (Streamlit runs a download_button's data callable on its own thread; inputs are bound up front
    with functools.partial). Layers are saved parametrically (app.project_bundle.ProjectBundle's
    layer_stack): a Text/Shape/Path layer reopens editable, and each Image layer's pixels go into
    their own PNG entry instead of base64 inside JSON."""
    stack: list[dict] = []
    images: dict[str, bytes] = {}
    for layer in layers:
        entry = {key: value for key, value in layer.items() if key != "image_b64"}
        if layer.get("kind", "image") == "image":
            name = f"{uuid.UUID(hex=str(layer.get('id') or uuid.uuid4().hex)).hex}.png"
            images[name] = base64.b64decode(layer["image_b64"])
            entry["image_file"] = name
        stack.append(entry)
    return ProjectBundle(
        source_name=source_name, source_bytes=source_bytes, recipe=recipe,
        layer_stack=stack if stack else None, layer_images=images or None,
    ).to_bytes()


def estimated_project_bytes(source_bytes: bytes, layers: list[dict]) -> int:
    """Roughly what project_archive() will write (PNG entries are already compressed, so the
    archive is about the sum of its parts) -- checked before offering the download, since an error
    raised inside the download callable can't be shown to the user."""
    return len(source_bytes) + sum(len(layer.get("image_b64", "")) * 3 // 4 for layer in layers) + 64 * 1024


def layers_from_project(project: ProjectBundle) -> list[dict]:
    """The Layers tab's layer dicts for a loaded project: parametric (editable) for a current
    project, flattened Image layers for one saved before layers were stored parametrically."""
    restored: list[dict] = []
    if project.layer_stack is not None:
        for layer in project.layer_stack:
            entry = {key: value for key, value in layer.items() if key != "image_file"}
            entry["id"] = uuid.uuid4().hex
            entry.setdefault("kind", "image")
            entry.setdefault("name", f"Layer {len(restored) + 1}")
            entry.setdefault("opacity", 1.0)
            entry.setdefault("blend_mode", "normal")
            entry.setdefault("visible", True)
            if entry["blend_mode"] not in BLEND_MODE_LABELS:
                entry["blend_mode"] = "normal"
            if entry["kind"] == "image":
                content = (project.layer_images or {}).get(str(layer.get("image_file")))
                if content is None:
                    continue
                Image.open(io.BytesIO(content)).verify()  # an unreadable PNG fails the load, cleanly
                entry["image_b64"] = base64.b64encode(content).decode("ascii")
            restored.append(entry)
        return restored
    workspace = project.layer_workspace_object()
    if workspace:
        # workspace.layers[0] is the LayerWorkspace-internal "Background" (the source image itself,
        # already restored separately) -- everything from index 1 on is a real overlay layer.
        for overlay_layer in workspace.layers[1:]:
            overlay_buffer = io.BytesIO()
            overlay_layer.image.save(overlay_buffer, format="PNG")
            restored.append(
                {
                    "id": uuid.uuid4().hex,
                    "kind": "image",  # older projects stored flattened pixels, even for a Text layer
                    "name": overlay_layer.name,
                    "image_b64": base64.b64encode(overlay_buffer.getvalue()).decode("ascii"),
                    "opacity": overlay_layer.opacity,
                    "blend_mode": overlay_layer.blend_mode,
                    "visible": overlay_layer.visible,
                }
            )
    return restored


def open_project_bytes(content: bytes) -> None:
    """Load a .photoproject into this session (the caller reruns). Raises ValueError/KeyError/
    OSError for an unreadable file -- never a raw traceback on the page."""
    loaded_project = ProjectBundle.load(content)
    # Merged over RECIPE_DEFAULTS so any control the saved recipe doesn't mention (a project saved
    # before that control existed, or a hand-trimmed one) is reset to its default, rather than
    # silently keeping whatever the previous photo's edit had there.
    pending_recipe = {
        **RECIPE_DEFAULTS, "mask_layers": [], "removal_mask_layers": [], "curve_extra_points": [], "liquify_strokes": [], "plugin_filters": [],
        **migrate_legacy_keys(loaded_project.recipe),  # before the merge, so old keys aren't shadowed by new defaults
    }
    pending_recipe["layers"] = layers_from_project(loaded_project)
    st.session_state["restored_source_bytes"] = loaded_project.source_bytes
    st.session_state["restored_source_name"] = loaded_project.source_name
    st.session_state.pop("_restored_source_path", None)
    # Empties the main "Open Image" uploader on the next run (see its key) so the project's bundled
    # source, not a previously opened photo, is what gets edited.
    st.session_state["_main_upload_generation"] = st.session_state.get("_main_upload_generation", 0) + 1
    st.session_state["_pending_restore"] = pending_recipe


def open_library_photo(path: Path) -> None:
    """Open a catalogued photo in the editor (F8): the same restored-source path a reopened project
    uses, emptying the main uploader so it doesn't take priority. The adjustments are kept, so a
    look can be carried from one photo of a shoot to the next."""
    st.session_state["restored_source_bytes"] = path.read_bytes()
    st.session_state["restored_source_name"] = path.name
    st.session_state["_restored_source_path"] = str(path)
    st.session_state["_main_upload_generation"] = st.session_state.get("_main_upload_generation", 0) + 1
    # Which catalogued file is open (virtual copies are saved against it, F21).
    st.session_state["_library_open_path"] = str(path)


def open_launch_file() -> None:
    """A file the desktop launcher was started with (double-clicking a .photoproject, or "Open
    with" on a photo -- see app.desktop.OPEN_PATH_ENV), opened once per session."""
    path_text = os.environ.get("PHOTO_STUDIO_OPEN_PATH")
    if not path_text or st.session_state.get("_launch_file_opened"):
        return
    st.session_state["_launch_file_opened"] = True
    path = Path(path_text)
    try:
        if path.suffix.lower() in (".photoproject", ".zip"):
            content = path.read_bytes()
            open_project_bytes(content)
            remember_opened_project(path.name, content, str(path))
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
            st.session_state["restored_source_bytes"] = path.read_bytes()
            st.session_state["restored_source_name"] = path.name
            st.session_state["_restored_source_path"] = str(path)
        else:
            return
    except (ValueError, KeyError, OSError) as error:
        st.session_state["_launch_file_error"] = f"Could not open {path.name}: {error}"
        return
    st.rerun()


# --- Recent files ---------------------------------------------------------------------------------

def remember_opened_project(name: str, content: bytes, path: str | None = None) -> None:
    """List an opened project; its thumbnail follows once its photo has decoded (next run)."""
    entry_id = recent_files.remember(name, content, "project", path=path)
    st.session_state["_recent_project_id"] = entry_id


def remember_opened_file(source_digest: str, name: str, content: bytes, source: Image.Image, from_upload: bool, listed: bool = True) -> None:
    """Once per opened photo, after it decoded: list it in Recent files -- or, for a project just
    opened, give the project's entry its thumbnail instead. `listed=False` (the bundled sample)
    leaves it out."""
    if st.session_state.get("_recent_recorded") == source_digest:
        return
    st.session_state["_recent_recorded"] = source_digest
    project_id = st.session_state.pop("_recent_project_id", None)
    if project_id:
        recent_files.set_thumbnail(project_id, source)
    elif listed:
        path = None if from_upload else st.session_state.get("_restored_source_path")
        recent_files.remember(name, content, "photo", source, path=path)


def _open_recent(entry_id: str) -> None:
    """A recent file's button (a callback: it runs before the script, like the other openers)."""
    entry = next((entry for entry in recent_files.load() if entry["id"] == entry_id), None)
    if entry is None:
        return
    try:
        content = recent_files.read_bytes(entry)
        if entry["kind"] == "project":
            open_project_bytes(content)
            st.session_state["_recent_project_id"] = entry_id
            recent_files.remember(entry["name"], content, "project", path=entry.get("path"))  # back to the top
        else:
            st.session_state["restored_source_bytes"] = content
            st.session_state["restored_source_name"] = entry["name"]
            st.session_state["_main_upload_generation"] = st.session_state.get("_main_upload_generation", 0) + 1
            if entry.get("path"):
                st.session_state["_restored_source_path"] = entry["path"]
            else:
                st.session_state.pop("_restored_source_path", None)
    except (ValueError, KeyError, OSError) as error:
        st.session_state["_launch_file_error"] = f"Could not open {entry['name']}: {error}"
        recent_files.forget(entry_id)


def _when(timestamp: float) -> str:
    import time

    days = int((time.time() - timestamp) // 86400)
    return t("today") if days < 1 else t("yesterday") if days < 2 else t("{0} days ago").format(days)


def recent_files_section(compact: bool) -> None:
    """Recent files: a grid of thumbnails on the start screen, a short list in the open bar."""
    entries = recent_files.load()
    if not entries:
        return
    if compact:
        st.caption(t("🕘 Recent"))
        for entry in entries:
            icon = "📁" if entry["kind"] == "project" else "🖼️"
            st.button(f"{icon} {entry['name']}", key=f"recent-open-{entry['id']}", on_click=_open_recent, args=(entry["id"],),
                      use_container_width=True, help=t("Opened {0}").format(_when(entry.get("opened", 0))))
        return
    st.markdown(t("**🕘 Recent**"))
    for row_start in range(0, len(entries), 4):
        for column, entry in zip(st.columns(4), entries[row_start:row_start + 4]):
            with column:
                thumb = recent_files.thumbnail_path(entry)
                if thumb is not None:
                    st.image(str(thumb), use_container_width=True)
                icon = "📁" if entry["kind"] == "project" else "🖼️"
                st.button(f"{icon} {entry['name']}", key=f"recent-open-{entry['id']}", on_click=_open_recent, args=(entry["id"],),
                          use_container_width=True, help=t("Opened {0}").format(_when(entry.get("opened", 0))))
                st.caption(t("Project · {0}").format(_when(entry.get("opened", 0))) if entry["kind"] == "project" else _when(entry.get("opened", 0)))
    st.button(t("🗑️ Clear recent files"), key="recent-clear-btn", on_click=recent_files.clear,
              help=t("Forget this list (and the copies of uploaded files kept for it). Your own files aren't touched."))
