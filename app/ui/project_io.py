"""Save/Open Project, and opening a photo from the Library or the desktop launcher's file argument.

Opening anything puts its bytes in st.session_state["restored_source_bytes"/"restored_source_name"]
and bumps "_main_upload_generation" (which empties the main uploader by re-keying it), so the
opened file, not a previously uploaded one, is what gets edited. A project's recipe goes through
"_pending_restore" like every other outside write into the widgets.
"""
from __future__ import annotations

import base64
import io
import os
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

from app.edit_schema import migrate_legacy_keys
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
        **RECIPE_DEFAULTS, "mask_layers": [], "removal_mask_layers": [], "curve_extra_points": [],
        **migrate_legacy_keys(loaded_project.recipe),  # before the merge, so old keys aren't shadowed by new defaults
    }
    pending_recipe["layers"] = layers_from_project(loaded_project)
    st.session_state["restored_source_bytes"] = loaded_project.source_bytes
    st.session_state["restored_source_name"] = loaded_project.source_name
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
    st.session_state["_main_upload_generation"] = st.session_state.get("_main_upload_generation", 0) + 1


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
            open_project_bytes(path.read_bytes())
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
            st.session_state["restored_source_bytes"] = path.read_bytes()
            st.session_state["restored_source_name"] = path.name
        else:
            return
    except (ValueError, KeyError, OSError) as error:
        st.session_state["_launch_file_error"] = f"Could not open {path.name}: {error}"
        return
    st.rerun()
