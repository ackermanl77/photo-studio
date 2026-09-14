from __future__ import annotations

import base64
import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from app.advanced_masks import MaskLayer, MaskStack, color_mask, gradient_mask, luminance_mask, radial_mask, soften_mask
from app.backend_client import ensure_backend_running, process_via_backend
from app.batch_release import make_zip, process_files
from app.custom_presets import PresetStore
from app.edit_history import EditHistory
from app.image_processor import PhotoProcessor
from app.layer_workspace import LayerWorkspace, WorkspaceLayer
from app.library import index_folder
from app.major_features import PhotoCatalog
from app.presets import PRESETS
from app.portrait_tools import detect_faces
from app.professional_tools import EXPORT_PROFILES, apply_mask_overlay, export_profile, preview_image, zoom_preview
from app.project_bundle import ProjectBundle

# Aspect ratios offered on the Crop tab (width, height).
ASPECT_RATIOS = {
    "1:1 Square": (1, 1),
    "4:5 Portrait": (4, 5),
    "5:4 Landscape": (5, 4),
    "3:2 Landscape": (3, 2),
    "16:9 Widescreen": (16, 9),
    "9:16 Story": (9, 16),
}

# Maps a JSON-serializable recipe key to the session_state key of the widget that reads/writes it.
# Shared by Undo/Redo (EditHistory snapshots) and Save/Load Project (ProjectBundle.recipe) so both
# features capture and restore exactly the same set of controls.
RECIPE_WIDGET_KEYS = {
    "exposure": "t-exp", "contrast": "t-cont", "highlights": "t-high", "shadows": "t-shad", "preset": "tone-preset",
    "temperature": "c-temp", "tint": "c-tint", "saturation": "c-sat",
    "curve_enabled": "c-curve-en", "curve_black": "c-p0", "curve_shadows": "c-p1", "curve_mid": "c-p2",
    "curve_highlights": "c-p3", "curve_white": "c-p4",
    "texture": "d-text", "clarity": "d-clar", "denoise": "d-den", "sharpen": "d-sharp",
    "portrait_smoothing": "port-sm",
    "mask_target_exposure": "m-texp", "mask_target_contrast": "m-tcont", "mask_target_saturation": "m-tsat",
    "crop_aspect": "crop-aspect",
}
RECIPE_DEFAULTS = {
    "exposure": 1.0, "contrast": 1.0, "highlights": 0, "shadows": 0, "preset": "None",
    "temperature": 0, "tint": 0, "saturation": 1.0,
    "curve_enabled": False, "curve_black": 0, "curve_shadows": 64, "curve_mid": 128, "curve_highlights": 192, "curve_white": 255,
    "texture": 0, "clarity": 0, "denoise": 0, "sharpen": False,
    "portrait_smoothing": 0.0,
    "mask_target_exposure": 1.0, "mask_target_contrast": 1.0, "mask_target_saturation": 1.0,
    "crop_aspect": "Original",
}


def _build_recipe() -> dict:
    """Capture every editable control's current value into one JSON-serializable dict: used both
    for Undo/Redo snapshots and for the recipe saved inside a .photoproject file. The uploaded
    layer overlay image is deliberately NOT included here (see the Layers tab and the Project File
    section) -- it is saved with the project file, but kept out of per-step Undo/Redo history so
    that history snapshots stay small.
    """
    recipe = {key: st.session_state.get(widget_key, RECIPE_DEFAULTS[key]) for key, widget_key in RECIPE_WIDGET_KEYS.items()}
    recipe["mask_layers"] = st.session_state.get("mask_layers", [])
    return recipe


def _restore_recipe_to_state(recipe: dict) -> None:
    """Write a saved recipe (from Undo/Redo or a loaded project) into the session_state keys the
    corresponding widgets read their initial value from.

    Streamlit forbids writing to a widget's session_state key after that widget has already been
    instantiated in the current script run. So this must only be called before ANY tab widget below
    is created -- in practice, right at the top of the very next run after a st.rerun() triggered by
    Undo, Redo, or loading a project (see the "_pending_restore" handling right after the page
    style, before any tab is built).
    """
    for key, widget_key in RECIPE_WIDGET_KEYS.items():
        if key in recipe:
            st.session_state[widget_key] = recipe[key]
    if "mask_layers" in recipe:
        st.session_state.mask_layers = recipe["mask_layers"]
    layer_payload = recipe.get("layer")
    if layer_payload:
        st.session_state["restored_layer_bytes"] = base64.b64decode(layer_payload["image_b64"])
        st.session_state["l-opac"] = layer_payload.get("opacity", 1.0)
        st.session_state["l-blend"] = layer_payload.get("blend", "normal")


@st.cache_data(show_spinner=False)
def _cached_process_via_backend(image_bytes: bytes, filename: str, options: dict, local_mask_bytes: bytes | None, layer_bytes: bytes | None):
    """Cached so that reruns triggered by controls that don't affect the pipeline (zoom, Show
    Original, etc.) don't repeat a full HTTP round trip with the whole image."""
    return process_via_backend(image_bytes, filename, options, local_mask_bytes, layer_bytes)


st.set_page_config(page_title="Photo Studio Pro", page_icon="🎨", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* Hide Streamlit default header/footer to prevent cutting off the top */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Clean padding and tight vertical layout */
.block-container {
    max-width: 1800px;
    padding: 1.2rem 1.5rem 1.5rem !important;
}
[data-testid="stVerticalBlock"] { gap: 0.35rem; }

/* Top Header Bar */
.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #e2e8f0;
    padding: 0.2rem 0 0.5rem 0;
    margin-bottom: 0.6rem;
}
.brand-title {
    font: 800 1.5rem/1.1 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.brand-badge {
    background: linear-gradient(135deg, #ea580c, #c2410c);
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.header-info {
    font-size: 0.8rem;
    font-weight: 600;
    color: #475569;
    background: #f8fafc;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    border: 1px solid #cbd5e1;
}

/* Tool Panel & Canvas Panel styling */
.panel-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.panel-header-title {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.35rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Image Canvas */
.canvas-container {
    background: #090d16;
    border-radius: 8px;
    padding: 0.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid #1e293b;
    margin-bottom: 0.35rem;
}
.canvas-container img {
    max-width: 100%;
    max-height: 60vh;
    object-fit: contain;
    border-radius: 4px;
}

/* Export Panel */
.export-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.55rem 0.85rem;
}

/* Workspace tab styling */
button[data-baseweb="tab"] {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    padding: 0.35rem 0.6rem !important;
}
</style>
""", unsafe_allow_html=True)

# Top Header
st.markdown("""
<div class="top-header">
    <div class="brand-title">
        <span>🎨 Photo Studio</span>
        <span class="brand-badge">CPU Pro</span>
    </div>
    <div class="header-info">⚡ Professional Photo Workstation</div>
</div>
""", unsafe_allow_html=True)

# Apply any pending Undo/Redo or Load-Project restoration. This MUST run before any tab widget
# below is created (see _restore_recipe_to_state's docstring) -- it only ever fires on the rerun
# immediately triggered by clicking Undo, Redo, or loading a project.
if "_pending_restore" in st.session_state:
    _restore_recipe_to_state(st.session_state.pop("_pending_restore"))

# File loading section
upload_col1, upload_col2 = st.columns([2.5, 1], gap="small")
with upload_col1:
    uploaded = st.file_uploader(
        "Open Image",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff", "cr2", "cr3", "nef", "arw", "dng"],
        help="Open image (JPG, PNG, WEBP, TIFF, or Camera RAW). Original is never overwritten.",
        label_visibility="collapsed",
    )
with upload_col2:
    st.caption("📂 Open any JPG, PNG, WEBP, TIFF, or camera RAW file.")

with st.expander("📁 Open a saved project (.photoproject)"):
    project_upload = st.file_uploader("Project file", type=["photoproject", "zip"], key="proj-upload")
    if project_upload and st.button("Load Project", key="proj-load-btn"):
        loaded_project = ProjectBundle.load(project_upload.getvalue())
        st.session_state["restored_source_bytes"] = loaded_project.source_bytes
        st.session_state["restored_source_name"] = loaded_project.source_name
        # ProjectBundle stores the overlay layer in its own structured `layer_workspace` field
        # (a serialized LayerWorkspace), not base64-embedded in the recipe like the old
        # PhotoProject format -- translate it into the same recipe["layer"] shape
        # _restore_recipe_to_state() already knows how to apply, so that function needs no changes.
        pending_recipe = dict(loaded_project.recipe)
        workspace = loaded_project.layer_workspace_object()
        if workspace and len(workspace.layers) > 1:
            overlay_layer = workspace.layers[1]
            overlay_buffer = io.BytesIO()
            overlay_layer.image.save(overlay_buffer, format="PNG")
            pending_recipe["layer"] = {
                "image_b64": base64.b64encode(overlay_buffer.getvalue()).decode("ascii"),
                "opacity": overlay_layer.opacity,
                "blend": overlay_layer.blend_mode,
            }
        st.session_state["_pending_restore"] = pending_recipe
        st.rerun()

if not uploaded and "restored_source_bytes" not in st.session_state:
    st.info("👆 Upload an image above (or load a saved project) to open the editing workspace.")
    st.stop()

# A freshly uploaded file always takes priority; otherwise fall back to the image bundled inside a
# loaded .photoproject so re-opening a project doesn't also require re-uploading its source image.
upload_name = uploaded.name if uploaded else st.session_state["restored_source_name"]
upload_bytes = uploaded.getvalue() if uploaded else st.session_state["restored_source_bytes"]

# Load image safely. PhotoProcessor.from_bytes() routes camera RAW files (.cr2/.cr3/.nef/.arw/.dng)
# through rawpy via app.raw_import, and falls back to a plain PIL open for every other format --
# using it here (instead of Image.open() directly) is what makes the RAW types advertised in the
# uploader above actually work instead of failing with an unreadable-file error.
try:
    processor = PhotoProcessor.from_bytes(upload_bytes, upload_name)
except RuntimeError as error:
    st.error(f"⚠️ {error}")
    st.stop()
source = processor.image

# The actual edit pipeline runs on app/backend.py, not in this process (see docs/PROJECT_STATUS.md);
# `processor`/`source` above are only used locally for decoding, mask geometry, and previews.
try:
    ensure_backend_running()
except RuntimeError as error:
    st.error(f"⚠️ {error}")
    st.stop()

# Main Two Columns: Level & Parallel Layout
tools_col, canvas_col = st.columns([1.1, 1.4], gap="large")

with tools_col:
    st.markdown('<div class="panel-header-title">🛠️ Tool Workspace</div>', unsafe_allow_html=True)
    
    tab_tone, tab_color, tab_detail, tab_layers, tab_masks, tab_portrait, tab_crop, tab_presets, tab_more = st.tabs([
        "☼ Tone", "🎨 Color", "🔬 Detail", "◈ Layers", "◉ Masks", "◌ Portrait", "⛶ Crop", "✦ Presets", "☰ More"
    ])

    show_original = False
    compare_mode = False
    zoom_level = 1.0
    pan_x = 0.5
    pan_y = 0.5
    edited_image = source
    layer_bytes = None
    l_opacity = 1.0
    l_blend = "normal"
    aspect_code = "Original"
    crop_box = None

    # TAB 1: TONE & EXPOSURE
    with tab_tone:
        preset_choice = st.selectbox("Preset Look", ["None", *sorted(PRESETS)], key="tone-preset")
        exp_val = st.slider("Exposure", 0.5, 1.5, 1.0, 0.01, key="t-exp")
        cont_val = st.slider("Contrast", 0.5, 1.5, 1.0, 0.01, key="t-cont")
        high_val = st.slider("Highlights", -100, 100, 0, key="t-high")
        shad_val = st.slider("Shadows", -100, 100, 0, key="t-shad")

    # TAB 2: COLOR & CURVES
    with tab_color:
        temp_val = st.slider("Temperature (Cool/Warm)", -100, 100, 0, key="c-temp")
        tint_val = st.slider("Tint (Green/Magenta)", -100, 100, 0, key="c-tint")
        sat_val = st.slider("Saturation", 0.0, 2.0, 1.0, 0.01, key="c-sat")
        
        with st.expander("RGB Tone Curve", expanded=False):
            use_curve = st.checkbox("Enable RGB Curve", value=False, key="c-curve-en")
            c_black = st.slider("Blacks", 0, 64, 0, key="c-p0")
            c_low = st.slider("Shadows", 0, 128, 64, key="c-p1")
            c_mid = st.slider("Midtones", 64, 192, 128, key="c-p2")
            c_high = st.slider("Highlights", 128, 255, 192, key="c-p3")
            c_white = st.slider("Whites", 192, 255, 255, key="c-p4")

    # TAB 3: DETAIL & SHARPNESS
    with tab_detail:
        text_val = st.slider("Texture", -100, 100, 0, key="d-text")
        clar_val = st.slider("Clarity", -100, 100, 0, key="d-clar")
        denoise_val = st.slider("Denoise", 0, 9, 0, step=2, key="d-den")
        sharp_en = st.checkbox("Sharpen (Unsharp Mask)", value=False, key="d-sharp")

    # TAB 4: LAYERS & COMPOSITING
    with tab_layers:
        st.markdown("**Add Image Layer**")
        layer_file = st.file_uploader("Layer File", type=["png", "jpg", "jpeg", "webp"], key="l-file")
        # A freshly uploaded layer wins; otherwise fall back to the overlay bundled inside a loaded
        # .photoproject, so re-opening a project restores its layer without asking for a re-upload.
        if layer_file:
            layer_bytes = layer_file.getvalue()
        elif "restored_layer_bytes" in st.session_state:
            layer_bytes = st.session_state["restored_layer_bytes"]
        if layer_bytes:
            l_opacity = st.slider("Opacity", 0.0, 1.0, 1.0, 0.01, key="l-opac")
            l_blend = st.selectbox("Blend Mode", ["normal", "multiply", "screen"], key="l-blend")
            st.caption("This layer is composited on the processing backend when the image renders.")

    # TAB 5: MASKS & LOCAL ADJUSTMENTS
    with tab_masks:
        st.markdown("**Mask Layers**")
        st.caption("Build several named selections and combine them into one mask (union). The target adjustments below apply only where the combined mask is active.")

        if "mask_layers" not in st.session_state:
            st.session_state.mask_layers = []

        def _build_mask_image(kind: str, params: dict):
            if kind == "Luminance":
                return luminance_mask(source, params["low"], params["high"])
            if kind == "Color":
                return color_mask(source, params["color"], params["tolerance"])
            if kind == "Radial":
                return radial_mask(source.size)
            return gradient_mask(source.size, "horizontal" if "Horizontal" in kind else "vertical")

        # --- Add a new mask layer ---
        with st.expander("➕ Add a mask layer", expanded=not st.session_state.mask_layers):
            new_kind = st.selectbox(
                "Selection Type", ["Luminance", "Color", "Horizontal gradient", "Vertical gradient", "Radial"], key="m-new-kind",
            )
            new_params: dict = {}
            if new_kind == "Luminance":
                new_params["low"] = st.slider("Min Luma", 0, 255, 0, key="m-new-lmin")
                new_params["high"] = st.slider("Max Luma", 0, 255, 255, key="m-new-lmax")
            elif new_kind == "Color":
                new_color_hex = st.color_picker("Target Color", "#806450", key="m-new-color")
                new_params["color"] = tuple(int(new_color_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
                new_params["tolerance"] = st.slider("Color Tolerance", 0, 255, 50, key="m-new-ctol")
            new_soften = st.slider("Soften edges (px)", 0, 40, 0, key="m-new-soften")
            new_invert = st.checkbox("Invert selection", key="m-new-invert")
            new_name = st.text_input("Layer name", value=f"{new_kind} {len(st.session_state.mask_layers) + 1}", key="m-new-name")
            if st.button("Add mask layer", key="m-add-btn"):
                st.session_state.mask_layers.append(
                    {
                        "name": new_name.strip() or new_kind,
                        "kind": new_kind,
                        "params": new_params,
                        "soften": new_soften,
                        "invert": new_invert,
                        "opacity": 1.0,
                        "enabled": True,
                    }
                )
                st.rerun()

        # --- Existing mask layers: edit, toggle, or remove ---
        mask_stack = MaskStack(source.size)
        for idx, layer in enumerate(st.session_state.mask_layers):
            status_icon = "✅" if layer["enabled"] else "⬜"
            with st.expander(f"{status_icon} {layer['name']} — {layer['kind']}", expanded=False):
                layer["name"] = st.text_input("Name", layer["name"], key=f"m-name-{idx}")
                layer["enabled"] = st.checkbox("Enabled", layer["enabled"], key=f"m-en-{idx}")
                layer["invert"] = st.checkbox("Invert", layer["invert"], key=f"m-inv-{idx}")
                layer["opacity"] = st.slider("Opacity", 0.0, 1.0, layer["opacity"], 0.05, key=f"m-op-{idx}")
                layer["soften"] = st.slider("Soften edges (px)", 0, 40, layer.get("soften", 0), key=f"m-sf-{idx}")
                if st.button("🗑️ Remove layer", key=f"m-rm-{idx}"):
                    st.session_state.mask_layers.pop(idx)
                    st.rerun()
            base_mask = _build_mask_image(layer["kind"], layer["params"])
            if layer["soften"]:
                base_mask = soften_mask(base_mask, layer["soften"])
            mask_stack.add(MaskLayer(layer["name"], base_mask, layer["enabled"], layer["invert"], layer["opacity"]))

        # Union of every enabled layer (or an all-zero, no-op mask when there are none).
        combined_mask = mask_stack.combined()

        if not st.session_state.mask_layers:
            st.caption("No mask layers yet — the sliders below have no effect until you add one.")
        elif not any(layer["enabled"] for layer in st.session_state.mask_layers):
            st.caption("All mask layers are disabled — the sliders below have no effect.")

        if st.checkbox("👁️ Preview combined mask overlay", key="m-preview-overlay") and st.session_state.mask_layers:
            # Overlaid on the untouched source -- the backend-processed preview isn't available
            # locally, so this debug view shows the selection rather than the live edited result.
            st.image(apply_mask_overlay(source, combined_mask), caption="Red = selected by the combined mask", use_container_width=True)

        m_exp = st.slider("Target Exposure", 0.5, 1.5, 1.0, 0.01, key="m-texp")
        m_cont = st.slider("Target Contrast", 0.5, 1.5, 1.0, 0.01, key="m-tcont")
        m_sat = st.slider("Target Saturation", 0.0, 2.0, 1.0, 0.01, key="m-tsat")

    # TAB 6: PORTRAIT RETOUCHING
    with tab_portrait:
        st.markdown("**Face & Skin Retouching**")
        # Defaults to 0 (no change) so skin smoothing is opt-in, not silently forced onto every
        # export just because this tab exists in the workspace.
        p_smooth = st.slider(
            "Skin Smoothing", 0.0, 1.0, 0.0, 0.01, key="port-sm",
            help="0 leaves skin untouched. Increase to smooth detected skin tones.",
        )
        detected_faces = detect_faces(source)
        st.caption(f"👤 Detected faces: {len(detected_faces)}")

    # TAB: CROP & ASPECT RATIO
    with tab_crop:
        st.markdown("**Crop / Aspect Ratio**")
        crop_choice = st.selectbox("Aspect Ratio", ["Original", "Free-form", *ASPECT_RATIOS], key="crop-aspect")
        if crop_choice == "Free-form":
            st.caption("Drag the handles to choose an arbitrary crop box.")
            src_w, src_h = source.size
            # A slider's stored session_state value can't exceed the bounds passed to it -- if the
            # user opened a smaller image after setting these against a larger one, drop the stale
            # value so the widget falls back to its full-frame default instead of raising.
            for axis_key, bound in (("crop-x", src_w), ("crop-y", src_h)):
                stored = st.session_state.get(axis_key)
                if stored and (stored[0] > bound or stored[1] > bound):
                    del st.session_state[axis_key]
            x_range = st.slider("Horizontal (X)", 0, src_w, (0, src_w), key="crop-x")
            y_range = st.slider("Vertical (Y)", 0, src_h, (0, src_h), key="crop-y")
            if x_range[1] - x_range[0] < 1 or y_range[1] - y_range[0] < 1:
                st.warning("Select a non-empty region.")
            else:
                crop_box = (x_range[0], y_range[0], x_range[1], y_range[1])
                preview = source.copy()
                ImageDraw.Draw(preview).rectangle(crop_box, outline=(255, 60, 40), width=max(2, src_w // 200))
                st.image(preview, caption=f"Crop preview: {crop_box[2] - crop_box[0]}×{crop_box[3] - crop_box[1]}px", use_container_width=True)
        elif crop_choice != "Original":
            crop_w_ratio, crop_h_ratio = ASPECT_RATIOS[crop_choice]
            st.caption(f"Center-cropping to {crop_choice} ({crop_w_ratio}:{crop_h_ratio}).")
            aspect_code = f"{crop_w_ratio}:{crop_h_ratio}"
        else:
            st.caption("No crop applied — showing the full frame.")

    # TAB 7: PRESETS
    with tab_presets:
        st.markdown("**Preset Library**")
        p_store = PresetStore()
        st.caption(f"Saved: {', '.join(p_store.names()) if p_store.names() else 'None'}")
        p_new_name = st.text_input("New Preset Name", key="p-name-in")
        if st.button("Save Current Look as Preset", key="p-save-btn"):
            if p_new_name.strip():
                p_store.save(p_new_name.strip(), {
                    "exposure": exp_val,
                    "contrast": cont_val,
                    "saturation": sat_val,
                    "temperature": temp_val,
                    "tint": tint_val,
                })
                st.success(f"Preset '{p_new_name}' saved!")

        st.markdown("---")
        st.markdown("**Edit History**")
        if "edit_history" not in st.session_state:
            st.session_state.edit_history = EditHistory()
        history = st.session_state.edit_history
        history.record(_build_recipe())
        hist_c1, hist_c2 = st.columns(2)
        with hist_c1:
            if st.button("↶ Undo", key="hist-undo-btn", disabled=not history.can_undo, use_container_width=True):
                st.session_state["_pending_restore"] = history.undo() or {}
                st.rerun()
        with hist_c2:
            if st.button("↷ Redo", key="hist-redo-btn", disabled=not history.can_redo, use_container_width=True):
                st.session_state["_pending_restore"] = history.redo() or {}
                st.rerun()

        st.markdown("---")
        st.markdown("**Project File**")
        st.caption("Save every adjustment (plus the source image and any layer) into one file you can reopen later.")
        if st.button("💾 Save Project", key="proj-save-btn"):
            project_recipe = _build_recipe()
            layer_workspace_json = None
            if layer_bytes:
                overlay = Image.open(io.BytesIO(layer_bytes)).convert("RGBA")
                workspace = LayerWorkspace(source).add(WorkspaceLayer("Overlay", overlay, l_opacity, l_blend))
                layer_workspace_json = workspace.to_json()
            project = ProjectBundle(source_name=upload_name, source_bytes=upload_bytes, recipe=project_recipe, layer_workspace=layer_workspace_json)
            proj_buf = io.BytesIO()
            project.save(proj_buf)
            st.download_button(
                "⬇️ Download .photoproject",
                proj_buf.getvalue(),
                file_name=f"{Path(upload_name).stem}.photoproject",
                mime="application/zip",
                key="proj-dl-btn",
            )

    # TAB 8: MORE (Catalog, Batch, System)
    with tab_more:
        more_mode = st.radio("Tool", ["Library Catalog", "Batch Processor", "System Health"], horizontal=True, label_visibility="collapsed")
        if more_mode == "Library Catalog":
            lib_folder = st.text_input("Folder", str(Path.home() / "Pictures"), key="lib-f-in")
            lib_db = Path.home() / ".photo-studio" / "catalog.sqlite"
            if st.button("Scan Folder", key="lib-btn"):
                st.session_state.records = index_folder(lib_folder, lib_db, True)
                st.success("Indexed!")
            rec = st.session_state.get("records", [])
            q = st.text_input("Search", key="lib-q-in")
            r_min = st.slider("Min Stars", 0, 5, 0, key="lib-r-in")
            st.dataframe(PhotoCatalog(lib_db).search(q, r_min) if rec else [], use_container_width=True)
        elif more_mode == "Batch Processor":
            b_uploaded = st.file_uploader("Batch Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="b-up")
            b_prof = st.selectbox("Profile", list(EXPORT_PROFILES), key="b-p-sel")
            if b_uploaded and st.button("Run Batch", key="b-run-btn"):
                b_root = Path.home() / ".photo-studio" / "batch-input"
                b_root.mkdir(parents=True, exist_ok=True)
                b_paths = []
                for f in b_uploaded:
                    t = b_root / f.name
                    t.write_bytes(f.getvalue())
                    b_paths.append(t)
                outs, rep = process_files(b_paths, None, b_prof)
                st.dataframe([{"source": x.source.name, "status": x.status, "error": x.error or ""} for x in rep], use_container_width=True)
                st.download_button("Download ZIP", make_zip(outs), "batch.zip", "application/zip", key="b-dl")
        else:
            from app.diagnostics import check_environment
            st.dataframe([{"check": c.name, "status": c.status, "message": c.message} for c in check_environment()], use_container_width=True)

    # Every tab above runs on every Streamlit rerun regardless of which tab is visible, and none of
    # them mutate `processor` anymore -- they only capture control values. The actual pixel pipeline
    # (Tone -> Color -> Detail -> Layers -> Masks -> Portrait -> Crop, matching the tab order above)
    # runs once, here, on app/backend.py -- see that module's process_image() for the mirrored order.
    options = {
        "exposure": exp_val,
        "contrast": cont_val,
        "highlights": high_val,
        "shadows": shad_val,
        "preset": preset_choice,
        "temperature": temp_val,
        "tint": tint_val,
        "saturation": sat_val,
        "curve": [(0, c_black), (64, c_low), (128, c_mid), (192, c_high), (255, c_white)] if use_curve else None,
        "texture": text_val,
        "clarity": clar_val,
        "denoise": denoise_val,
        "sharpen": sharp_en,
        "layer_opacity": l_opacity,
        "layer_blend": l_blend,
        "mask_exposure": m_exp,
        "mask_contrast": m_cont,
        "mask_saturation": m_sat,
        "portrait_smoothing": p_smooth,
        "aspect": aspect_code,
        "crop_box": list(crop_box) if crop_box else None,
    }

    local_mask_bytes = None
    if st.session_state.mask_layers:
        mask_buffer = io.BytesIO()
        combined_mask.save(mask_buffer, format="PNG")
        local_mask_bytes = mask_buffer.getvalue()

    try:
        edited_image = _cached_process_via_backend(upload_bytes, upload_name, options, local_mask_bytes, layer_bytes)
    except RuntimeError as error:
        st.error(f"⚠️ {error}")
        st.stop()

with canvas_col:
    # Canvas Header Bar
    top_c1, top_c2, top_c3 = st.columns([1.5, 1, 1], gap="small")
    with top_c1:
        st.markdown(f'<div class="panel-header-title">🖼️ Live View: <span>{upload_name} ({edited_image.width}×{edited_image.height})</span></div>', unsafe_allow_html=True)
    with top_c2:
        zoom_level = st.selectbox("Zoom View", [1.0, 0.5, 1.5, 2.0, 4.0], format_func=lambda z: f"🔍 {int(z*100)}%" if z>=1 else f"🔍 {int(z*100)}% (Fit)", key="cv-zoom", label_visibility="collapsed")
    with top_c3:
        show_original = st.checkbox("👁️ Original", value=False, key="cv-orig")
        compare_mode = st.checkbox("⇄ Split View", value=False, key="cv-split")

    if zoom_level > 1.0:
        pan_c1, pan_c2 = st.columns(2)
        with pan_c1:
            pan_x = st.slider("Pan Horizontal", 0.0, 1.0, 0.5, 0.01, key="cv-panx")
        with pan_c2:
            pan_y = st.slider("Pan Vertical", 0.0, 1.0, 0.5, 0.01, key="cv-pany")

    # Main Image Canvas
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    if show_original and compare_mode:
        split_l, split_r = st.columns(2)
        with split_l:
            st.image(preview_image(source), caption="Original", use_container_width=True)
        with split_r:
            st.image(zoom_preview(edited_image, zoom_level, (pan_x, pan_y)), caption="Edited Output", use_container_width=True)
    elif show_original:
        st.image(preview_image(source), caption="Original Untouched", use_container_width=True)
    else:
        st.image(zoom_preview(edited_image, zoom_level, (pan_x, pan_y)), caption="Current Edited Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Export Footer Bar
    st.markdown('<div class="export-panel">', unsafe_allow_html=True)
    exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 1.2], gap="small")
    with exp_c1:
        out_format = st.selectbox("Export Format", ["JPEG", "PNG", "WEBP", "TIFF"], key="exp-fmt", label_visibility="collapsed")
    with exp_c2:
        out_quality = st.slider("Quality", 50, 100, 92, key="exp-qual", disabled=out_format in ["PNG", "TIFF"], label_visibility="collapsed")
    with exp_c3:
        buf = io.BytesIO()
        f_name = {"JPEG": "JPEG", "PNG": "PNG", "WEBP": "WEBP", "TIFF": "TIFF"}[out_format]
        out_img = edited_image.convert("RGB") if f_name in ["JPEG", "WEBP"] else edited_image
        opts = {"format": f_name}
        if f_name in ["JPEG", "WEBP"]:
            opts["quality"] = out_quality
        out_img.save(buf, **opts)
        ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif"}[out_format]
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "TIFF": "image/tiff"}[out_format]
        st.download_button(
            label=f"💾 Save {out_format}",
            data=buf.getvalue(),
            file_name=f"{Path(upload_name).stem}-edited{ext}",
            mime=mime,
            use_container_width=True,
            key="exp-download-btn",
        )
    st.markdown('</div>', unsafe_allow_html=True)
