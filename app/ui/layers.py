"""The Layers panel: any number of Image/Text/Shape/Path layers, each with its own name,
visibility, opacity, blend mode and Layer Styles, reorderable and duplicable -- and their
rasterization into the /process wire format (app.layer_workspace.layers_to_json).

A layer is a plain dict in st.session_state.image_layers, bottom-to-top (index 0 sits on the photo),
keyed by a stable id. Text/Shape/Path layers stay parametric and are rasterized on demand
(layer_dict_to_image); an Image layer stores its upload as base64 PNG.
"""
from __future__ import annotations

import base64
import io
import json
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

from app.layer_styles import apply_layer_styles
from app import fonts
from app.layer_workspace import WorkspaceLayer, layers_to_json, render_image_layer, render_text_layer
from app.major_features import BLEND_MODE_LABELS
from app.professional_tools import preview_image
from app.ui.state import consume_new_commit, hex_to_rgba, request_rerun, show_component, tab_visible
from app.ui_editors import _pen_editor_html
from app.vector_shapes import SHAPE_TYPES, render_path_layer, render_shape_layer
from app.i18n import t, t_format, t_option


_STYLE_EFFECT_PREFIXES = ("drop_shadow", "outer_glow", "stroke", "color_overlay")
_STYLE_COLOR_KEYS = ("drop_shadow_color", "outer_glow_color", "stroke_color", "color_overlay_color")


def resolved_layer_styles(styles: dict) -> dict:
    """`styles` stores its colors as `st.color_picker` hex strings (e.g. "#FF0000") -- convert to
    the plain RGB int tuples app.layer_styles.apply_layer_styles() expects, leaving every other
    field (enabled flags, blur/offset/width/opacity numbers) untouched."""
    resolved = dict(styles)
    for key in _STYLE_COLOR_KEYS:
        if key in resolved:
            resolved[key] = hex_to_rgba(resolved[key])[:3]
    return resolved


def scaled_styles(styles: dict, scale: float) -> dict:
    """Layer Style sizes are pixels of the full-resolution photo; a proxy render scales them."""
    if scale == 1.0:
        return styles
    scaled = dict(styles)
    for key in ("drop_shadow_offset_x", "drop_shadow_offset_y", "drop_shadow_blur", "outer_glow_blur", "stroke_width"):
        if key in scaled:
            scaled[key] = scaled[key] * scale
    return scaled


def layer_dict_to_image(layer: dict, canvas_size: tuple[int, int], scale: float = 1.0) -> Image.Image:
    """Rasterize one entry of st.session_state.image_layers into an actual RGBA image -- an
    "Image" layer decodes its stored upload (resized to the canvas's own size so Layer Styles below
    compute their blur/offset/width in the same real pixel scale for either kind); a "Text" layer
    renders fresh from its parametric fields every time (cheap: a single PIL draw call), the same
    "stay parametric, only rasterize on demand" approach app.advanced_masks.MaskLayer's selections
    already use, so editing the text after adding it doesn't require re-uploading anything. Shared
    by render_layer_stack() (the backend wire payload) and the Save Project handler so the two
    never diverge on how a layer's kind maps to pixels.

    Layer Styles (Drop Shadow/Outer Glow/Stroke/Color Overlay, see app.layer_styles) are applied
    here too, after the base content is rasterized and before it becomes a WorkspaceLayer -- so the
    normal opacity/blend-mode compositing on the resulting WorkspaceLayer runs on top of whatever
    the styles produce, unmodified.
    """
    if layer.get("kind") == "text":
        # font_size_pct: % of the canvas's short edge (F5) -- the same look on a 2 MP or 45 MP
        # photo and on a preview proxy. Older layers stored pixels (font_size).
        if "font_size_pct" in layer:
            font_size = float(layer["font_size_pct"]) / 100.0 * min(canvas_size)
        else:
            font_size = float(layer.get("font_size", 48)) * scale
        content = render_text_layer(
            canvas_size, layer.get("text", ""), font_size, hex_to_rgba(layer.get("color", "#FFFFFF")),
            layer.get("align", "left"), layer.get("x_pct", 50.0), layer.get("y_pct", 50.0), layer.get("font", ""),
        )
    elif layer.get("kind") == "shape":
        # fill_color/stroke_color are stored as st.color_picker hex strings, same UI convention as
        # a Text layer's "color" -- render_shape_layer() wants plain RGB int tuples.
        shape_params = {
            **layer,
            "fill_color": hex_to_rgba(layer.get("fill_color", "#FFFFFF"))[:3],
            "stroke_color": hex_to_rgba(layer.get("stroke_color", "#000000"))[:3],
            "stroke_width": max(1, round(layer.get("stroke_width", 4) * scale)),
            "line_thickness": max(1, round(layer.get("line_thickness", 6) * scale)),
        }
        content = render_shape_layer(canvas_size, shape_params)
    elif layer.get("kind") == "path":
        path_params = {
            **layer,
            "fill_color": hex_to_rgba(layer.get("fill_color", "#FFFFFF"))[:3],
            "stroke_color": hex_to_rgba(layer.get("stroke_color", "#FFFFFF"))[:3],
            "stroke_width": max(1, round(layer.get("stroke_width", 3) * scale)),
        }
        content = render_path_layer(canvas_size, path_params)
    else:
        content = Image.open(io.BytesIO(base64.b64decode(layer["image_b64"]))).convert("RGBA")
        if layer.get("stretch", "x_pct" not in layer):
            # Layers added before image layers had placement (and "Stretch to fill" on) fill the canvas.
            if content.size != canvas_size:
                content = content.resize(canvas_size, Image.Resampling.LANCZOS)
        else:
            content = render_image_layer(
                canvas_size, content, layer.get("x_pct", 50.0), layer.get("y_pct", 50.0),
                layer.get("scale_pct", 100.0), layer.get("rotation_deg", 0.0),
            )

    styles = layer.get("styles")
    if styles and any(styles.get(f"{prefix}_enabled") for prefix in _STYLE_EFFECT_PREFIXES):
        content = apply_layer_styles(content, resolved_layer_styles(scaled_styles(styles, scale)))
    return content


def render_layer_styles(layer: dict, layer_id: str) -> None:
    """Layer Styles sub-section inside one layer's own expander in render_layer_stack below
    (Drop Shadow/Outer Glow/Stroke/Color Overlay -- see app.layer_styles.apply_layer_styles, which
    this panel's values ultimately feed via resolved_layer_styles). Plain checkboxes + inline
    widget rows rather than a further st.expander: Streamlit doesn't allow nesting one expander
    inside another, and this already lives inside the layer's own."""
    layer.setdefault("styles", {})
    styles = layer["styles"]
    st.markdown(t("**Layer Styles**"))

    styles["drop_shadow_enabled"] = st.checkbox(
        t("Drop Shadow"), styles.get("drop_shadow_enabled", False), key=f"l-ds-en-{layer_id}",
        help=t("Casts a soft shadow behind this layer's own silhouette, offset and blurred below."),
    )
    if styles["drop_shadow_enabled"]:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            styles["drop_shadow_offset_x"] = st.slider(t("Offset X"), -50, 50, int(styles.get("drop_shadow_offset_x", 8)), key=f"l-ds-ox-{layer_id}", help=t("Horizontal shadow displacement in pixels -- negative shifts it left, positive right."))
        with c2:
            styles["drop_shadow_offset_y"] = st.slider(t("Offset Y"), -50, 50, int(styles.get("drop_shadow_offset_y", 8)), key=f"l-ds-oy-{layer_id}", help=t("Vertical shadow displacement in pixels -- negative shifts it up, positive down."))
        with c3:
            styles["drop_shadow_blur"] = st.slider(t("Blur"), 0, 40, int(styles.get("drop_shadow_blur", 6)), key=f"l-ds-bl-{layer_id}", help=t("How soft the shadow's edge is -- 0 is a hard-edged silhouette, higher spreads it into a soft blur."))
        with c4:
            styles["drop_shadow_color"] = st.color_picker(t("Color"), styles.get("drop_shadow_color", "#000000"), key=f"l-ds-col-{layer_id}")
        styles["drop_shadow_opacity"] = st.slider(t("Shadow Opacity"), 0.0, 1.0, float(styles.get("drop_shadow_opacity", 0.6)), 0.01, key=f"l-ds-op-{layer_id}", help=t("How dark/visible the shadow is, independent of the layer's own opacity."))

    styles["outer_glow_enabled"] = st.checkbox(
        t("Outer Glow"), styles.get("outer_glow_enabled", False), key=f"l-og-en-{layer_id}",
        help=t("Adds a soft halo of color radiating outward from this layer's silhouette."),
    )
    if styles["outer_glow_enabled"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            styles["outer_glow_blur"] = st.slider(t("Blur"), 0, 60, int(styles.get("outer_glow_blur", 10)), key=f"l-og-bl-{layer_id}", help=t("How far the glow spreads outward from the layer's edge."))
        with c2:
            styles["outer_glow_color"] = st.color_picker(t("Color"), styles.get("outer_glow_color", "#FFDC78"), key=f"l-og-col-{layer_id}")
        with c3:
            styles["outer_glow_opacity"] = st.slider(t("Glow Opacity"), 0.0, 1.0, float(styles.get("outer_glow_opacity", 0.75)), 0.01, key=f"l-og-op-{layer_id}", help=t("How strong/visible the glow is."))

    styles["stroke_enabled"] = st.checkbox(
        t("Stroke"), styles.get("stroke_enabled", False), key=f"l-st-en-{layer_id}",
        help=t("Outlines this layer's silhouette with a solid-color border."),
    )
    if styles["stroke_enabled"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            styles["stroke_width"] = st.slider(t("Width"), 1, 40, int(styles.get("stroke_width", 4)), key=f"l-st-w-{layer_id}", help=t("Border thickness in pixels."))
        with c2:
            styles["stroke_color"] = st.color_picker(t("Color"), styles.get("stroke_color", "#FFFFFF"), key=f"l-st-col-{layer_id}")
        with c3:
            styles["stroke_opacity"] = st.slider(t("Stroke Opacity"), 0.0, 1.0, float(styles.get("stroke_opacity", 1.0)), 0.01, key=f"l-st-op-{layer_id}", help=t("How solid/visible the outline is."))

    styles["color_overlay_enabled"] = st.checkbox(
        t("Color Overlay"), styles.get("color_overlay_enabled", False), key=f"l-co-en-{layer_id}",
        help=t("Tints this layer's visible pixels with a flat color, blended by the opacity below."),
    )
    if styles["color_overlay_enabled"]:
        c1, c2 = st.columns(2)
        with c1:
            styles["color_overlay_color"] = st.color_picker(t("Color"), styles.get("color_overlay_color", "#FF0000"), key=f"l-co-col-{layer_id}")
        with c2:
            styles["color_overlay_opacity"] = st.slider(t("Overlay Opacity"), 0.0, 1.0, float(styles.get("color_overlay_opacity", 0.6)), 0.01, key=f"l-co-op-{layer_id}", help=t("How strongly the color tints the layer -- 0 has no effect, 1 fully replaces its color."))


def font_select(label: str, current: str, key: str) -> str:
    """A Font choice: "" (the automatic default) plus every font app.fonts finds. A stored font this
    machine doesn't have stays selectable (marked), so opening a project doesn't silently drop it."""
    names = list(fonts.available_fonts())
    options = ["", *names] if not current or current in names else ["", current, *names]
    return st.selectbox(
        label, options, index=options.index(current) if current in options else 0, key=key,
        format_func=lambda name: t("Automatic (default)") if not name
        else (name if name in names else t("{0} (not installed)").format(name)),
    )


def font_upload_section() -> None:
    """"Add a font": install an uploaded .ttf/.otf once (it then appears in every Font choice)."""
    upload = st.file_uploader(
        t("Add a font (.ttf, .otf)"), type=["ttf", "otf", "ttc"], key="font-upload",
        help=t("The font is copied into the app's data folder and listed in every Font choice."),
    )
    if upload is None:
        return
    installed = st.session_state.setdefault("_fonts_installed", set())
    if upload.file_id in installed:  # a file_uploader keeps its file across reruns
        return
    installed.add(upload.file_id)
    try:
        name = fonts.install_font(upload.getvalue(), upload.name)
    except ValueError as error:
        st.error(t("⚠️ {0}").format(t(str(error))))
    else:
        st.toast(t("Font '{0}' added").format(name), icon="🔤")


def render_pen_path_editor(source: Image.Image, key_prefix: str, points: list[dict], closed: bool) -> tuple[list[dict], bool]:
    """The on-canvas Pen tool editor (_pen_editor_html) plus its Closed Path/Clear controls --
    shared between the "Add a layer" draft flow (building a brand new path before it becomes a
    layer) and an existing Path layer's own per-layer expander (refining it afterward). Returns the
    possibly-updated (points, closed); the caller is responsible for writing that back into
    wherever it actually lives (a draft session_state pair, or this specific layer's own dict),
    since this helper has no way to know which."""
    show_component(lambda: _pen_editor_html(preview_image(source, (480, 480)), key_prefix, points, closed))
    with st.container(key=f"{key_prefix}-hidden"):
        pending_points = st.text_area(t("Pen path points"), key=f"{key_prefix}-data", label_visibility="collapsed")
    if consume_new_commit(pending_points, f"{key_prefix}-committed"):
        try:
            parsed = json.loads(pending_points)
            points = [
                {
                    "x_pct": float(point["x_pct"]), "y_pct": float(point["y_pct"]),
                    "handle_dx_pct": float(point.get("handle_dx_pct", 0.0)), "handle_dy_pct": float(point.get("handle_dy_pct", 0.0)),
                }
                for point in parsed
            ]
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            pass
        else:
            request_rerun()
    closed = st.checkbox(t("Closed Path"), closed, key=f"{key_prefix}-closed")
    if points and st.button(t("🗑️ Clear Points"), key=f"{key_prefix}-clear"):
        points = []
        request_rerun()
    return points, closed


MAX_PSD_LAYERS = 64


def psd_layer_dicts(content: bytes, skip_bottom: bool) -> tuple[list[dict], int]:
    """A PSD's raster layers (app.psd, bottom first) as Image layer dicts for the Layers stack, each
    placed on the document's canvas and stretched to the photo's frame, plus how many layers had no
    pixels to import. `skip_bottom` leaves out the lowest layer -- usually the photo itself."""
    from app.psd import read_psd

    document = read_psd(content)
    psd_layers = document.layers[1:] if skip_bottom else document.layers
    dicts = []
    for psd_layer in psd_layers[:MAX_PSD_LAYERS]:
        buffer = io.BytesIO()
        psd_layer.on_canvas((document.width, document.height)).save(buffer, format="PNG")
        dicts.append({
            "id": uuid.uuid4().hex, "kind": "image", "name": psd_layer.name[:80] or "Layer",
            "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "x_pct": 50.0, "y_pct": 50.0, "scale_pct": 100.0, "rotation_deg": 0.0, "stretch": True,
            "opacity": round(psd_layer.opacity, 3), "blend_mode": psd_layer.blend_mode, "visible": psd_layer.visible,
        })
    return dicts, document.skipped


def import_psd_layers(layers: list[dict]) -> None:
    """➕ Add a layer → PSD (F22): every raster layer of a Photoshop file, once per uploaded file."""
    st.caption(t("Adds each layer of a Photoshop (.psd) file as an Image layer, with its name, opacity, blend mode and visibility. "
                 "Text, shapes and smart objects arrive rasterized; adjustment layers, layer effects and vector masks aren't imported."))
    skip_bottom = st.checkbox(t("Skip the bottom layer (the photo itself)"), value=True, key="l-psd-skip-bottom")
    psd_file = st.file_uploader(t("Photoshop file"), type=["psd"], key="l-new-psd")
    if psd_file is None:
        return
    file_id = getattr(psd_file, "file_id", psd_file.name)
    if st.session_state.get("l-last-added-psd-id") == file_id:
        return
    st.session_state["l-last-added-psd-id"] = file_id
    from app.psd import PsdError

    try:
        new_layers, skipped = psd_layer_dicts(psd_file.getvalue(), skip_bottom)
    except PsdError as error:
        st.error(t("⚠️ {0}").format(t(str(error))))
        return
    layers.extend(new_layers)
    message = t("Added {0} layer(s) from {1}.").format(len(new_layers), psd_file.name)
    if skipped:
        message += " " + t("{0} layer(s) without pixels (adjustments, empty layers) were skipped.").format(skipped)
    st.session_state["_psd_import_message"] = message
    request_rerun()


def render_layer_stack(source: Image.Image) -> str | None:
    """Layers tab: any number of stacked Image or Text layers, each with its own name/visibility/
    opacity/blend mode, reorderable and duplicable -- the same "add, then list/edit/remove" shape
    _render_selection_stack established for Masks/Repair. The compositing engine underneath
    (app.layer_workspace.LayerWorkspace) already supported an arbitrary number of layers; only this
    panel was ever limited to exactly one uploaded overlay image with no text option at all. Returns
    the stack's JSON signature for _layers_payload() (which builds process_via_backend's
    `layers_bytes`, app.layer_workspace.layers_to_json's wire format), or None when it's empty.

    Storage/composite order is bottom-to-top (index 0 sits directly on the photo, the last entry is
    on top) -- also the order layers are listed in below, called out in a caption since Photoshop's
    own panel lists the topmost layer first.
    """
    if "image_layers" not in st.session_state:
        st.session_state.image_layers = []
    layers = st.session_state.image_layers
    for layer in layers:
        layer.setdefault("id", uuid.uuid4().hex)
        layer.setdefault("kind", "image")  # backfill for layers added before Text layers existed

    with st.expander(t("➕ Add a layer"), expanded=not layers):
        new_layer_kind = st.radio(t("Layer type"), ["Image", "Text", "Shape", "Pen", "PSD"], horizontal=True, key="l-new-kind", format_func=t_option)
        if new_layer_kind == "PSD":
            import_psd_layers(layers)
        elif new_layer_kind == "Text":
            if st.button(t("Add Text Layer"), key="l-add-text-btn"):
                layers.append(
                    {
                        "id": uuid.uuid4().hex,
                        "kind": "text",
                        "name": f"Text {len(layers) + 1}",
                        "text": "Text",
                        "font": "",
                        "font_size_pct": 6.0,
                        "color": "#FFFFFF",
                        "align": "left",
                        "x_pct": 50.0,
                        "y_pct": 50.0,
                        "opacity": 1.0,
                        "blend_mode": "normal",
                        "visible": True,
                    }
                )
                request_rerun()
        elif new_layer_kind == "Shape":
            new_shape_type = st.selectbox(t("Shape"), SHAPE_TYPES, key="l-new-shape-type", format_func=t_option)
            if st.button(t("Add Shape Layer"), key="l-add-shape-btn"):
                layers.append(
                    {
                        "id": uuid.uuid4().hex,
                        "kind": "shape",
                        "name": f"{new_shape_type} {len(layers) + 1}",
                        "shape_type": new_shape_type,
                        "x_pct": 50.0,
                        "y_pct": 50.0,
                        "width_pct": 30.0,
                        "height_pct": 30.0,
                        "rotation_deg": 0.0,
                        "fill_enabled": True,
                        "fill_color": "#FFFFFF",
                        "fill_opacity": 1.0,
                        "stroke_enabled": False,
                        "stroke_color": "#000000",
                        "stroke_width": 4,
                        "sides": 5,
                        "inner_ratio": 0.5,
                        "line_thickness": 6,
                        "opacity": 1.0,
                        "blend_mode": "normal",
                        "visible": True,
                    }
                )
                request_rerun()
        elif new_layer_kind == "Pen":
            st.caption(t('Click points on the photo below to trace a path, then "Add Path Layer" when done.'))
            # draft_closed itself needs no separate write-back: it's the "l-path-draft-closed"
            # checkbox's own widget key, already persisted by Streamlit the moment that widget
            # renders -- re-assigning it here would violate the "never write into an
            # already-instantiated widget's session_state key" rule (see CLAUDE.md) and crash.
            # draft_points has no widget of its own (it's a plain list, like mask_layers), so it
            # does need this explicit write-back to survive into the next rerun.
            draft_points, draft_closed = render_pen_path_editor(
                source, "l-path-draft", st.session_state.get("l-path-draft-points", []), st.session_state.get("l-path-draft-closed", False),
            )
            st.session_state["l-path-draft-points"] = draft_points
            if st.button(t("➕ Add Path Layer"), key="l-add-path-btn", disabled=len(draft_points) < 2):
                layers.append(
                    {
                        "id": uuid.uuid4().hex,
                        "kind": "path",
                        "name": f"Path {len(layers) + 1}",
                        "points": draft_points,
                        "closed": draft_closed,
                        "fill_enabled": False,
                        "fill_color": "#FFFFFF",
                        "fill_opacity": 1.0,
                        "stroke_enabled": True,
                        "stroke_color": "#FFFFFF",
                        "stroke_width": 3,
                        "opacity": 1.0,
                        "blend_mode": "normal",
                        "visible": True,
                    }
                )
                # The next path starts empty and open. "l-path-draft-closed" is the Closed Path
                # checkbox's own, already instantiated widget key: it can't be assigned to (same
                # rule as above), but deleting it is allowed, and the checkbox then comes back
                # unchecked on the next run (its default is read with .get(..., False)).
                st.session_state["l-path-draft-points"] = []
                st.session_state.pop("l-path-draft-closed", None)
                request_rerun()
        else:
            new_layer_file = st.file_uploader(t("Layer image"), type=["png", "jpg", "jpeg", "webp"], key="l-new-file")
            # Streamlit keeps a file_uploader's value across reruns until the user picks a different
            # file or clears it -- adding on every rerun where it's truthy would append the same
            # layer over and over. file_id is a fresh identifier per upload, so this only fires once
            # per genuinely new file, the same "add exactly once per real commit" shape
            # consume_new_commit gives the on-canvas editors elsewhere in this file.
            if new_layer_file is not None:
                file_id = getattr(new_layer_file, "file_id", new_layer_file.name)
                if st.session_state.get("l-last-added-file-id") != file_id:
                    st.session_state["l-last-added-file-id"] = file_id
                    try:
                        decoded = Image.open(io.BytesIO(new_layer_file.getvalue())).convert("RGBA")
                    except OSError:
                        st.error(t("⚠️ Couldn't read that layer image — it may be corrupted."))
                    else:
                        buffer = io.BytesIO()
                        decoded.save(buffer, format="PNG")
                        layers.append(
                            {
                                "id": uuid.uuid4().hex,
                                "kind": "image",
                                "name": Path(new_layer_file.name).stem or f"Layer {len(layers) + 1}",
                                "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                                "x_pct": 50.0, "y_pct": 50.0, "scale_pct": 100.0, "rotation_deg": 0.0, "stretch": False,
                                "opacity": 1.0,
                                "blend_mode": "normal",
                                "visible": True,
                            }
                        )
                        request_rerun()

    psd_message = st.session_state.pop("_psd_import_message", None)
    if psd_message:
        st.success(psd_message)
    if not layers:
        st.caption(t("No layers yet — add one above."))
    else:
        st.caption(t("Listed bottom-to-top: the last layer here sits on top, same as the composite order."))

    blend_mode_choices = list(BLEND_MODE_LABELS)
    align_choices = ["left", "center", "right"]
    for idx, layer in enumerate(layers):
        layer_id = layer["id"]
        status_icon = "✅" if layer["visible"] else "⬜"
        kind_label = {"text": "Text", "shape": layer.get("shape_type", "Shape"), "path": "Path"}.get(layer["kind"], BLEND_MODE_LABELS[layer["blend_mode"]])
        with st.expander(t("{0} {1} — {2}").format(t(status_icon), t(layer['name']), t(kind_label)), expanded=False):
            # Sub-tabs inside the expander (allowed, unlike a nested expander). Every sub-tab body
            # still runs on every rerun, so the usual widget rules apply unchanged.
            tab_content, tab_styles, tab_settings = st.tabs([t("🎨 Content"), t("✨ Styles"), t("⚙️ Settings")])
            with tab_content:
                if layer["kind"] == "text":
                    layer["text"] = st.text_area(t("Text"), layer.get("text", "Text"), key=f"l-text-{layer_id}")
                    layer["font"] = font_select(t("Font"), layer.get("font", ""), key=f"l-font-{layer_id}")
                    text_col1, text_col2 = st.columns(2)
                    with text_col1:
                        if "font_size_pct" not in layer:  # a pixel size from an older layer -> % of the short edge
                            layer["font_size_pct"] = round(min(50.0, max(0.5, layer.get("font_size", 48) / min(source.size) * 100)), 1)
                        layer["font_size_pct"] = st.slider(
                            t("Font size (% of short edge)"), 0.5, 50.0, float(layer["font_size_pct"]), 0.5, key=f"l-fsizepct-{layer_id}",
                            help=t("Relative to the photo, so the text looks the same at any resolution."),
                        )
                    with text_col2:
                        layer["color"] = st.color_picker(t("Color"), layer.get("color", "#FFFFFF"), key=f"l-fcolor-{layer_id}")
                    layer["align"] = st.selectbox(
                        t("Alignment"), align_choices, index=align_choices.index(layer.get("align", "left")), key=f"l-falign-{layer_id}", format_func=t_option,
                    )
                    pos_col1, pos_col2 = st.columns(2)
                    with pos_col1:
                        layer["x_pct"] = st.slider(t("Horizontal position (%)"), 0.0, 100.0, float(layer.get("x_pct", 50.0)), key=f"l-fx-{layer_id}")
                    with pos_col2:
                        layer["y_pct"] = st.slider(t("Vertical position (%)"), 0.0, 100.0, float(layer.get("y_pct", 50.0)), key=f"l-fy-{layer_id}")
                    # A small live preview, computed locally (not through the backend) -- same
                    # "cheap enough to recompute every rerun" pattern as the Masks tab's combined
                    # overlay preview. A dark neutral background so light-colored text stays visible,
                    # since the layer's own image is otherwise fully transparent outside the glyphs.
                    preview_ratio = min(260 / source.width, 260 / source.height, 1.0)
                    preview_size = (max(1, int(source.width * preview_ratio)), max(1, int(source.height * preview_ratio)))
                    preview_bg = Image.new("RGBA", preview_size, (40, 40, 40, 255))
                    preview_text = render_text_layer(
                        preview_size, layer["text"], layer["font_size_pct"] / 100.0 * min(preview_size), hex_to_rgba(layer["color"]),
                        layer["align"], layer["x_pct"], layer["y_pct"], layer["font"],
                    )
                    if tab_visible():
                        st.image(Image.alpha_composite(preview_bg, preview_text))
                elif layer["kind"] == "shape":
                    layer["shape_type"] = st.selectbox(
                        t("Shape"), SHAPE_TYPES, index=SHAPE_TYPES.index(layer.get("shape_type", "Rectangle")), key=f"l-shp-type-{layer_id}", format_func=t_option,
                    )
                    shp_pos1, shp_pos2 = st.columns(2)
                    with shp_pos1:
                        layer["x_pct"] = st.slider(t("Horizontal position (%)"), 0.0, 100.0, float(layer.get("x_pct", 50.0)), key=f"l-shp-x-{layer_id}")
                    with shp_pos2:
                        layer["y_pct"] = st.slider(t("Vertical position (%)"), 0.0, 100.0, float(layer.get("y_pct", 50.0)), key=f"l-shp-y-{layer_id}")
                    shp_size1, shp_size2 = st.columns(2)
                    with shp_size1:
                        layer["width_pct"] = st.slider(t("Width (%)"), 1.0, 100.0, float(layer.get("width_pct", 30.0)), key=f"l-shp-w-{layer_id}")
                    with shp_size2:
                        layer["height_pct"] = st.slider(
                            t("Height (%)"), 1.0, 100.0, float(layer.get("height_pct", 30.0)), key=f"l-shp-h-{layer_id}", disabled=layer["shape_type"] == "Line",
                        )
                    layer["rotation_deg"] = st.slider(t("Rotation"), 0.0, 360.0, float(layer.get("rotation_deg", 0.0)), key=f"l-shp-rot-{layer_id}")
                    if layer["shape_type"] in ("Polygon", "Star"):
                        shp_sides1, shp_sides2 = st.columns(2)
                        with shp_sides1:
                            layer["sides"] = st.slider(
                                t("Sides" if layer["shape_type"] == "Polygon" else "Points"), 3, 12, int(layer.get("sides", 5)), key=f"l-shp-sides-{layer_id}",
                            )
                        if layer["shape_type"] == "Star":
                            with shp_sides2:
                                layer["inner_ratio"] = st.slider(t("Inner Radius"), 0.05, 0.95, float(layer.get("inner_ratio", 0.5)), key=f"l-shp-inner-{layer_id}")
                    if layer["shape_type"] == "Line":
                        shp_line1, shp_line2 = st.columns(2)
                        with shp_line1:
                            layer["line_thickness"] = st.slider(t("Thickness (px)"), 1, 60, int(layer.get("line_thickness", 6)), key=f"l-shp-thick-{layer_id}")
                        with shp_line2:
                            layer["fill_color"] = st.color_picker(t("Color"), layer.get("fill_color", "#FFFFFF"), key=f"l-shp-fcolor-{layer_id}")
                    else:
                        layer["fill_enabled"] = st.checkbox(t("Fill"), layer.get("fill_enabled", True), key=f"l-shp-fen-{layer_id}")
                        if layer["fill_enabled"]:
                            shp_fill1, shp_fill2 = st.columns(2)
                            with shp_fill1:
                                layer["fill_color"] = st.color_picker(t("Fill Color"), layer.get("fill_color", "#FFFFFF"), key=f"l-shp-fcolor-{layer_id}")
                            with shp_fill2:
                                layer["fill_opacity"] = st.slider(t("Fill Opacity"), 0.0, 1.0, float(layer.get("fill_opacity", 1.0)), 0.01, key=f"l-shp-fop-{layer_id}")
                        layer["stroke_enabled"] = st.checkbox(t("Outline"), layer.get("stroke_enabled", False), key=f"l-shp-sen-{layer_id}")
                        if layer["stroke_enabled"]:
                            shp_stroke1, shp_stroke2 = st.columns(2)
                            with shp_stroke1:
                                layer["stroke_color"] = st.color_picker(t("Outline Color"), layer.get("stroke_color", "#000000"), key=f"l-shp-scolor-{layer_id}")
                            with shp_stroke2:
                                layer["stroke_width"] = st.slider(t("Outline Width"), 1, 40, int(layer.get("stroke_width", 4)), key=f"l-shp-sw-{layer_id}")
                    # Same locally-computed live preview pattern as the Text branch above.
                    preview_ratio = min(260 / source.width, 260 / source.height, 1.0)
                    preview_size = (max(1, int(source.width * preview_ratio)), max(1, int(source.height * preview_ratio)))
                    preview_bg = Image.new("RGBA", preview_size, (40, 40, 40, 255))
                    shape_preview_params = {
                        **layer,
                        "fill_color": hex_to_rgba(layer.get("fill_color", "#FFFFFF"))[:3],
                        "stroke_color": hex_to_rgba(layer.get("stroke_color", "#000000"))[:3],
                    }
                    if tab_visible():
                        st.image(Image.alpha_composite(preview_bg, render_shape_layer(preview_size, shape_preview_params)))
                elif layer["kind"] == "path":
                    layer["points"], layer["closed"] = render_pen_path_editor(
                        source, f"l-path-{layer_id}", layer.get("points", []), layer.get("closed", False),
                    )
                    if layer["closed"]:
                        layer["fill_enabled"] = st.checkbox(t("Fill"), layer.get("fill_enabled", False), key=f"l-path-fen-{layer_id}")
                        if layer["fill_enabled"]:
                            path_fill1, path_fill2 = st.columns(2)
                            with path_fill1:
                                layer["fill_color"] = st.color_picker(t("Fill Color"), layer.get("fill_color", "#FFFFFF"), key=f"l-path-fcolor-{layer_id}")
                            with path_fill2:
                                layer["fill_opacity"] = st.slider(t("Fill Opacity"), 0.0, 1.0, float(layer.get("fill_opacity", 1.0)), 0.01, key=f"l-path-fop-{layer_id}")
                    layer["stroke_enabled"] = st.checkbox(t("Stroke"), layer.get("stroke_enabled", True), key=f"l-path-sen-{layer_id}")
                    if layer["stroke_enabled"]:
                        path_stroke1, path_stroke2 = st.columns(2)
                        with path_stroke1:
                            layer["stroke_color"] = st.color_picker(t("Stroke Color"), layer.get("stroke_color", "#FFFFFF"), key=f"l-path-scolor-{layer_id}")
                        with path_stroke2:
                            layer["stroke_width"] = st.slider(t("Stroke Width"), 1, 40, int(layer.get("stroke_width", 3)), key=f"l-path-sw-{layer_id}")
                else:
                    layer["stretch"] = st.checkbox(
                        t("Stretch to fill the photo"), layer.get("stretch", "x_pct" not in layer), key=f"l-img-stretch-{layer_id}",
                        help=t("Off: the image keeps its proportions and can be moved, scaled and rotated."),
                    )
                    if not layer["stretch"]:
                        img_c1, img_c2 = st.columns(2)
                        layer["x_pct"] = img_c1.slider(t("Horizontal position (%)"), 0.0, 100.0, float(layer.get("x_pct", 50.0)), key=f"l-img-x-{layer_id}")
                        layer["y_pct"] = img_c2.slider(t("Vertical position (%)"), 0.0, 100.0, float(layer.get("y_pct", 50.0)), key=f"l-img-y-{layer_id}")
                        layer["scale_pct"] = img_c1.slider(t("Scale (%)"), 1.0, 400.0, float(layer.get("scale_pct", 100.0)), key=f"l-img-scale-{layer_id}", help=t("100 % fits the image inside the photo."))
                        layer["rotation_deg"] = img_c2.slider(t("Rotation"), -180.0, 180.0, float(layer.get("rotation_deg", 0.0)), key=f"l-img-rot-{layer_id}")
                    if tab_visible():
                        thumb = Image.open(io.BytesIO(base64.b64decode(layer["image_b64"])))
                        st.image(preview_image(thumb, (160, 160)))
            with tab_styles:
                render_layer_styles(layer, layer_id)
            with tab_settings:
                layer["name"] = st.text_input(t("Name"), layer["name"], key=f"l-name-{layer_id}")
                layer["visible"] = st.checkbox(t("Visible"), layer["visible"], key=f"l-vis-{layer_id}")
                layer["opacity"] = st.slider(t("Opacity"), 0.0, 1.0, layer["opacity"], 0.01, key=f"l-op-{layer_id}")
                layer["blend_mode"] = st.selectbox(
                    t("Blend Mode"), blend_mode_choices, format_func=t_format(lambda mode: BLEND_MODE_LABELS[mode]),
                    index=blend_mode_choices.index(layer["blend_mode"]), key=f"l-blend-{layer_id}",
                )
                render_layer_mask_controls(layer, layer_id)
                # Every mutation below (reorder/duplicate/remove) is followed by `break`, not just
                # request_rerun() -- swapping or inserting into `layers` while an outer `enumerate()`
                # over that same list keeps iterating would re-visit the layer that just got swapped/
                # shifted into a later idx this same pass, rendering its widgets a second time under
                # the same id and raising StreamlitDuplicateElementKey. The deferred rerun (see
                # request_rerun()) redraws the rest of the list correctly on the very next pass.
                up_col, down_col, dup_col, rm_col = st.columns(4)
                with up_col:
                    if st.button(t("⬆️ Up"), key=f"l-up-{layer_id}", disabled=idx == len(layers) - 1):
                        layers[idx], layers[idx + 1] = layers[idx + 1], layers[idx]
                        request_rerun()
                        break
                with down_col:
                    if st.button(t("⬇️ Down"), key=f"l-down-{layer_id}", disabled=idx == 0):
                        layers[idx], layers[idx - 1] = layers[idx - 1], layers[idx]
                        request_rerun()
                        break
                with dup_col:
                    if st.button(t("📄 Duplicate"), key=f"l-dup-{layer_id}"):
                        layers.insert(idx + 1, {**layer, "id": uuid.uuid4().hex, "name": f"{layer['name']} copy"})
                        request_rerun()
                        break
                with rm_col:
                    if st.button(t("🗑️ Remove"), key=f"l-rm-{layer_id}"):
                        layers.pop(idx)
                        request_rerun()
                        break

    if not layers:
        return None
    # The stack's own JSON -- rasterized + serialized later (layers_payload, cached on it) at the
    # size the pipeline composites at: the full-resolution frame, or the preview proxy's. A layer
    # mask is carried as its selection's signature, so editing that selection re-renders the stack.
    return json.dumps([with_mask_signature(layer) for layer in layers], sort_keys=True)


def render_layer_mask_controls(layer: dict, layer_id: str) -> None:
    """Layer mask (F13): limit the layer to one of the Masks tab's selections (its own invert and
    opacity apply), optionally inverted -- the selection tools are shared rather than duplicated."""
    selections = {selection["id"]: selection.get("name", "Selection") for selection in st.session_state.get("mask_layers", [])}
    choices = [None, *selections]
    current = layer.get("mask_selection") if layer.get("mask_selection") in selections else None
    layer["mask_selection"] = st.selectbox(
        t("Layer mask"), choices, index=choices.index(current), key=f"l-mask-{layer_id}",
        format_func=t_format(lambda choice: "None (whole layer)" if choice is None else f"Selection: {selections[choice]}"),
        help=t("Show this layer only where a selection from ◉ Local → ◉ Masks is. Build the selection there; it can be one that isn't used for adjustments."),
    )
    layer["mask_invert"] = st.checkbox(t("Invert layer mask"), layer.get("mask_invert", False), key=f"l-mask-inv-{layer_id}", disabled=layer["mask_selection"] is None)
    if layer.get("mask_selection") and layer["mask_selection"] not in selections:
        layer["mask_selection"] = None


def with_mask_signature(layer: dict) -> dict:
    """A copy of `layer` carrying `mask_signature`: its mask selection's signature (enabled, on its
    own), or None for no mask or a selection that no longer exists."""
    from app.ui.selections import selections_signature

    selection = next((item for item in st.session_state.get("mask_layers", []) if item["id"] == layer.get("mask_selection")), None)
    signature = selections_signature([dict(selection, enabled=True, combine="add")]) if selection else None
    return {**layer, "mask_signature": signature}


def _layer_mask(layer: dict, canvas_size: tuple[int, int], mask_builder) -> Image.Image | None:
    if not layer.get("mask_signature") or mask_builder is None:
        return None
    mask = Image.open(io.BytesIO(mask_builder(layer["mask_signature"], tuple(canvas_size)))).convert("L")
    if mask.size != tuple(canvas_size):
        mask = mask.resize(tuple(canvas_size), Image.Resampling.BILINEAR)
    if layer.get("mask_invert"):
        mask = Image.eval(mask, lambda value: 255 - value)
    return mask


def build_layers_payload(signature: str, canvas_size: tuple[int, int], scale: float = 1.0, mask_builder=None) -> bytes:
    """Uncached layers_payload(), for the export/refine producers' thread. `mask_builder(selection
    signature, size) -> PNG bytes` renders layer masks (None: layers keep no mask)."""
    layers = json.loads(signature)
    workspace_layers = [
        WorkspaceLayer(
            layer["name"], layer_dict_to_image(layer, tuple(canvas_size), scale), layer["opacity"], layer["blend_mode"], layer["visible"],
            _layer_mask(layer, canvas_size, mask_builder),
        )
        for layer in layers
    ]
    return layers_to_json(workspace_layers).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=4)
def layers_payload(signature: str, canvas_size: tuple[int, int], scale: float = 1.0, source_key: str = "", _mask_source: Image.Image | None = None) -> bytes:
    """The Layers stack's /process payload at `canvas_size` (the full frame or the proxy's), cached
    on the stack's own JSON (layer masks' selection signatures included) and the photo's geometry
    key -- rasterizing (Layer Styles ~1 s per styled layer, a photo-sized image layer ~3 s) only
    happens when a layer actually changes."""
    from app.ui.selections import combined_mask_png

    mask_builder = None
    if _mask_source is not None:
        mask_builder = lambda mask_signature, size: combined_mask_png(source_key, _mask_source, mask_signature, size)  # noqa: E731
    return build_layers_payload(signature, canvas_size, scale, mask_builder)
