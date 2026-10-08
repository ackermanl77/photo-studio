"""The selection stack shared by the Masks and Repair panels, and its cached mask builders.

A selection is a plain dict in a session_state list (mask_layers / removal_mask_layers):
{"id", "name", "kind", "params", "soften", "invert", "opacity", "enabled", "combine", "adjust",
"adjustments"}. Its widgets are keyed by the selection's stable id, never its list index (a removed
selection would otherwise hand its cached widget state to the next one), and a mutation of the list
inside the render loop is followed by `break` (see CLAUDE.md).

Pixels are built from a selection list's signature (selections_signature) by cached builders, so a
rerun that changes nothing about the selections rebuilds nothing.
"""
from __future__ import annotations

import base64
import io
import json
import uuid

import streamlit as st
from PIL import Image

from app.advanced_masks import COMBINE_MODES, MaskLayer, MaskStack, color_mask, gradient_mask, luminance_mask, magic_wand_mask, radial_mask, soften_mask
from app.painted_masks import resolve_painted_mask
from app.professional_tools import preview_image
from app.ui.state import consume_new_commit, request_rerun, show_component
from app.ui_editors import _paint_editor_html, _wand_picker_html
from app.i18n import t, t_format, t_option

SELECTION_KINDS = ["Luminance", "Color", "Horizontal gradient", "Vertical gradient", "Radial", "Painted", "Magic Wand"]


def decode_painted_mask(mask_b64: str) -> Image.Image:
    """The paint editor draws solid white at varying alpha, so the alpha channel of the PNG it
    exports *is* the grayscale mask. Returned at the canvas's own (smaller) resolution:
    MaskLayer.normalized() resizes every mask kind to the source's size before use."""
    raw = base64.b64decode(mask_b64)
    return Image.open(io.BytesIO(raw)).convert("RGBA").split()[3]


def build_mask_image(source: Image.Image, kind: str, params: dict, painted: dict | None = None) -> Image.Image:
    if kind == "Luminance":
        return luminance_mask(source, params["low"], params["high"])
    if kind == "Color":
        return color_mask(source, params["color"], params["tolerance"])
    if kind == "Radial":
        # Center/radius/feather as fractions of the frame.
        return radial_mask(
            source.size, (float(params.get("center_x", 0.5)), float(params.get("center_y", 0.5))),
            float(params.get("radius", 0.5)), float(params.get("feather", 0.25)),
        )
    if kind == "Painted":
        return decode_painted_mask(resolve_painted_mask(params, painted))
    if kind == "Magic Wand":
        return magic_wand_mask(source, params["click_x"], params["click_y"], params["tolerance"], params.get("contiguous", True))
    return gradient_mask(source.size, "horizontal" if "Horizontal" in kind else "vertical")


def _new_selection(name: str, kind: str, params: dict, soften: int = 0, invert: bool = False) -> dict:
    return {"id": uuid.uuid4().hex, "name": name, "kind": kind, "params": params, "soften": soften, "invert": invert, "opacity": 1.0, "enabled": True}


def render_selection_stack(state_key: str, key_prefix: str, source: Image.Image, include_opacity: bool = True) -> str:
    """"Add a selection, then list/edit/remove the existing ones" -- the Masks panel (local
    adjustments) and the Repair panel (object removal) each call it with their own session_state
    list and widget-key prefix. Returns the signature of the selections that join the combined mask
    (selections_signature); the caller builds the mask from it (combined_mask_png)."""
    from app.painted_masks import intern_painted_mask

    if state_key not in st.session_state:
        st.session_state[state_key] = []
    layers = st.session_state[state_key]
    for layer in layers:
        layer.setdefault("id", uuid.uuid4().hex)  # selections saved before ids existed

    with st.expander(t("➕ Add a selection"), expanded=not layers):
        new_kind = st.selectbox(t("Selection Type"), SELECTION_KINDS, key=f"{key_prefix}-new-kind", format_func=t_option)
        if new_kind == "Magic Wand":
            new_tolerance = st.slider(t("Tolerance"), 1, 150, 40, key=f"{key_prefix}-wand-tol")
            new_contiguous = st.checkbox(t("Contiguous (connected region only)"), value=True, key=f"{key_prefix}-wand-contig")
            st.caption(t("Click a point on the photo below to select similar colors around it."))
            show_component(lambda: _wand_picker_html(preview_image(source, (480, 480)), key_prefix))
            with st.container(key=f"{key_prefix}-wand-hidden"):
                pending_wand = st.text_area(t("Magic wand click"), key=f"{key_prefix}-wand-data", label_visibility="collapsed")
            if consume_new_commit(pending_wand, f"{key_prefix}-wand-committed"):
                try:
                    click = json.loads(pending_wand)
                    click_x, click_y = float(click["x"]), float(click["y"])
                except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                    pass
                else:
                    params = {"click_x": click_x, "click_y": click_y, "tolerance": new_tolerance, "contiguous": new_contiguous}
                    layers.append(_new_selection(f"Magic Wand {len(layers) + 1}", "Magic Wand", params))
                    request_rerun()
        elif new_kind == "Painted":
            st.caption(t('Paint directly on the photo below — brush, radial, or gradient, plus an eraser. Click "Add to selections" inside the canvas when done.'))
            show_component(lambda: _paint_editor_html(preview_image(source, (480, 480)), key_prefix))
            with st.container(key=f"{key_prefix}-paint-hidden"):
                pending_paint = st.text_area(t("Painted mask data"), key=f"{key_prefix}-paint-data", label_visibility="collapsed")
            # A fresh editor iframe starts blank, so only a genuine new painting produces new data;
            # clicking "Add to selections" again without drawing is a no-op, not a duplicate.
            if consume_new_commit(pending_paint, f"{key_prefix}-paint-committed"):
                layers.append(_new_selection(f"Painted {len(layers) + 1}", "Painted", {"mask_ref": intern_painted_mask(pending_paint)}))
                request_rerun()
        else:
            new_params: dict = {}
            if new_kind == "Luminance":
                new_params["low"] = st.slider(t("Min Luma"), 0, 255, 0, key=f"{key_prefix}-new-lmin")
                new_params["high"] = st.slider(t("Max Luma"), 0, 255, 255, key=f"{key_prefix}-new-lmax")
            elif new_kind == "Color":
                new_color_hex = st.color_picker(t("Target Color"), "#806450", key=f"{key_prefix}-new-color")
                new_params["color"] = tuple(int(new_color_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
                new_params["tolerance"] = st.slider(t("Color Tolerance"), 0, 255, 50, key=f"{key_prefix}-new-ctol")
            elif new_kind == "Radial":
                radial_c1, radial_c2 = st.columns(2)
                new_params["center_x"] = radial_c1.slider(t("Center X"), 0.0, 1.0, 0.5, 0.01, key=f"{key_prefix}-new-rcx")
                new_params["center_y"] = radial_c2.slider(t("Center Y"), 0.0, 1.0, 0.5, 0.01, key=f"{key_prefix}-new-rcy")
                new_params["radius"] = radial_c1.slider(t("Radius"), 0.05, 1.5, 0.5, 0.01, key=f"{key_prefix}-new-rr")
                new_params["feather"] = radial_c2.slider(t("Feather"), 0.01, 1.0, 0.25, 0.01, key=f"{key_prefix}-new-rf")
            new_soften = st.slider(t("Soften edges (px)"), 0, 40, 0, key=f"{key_prefix}-new-soften")
            new_invert = st.checkbox(t("Invert selection"), key=f"{key_prefix}-new-invert")
            new_name = st.text_input(t("Layer name"), value=f"{new_kind} {len(layers) + 1}", key=f"{key_prefix}-new-name")
            if st.button(t("Add selection"), key=f"{key_prefix}-add-btn"):
                layers.append(_new_selection(new_name.strip() or new_kind, new_kind, new_params, new_soften, new_invert))
                request_rerun()

    for idx, layer in enumerate(layers):
        layer_id = layer["id"]
        status_icon = "✅" if layer["enabled"] else "⬜"
        with st.expander(t("{0} {1} — {2}").format(t(status_icon), t(layer['name']), t(layer['kind'])), expanded=False):
            layer["name"] = st.text_input(t("Name"), layer["name"], key=f"{key_prefix}-name-{layer_id}")
            layer["enabled"] = st.checkbox(t("Enabled"), layer["enabled"], key=f"{key_prefix}-en-{layer_id}")
            layer["invert"] = st.checkbox(t("Invert"), layer["invert"], key=f"{key_prefix}-inv-{layer_id}")
            if include_opacity:
                layer["opacity"] = st.slider(t("Opacity"), 0.0, 1.0, layer["opacity"], 0.05, key=f"{key_prefix}-op-{layer_id}")
            layer["soften"] = st.slider(t("Soften edges (px)"), 0, 40, layer.get("soften", 0), key=f"{key_prefix}-sf-{layer_id}")
            combine_choices = list(COMBINE_MODES)
            layer["combine"] = st.selectbox(
                t("Combine with the selections above"), combine_choices, index=combine_choices.index(layer.get("combine", "add")),
                format_func=t_format(lambda mode: {"add": "Add (union)", "subtract": "Subtract", "intersect": "Intersect"}[mode]),
                key=f"{key_prefix}-comb-{layer_id}", disabled=idx == 0 or layer.get("adjust", False),
            )
            if layer["kind"] == "Radial":
                params = layer["params"]
                edit_c1, edit_c2 = st.columns(2)
                params["center_x"] = edit_c1.slider(t("Center X"), 0.0, 1.0, float(params.get("center_x", 0.5)), 0.01, key=f"{key_prefix}-rcx-{layer_id}")
                params["center_y"] = edit_c2.slider(t("Center Y"), 0.0, 1.0, float(params.get("center_y", 0.5)), 0.01, key=f"{key_prefix}-rcy-{layer_id}")
                params["radius"] = edit_c1.slider(t("Radius"), 0.05, 1.5, float(params.get("radius", 0.5)), 0.01, key=f"{key_prefix}-rr-{layer_id}")
                params["feather"] = edit_c2.slider(t("Feather"), 0.01, 1.0, float(params.get("feather", 0.25)), 0.01, key=f"{key_prefix}-rf-{layer_id}")
            if include_opacity:
                # Per-selection adjustment set (B8): applied on its own with these values instead of
                # joining the combined mask the Target sliders drive.
                layer["adjust"] = st.checkbox(t("Own adjustments"), layer.get("adjust", False), key=f"{key_prefix}-own-{layer_id}")
                if layer["adjust"]:
                    adjustments = layer.setdefault("adjustments", {})
                    own_c1, own_c2 = st.columns(2)
                    adjustments["exposure_ev"] = own_c1.slider(t("Exposure (EV)"), -5.0, 5.0, float(adjustments.get("exposure_ev", 0.0)), 0.05, key=f"{key_prefix}-own-ev-{layer_id}")
                    adjustments["contrast"] = own_c2.slider(t("Contrast"), -100, 100, int(adjustments.get("contrast", 0)), key=f"{key_prefix}-own-ct-{layer_id}")
                    adjustments["saturation"] = own_c1.slider(t("Saturation"), -100, 100, int(adjustments.get("saturation", 0)), key=f"{key_prefix}-own-sat-{layer_id}")
                    adjustments["clarity"] = own_c2.slider(t("Clarity"), -100, 100, int(adjustments.get("clarity", 0)), key=f"{key_prefix}-own-cl-{layer_id}")
                    adjustments["temperature"] = own_c1.slider(t("Temperature"), -100, 100, int(adjustments.get("temperature", 0)), key=f"{key_prefix}-own-tmp-{layer_id}")
                    adjustments["tint"] = own_c2.slider(t("Tint"), -100, 100, int(adjustments.get("tint", 0)), key=f"{key_prefix}-own-tnt-{layer_id}")
            if layer["kind"] == "Painted":
                # The one kind whose shape can't be re-derived from numeric params -- a second pass
                # at the canvas needs its own toggle.
                if st.checkbox(t("🎨 Edit Painting"), key=f"{key_prefix}-editpaint-toggle-{layer_id}"):
                    edit_prefix = f"{key_prefix}-ep-{layer_id}"
                    show_component(lambda: _paint_editor_html(
                        preview_image(source, (480, 480)), edit_prefix,
                        preload_mask_b64=resolve_painted_mask(layer["params"]), commit_label="💾 Update Painting",
                    ))
                    with st.container(key=f"{edit_prefix}-paint-hidden"):
                        pending_edit = st.text_area(t("Edited painted mask data"), key=f"{edit_prefix}-paint-data", label_visibility="collapsed")
                    if consume_new_commit(pending_edit, f"{edit_prefix}-paint-committed"):
                        layer["params"] = {"mask_ref": intern_painted_mask(pending_edit)}
                        request_rerun()
            if st.button(t("🗑️ Remove"), key=f"{key_prefix}-rm-{layer_id}"):
                layers.pop(idx)
                request_rerun()
                break  # never keep iterating a list just mutated (see the module docstring)

    # Selections with their own adjustments are rendered separately (own_adjustment_layers).
    return selections_signature([layer for layer in layers if not layer.get("adjust")])


def selections_signature(layers: list[dict]) -> str:
    """Everything about a selection list that changes the combined mask, as one hashable string."""
    return json.dumps(
        [[layer["kind"], layer["params"], layer["soften"], layer["enabled"], layer["invert"], layer.get("opacity", 1.0), layer.get("combine", "add")] for layer in layers],
        sort_keys=True,
    )


def own_adjustment_layers(layers: list[dict]) -> list[dict]:
    """Enabled selections that carry their own adjustment set (the Masks panel's "Own adjustments")."""
    return [layer for layer in layers if layer.get("adjust") and layer.get("enabled", True)]


def build_selection_mask(mask_source: Image.Image, kind: str, params: dict, soften: int, painted: dict | None = None) -> Image.Image:
    base_mask = build_mask_image(mask_source, kind, params, painted)
    return soften_mask(base_mask, soften) if soften else base_mask


@st.cache_data(show_spinner=False, max_entries=24)
def _selection_mask(source_key: str, _mask_source: Image.Image, kind: str, params_json: str, soften: int) -> Image.Image:
    """One selection's full-resolution mask (before invert/opacity, which MaskStack applies),
    cached per (photo + geometry, kind, params, soften). A painted selection's params hold a
    content-hash reference (app.painted_masks), so the key changes exactly when the painting does."""
    return build_selection_mask(_mask_source, kind, json.loads(params_json), soften)


def _combine_selections(mask_source: Image.Image, signature: str, size, build) -> bytes:
    from app.ui.processing import png_bytes

    mask_stack = MaskStack(mask_source.size)
    for kind, params, soften, enabled, invert, opacity, combine in json.loads(signature):
        mask_stack.add(MaskLayer(kind, build(kind, params, soften), enabled, invert, opacity, combine))
    combined = mask_stack.combined()
    if size and combined.size != tuple(size):
        combined = combined.resize(tuple(size), Image.Resampling.BILINEAR)
    return png_bytes(combined)


@st.cache_data(show_spinner=False, max_entries=8)
def combined_mask_png(source_key: str, _mask_source: Image.Image, signature: str, size: tuple[int, int] | None = None) -> bytes:
    """The combination of every enabled selection in `signature`, PNG-encoded for the backend --
    at full resolution, or resized to `size` for a proxy preview. Cached on the signature."""
    return _combine_selections(
        _mask_source, signature, size,
        lambda kind, params, soften: _selection_mask(source_key, _mask_source, kind, json.dumps(params, sort_keys=True), soften),
    )


def combined_mask_png_uncached(mask_source: Image.Image, signature: str, painted: dict | None = None) -> bytes:
    """combined_mask_png() without st.cache_data, at full resolution -- for the export and refine
    producers, which run on a thread where st.* isn't available (`painted`: the session's
    painted-mask table, bound up front)."""
    return _combine_selections(mask_source, signature, None, lambda kind, params, soften: build_selection_mask(mask_source, kind, params, soften, painted))


def local_masks_payload(masks: list[bytes]) -> bytes:
    """/process's "local_masks" file: one PNG per local_adjustments entry, in order."""
    return json.dumps({"masks": [base64.b64encode(mask).decode("ascii") for mask in masks]}).encode("utf-8")
