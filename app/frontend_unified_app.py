from __future__ import annotations

import base64
import io
import json
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
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
from app.presets import PRESETS, apply_preset
from app.preset_exchange import export_presets, import_presets
from app.portrait_tools import detect_faces
from app.professional_color import HSL_BAND_ORDER, ICC_TRANSFORM_ERRORS, gamut_warning_overlay
from app.lens_corrections import LENS_PROFILES, apply_lens_corrections
from app.professional_tools import EXPORT_PROFILES, apply_mask_overlay, compute_histogram, export_profile, preview_image, zoom_preview
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
    "crop_aspect": "crop-aspect", "removal_radius": "rm-radius",
    "lens_profile": "lens-profile", "lens_distortion": "lens-distortion",
    "lens_vignette_amount": "lens-vignette-amt", "lens_vignette_midpoint": "lens-vignette-mid",
    "lens_chromatic_aberration": "lens-ca",
}
RECIPE_DEFAULTS = {
    "exposure": 1.0, "contrast": 1.0, "highlights": 0, "shadows": 0, "preset": "None",
    "temperature": 0, "tint": 0, "saturation": 1.0,
    "curve_enabled": False, "curve_black": 0, "curve_shadows": 64, "curve_mid": 128, "curve_highlights": 192, "curve_white": 255,
    "texture": 0, "clarity": 0, "denoise": 0, "sharpen": False,
    "portrait_smoothing": 0.0,
    "mask_target_exposure": 1.0, "mask_target_contrast": 1.0, "mask_target_saturation": 1.0,
    "crop_aspect": "Original", "removal_radius": 3,
    "lens_profile": "None", "lens_distortion": 0, "lens_vignette_amount": 0, "lens_vignette_midpoint": 50,
    "lens_chromatic_aberration": 0,
}


def _hsl_widget_key(band: str, channel: str) -> str:
    """channel is one of "h"/"s"/"l" -- e.g. _hsl_widget_key("Orange", "s") -> "hsl-orange-s"."""
    return f"hsl-{band.lower()}-{channel}"


# Every band/channel combination participates in Undo/Redo/Save-Project exactly like every other
# control, via the same RECIPE_WIDGET_KEYS/RECIPE_DEFAULTS mechanism -- generated here instead of
# 24 hand-written entries since it's a direct product of HSL_BAND_ORDER x {hue, sat, lum}.
for _band in HSL_BAND_ORDER:
    for _channel, _suffix in (("hue", "h"), ("sat", "s"), ("lum", "l")):
        RECIPE_WIDGET_KEYS[f"hsl_{_band.lower()}_{_channel}"] = _hsl_widget_key(_band, _suffix)
        RECIPE_DEFAULTS[f"hsl_{_band.lower()}_{_channel}"] = 0


def _build_recipe() -> dict:
    """Capture every editable control's current value into one JSON-serializable dict: used both
    for Undo/Redo snapshots and for the recipe saved inside a .photoproject file. The uploaded
    layer overlay image is deliberately NOT included here (see the Layers tab and the Project File
    section) -- it is saved with the project file, but kept out of per-step Undo/Redo history so
    that history snapshots stay small.
    """
    recipe = {key: st.session_state.get(widget_key, RECIPE_DEFAULTS[key]) for key, widget_key in RECIPE_WIDGET_KEYS.items()}
    recipe["mask_layers"] = st.session_state.get("mask_layers", [])
    recipe["removal_mask_layers"] = st.session_state.get("removal_mask_layers", [])
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
    if "removal_mask_layers" in recipe:
        st.session_state.removal_mask_layers = recipe["removal_mask_layers"]
    layer_payload = recipe.get("layer")
    if layer_payload:
        st.session_state["restored_layer_bytes"] = base64.b64decode(layer_payload["image_b64"])
        st.session_state["l-opac"] = layer_payload.get("opacity", 1.0)
        st.session_state["l-blend"] = layer_payload.get("blend", "normal")


def _decode_painted_mask(mask_b64: str) -> Image.Image:
    """The on-canvas paint editor (_paint_editor_html) always draws in solid white at varying
    alpha -- alpha 0 = unpainted, alpha 255 = fully selected -- so the alpha channel of the PNG
    it exports *is* the grayscale mask. Deliberately returned at whatever (smaller-than-source)
    resolution the canvas was drawn at: MaskLayer.normalized() already resizes every mask kind to
    the source's size before use, so upscaling here too would be redundant work."""
    raw = base64.b64decode(mask_b64)
    return Image.open(io.BytesIO(raw)).convert("RGBA").split()[3]


def _build_mask_image(source: Image.Image, kind: str, params: dict) -> Image.Image:
    if kind == "Luminance":
        return luminance_mask(source, params["low"], params["high"])
    if kind == "Color":
        return color_mask(source, params["color"], params["tolerance"])
    if kind == "Radial":
        return radial_mask(source.size)
    if kind == "Painted":
        return _decode_painted_mask(params["mask_b64"])
    return gradient_mask(source.size, "horizontal" if "Horizontal" in kind else "vertical")


def _render_selection_stack(state_key: str, key_prefix: str, source: Image.Image, include_opacity: bool = True) -> MaskStack:
    """Shared "add a named selection (Luminance/Color/gradient/Radial), then list/edit/remove
    existing ones" UI used by both the Masks tab (local tone/color adjustments) and the Repair
    tab (object removal) -- same selection tools, a different session_state list and widget-key
    prefix per caller so the two tabs' widgets don't collide. Returns the combined MaskStack; the
    caller decides what to do with it (masked adjustments vs. inpainting).
    """
    if state_key not in st.session_state:
        st.session_state[state_key] = []
    layers = st.session_state[state_key]
    # Widget keys below are per-layer, keyed by this id rather than by list index -- a layer's
    # position shifts whenever an earlier one is removed, and an index-based key would then point
    # a later layer's checkbox/slider at the *previous occupant's* still-cached session_state
    # value instead of its own (a real bug found in review: removing layer B silently corrupted
    # layer C's enabled/soften values). setdefault() backfills an id for layers restored from a
    # project/recipe saved before this field existed.
    for layer in layers:
        layer.setdefault("id", uuid.uuid4().hex)

    with st.expander("➕ Add a selection", expanded=not layers):
        new_kind = st.selectbox(
            "Selection Type",
            ["Luminance", "Color", "Horizontal gradient", "Vertical gradient", "Radial", "Painted"],
            key=f"{key_prefix}-new-kind",
        )
        if new_kind == "Painted":
            st.caption('Paint directly on the photo below — brush, radial, or gradient, plus an eraser. Click "Add to selections" inside the canvas when done.')
            paint_html, paint_height = _paint_editor_html(preview_image(source, (480, 480)), key_prefix)
            components.html(paint_html, height=paint_height)
            with st.container(key=f"{key_prefix}-paint-hidden"):
                pending_paint = st.text_area("Painted mask data", key=f"{key_prefix}-paint-data", label_visibility="collapsed")
            # Streamlit forbids writing to a widget's session_state key after it's already been
            # instantiated this run (the same rule Undo/Redo/_pending_restore work around
            # elsewhere), so this can't just clear the field after reading it. Instead, remember
            # the last string actually turned into a layer and compare against that -- since a
            # fresh components.html() iframe always starts with a blank canvas, only a genuine new
            # commit (the user drew something and clicked "Add to selections" again) produces
            # different base64 data; clicking it twice without drawing anything new is correctly
            # a no-op rather than a duplicate layer.
            paint_committed_key = f"{key_prefix}-paint-committed"
            if pending_paint and pending_paint != st.session_state.get(paint_committed_key):
                layers.append(
                    {
                        "id": uuid.uuid4().hex,
                        "name": f"Painted {len(layers) + 1}",
                        "kind": "Painted",
                        "params": {"mask_b64": pending_paint},
                        "soften": 0,
                        "invert": False,
                        "opacity": 1.0,
                        "enabled": True,
                    }
                )
                st.session_state[paint_committed_key] = pending_paint
                _request_rerun()
        else:
            new_params: dict = {}
            if new_kind == "Luminance":
                new_params["low"] = st.slider("Min Luma", 0, 255, 0, key=f"{key_prefix}-new-lmin")
                new_params["high"] = st.slider("Max Luma", 0, 255, 255, key=f"{key_prefix}-new-lmax")
            elif new_kind == "Color":
                new_color_hex = st.color_picker("Target Color", "#806450", key=f"{key_prefix}-new-color")
                new_params["color"] = tuple(int(new_color_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
                new_params["tolerance"] = st.slider("Color Tolerance", 0, 255, 50, key=f"{key_prefix}-new-ctol")
            new_soften = st.slider("Soften edges (px)", 0, 40, 0, key=f"{key_prefix}-new-soften")
            new_invert = st.checkbox("Invert selection", key=f"{key_prefix}-new-invert")
            new_name = st.text_input("Layer name", value=f"{new_kind} {len(layers) + 1}", key=f"{key_prefix}-new-name")
            if st.button("Add selection", key=f"{key_prefix}-add-btn"):
                layers.append(
                    {
                        "id": uuid.uuid4().hex,
                        "name": new_name.strip() or new_kind,
                        "kind": new_kind,
                        "params": new_params,
                        "soften": new_soften,
                        "invert": new_invert,
                        "opacity": 1.0,
                        "enabled": True,
                    }
                )
                _request_rerun()

    mask_stack = MaskStack(source.size)
    for idx, layer in enumerate(layers):
        layer_id = layer["id"]
        status_icon = "✅" if layer["enabled"] else "⬜"
        with st.expander(f"{status_icon} {layer['name']} — {layer['kind']}", expanded=False):
            layer["name"] = st.text_input("Name", layer["name"], key=f"{key_prefix}-name-{layer_id}")
            layer["enabled"] = st.checkbox("Enabled", layer["enabled"], key=f"{key_prefix}-en-{layer_id}")
            layer["invert"] = st.checkbox("Invert", layer["invert"], key=f"{key_prefix}-inv-{layer_id}")
            if include_opacity:
                layer["opacity"] = st.slider("Opacity", 0.0, 1.0, layer["opacity"], 0.05, key=f"{key_prefix}-op-{layer_id}")
            layer["soften"] = st.slider("Soften edges (px)", 0, 40, layer.get("soften", 0), key=f"{key_prefix}-sf-{layer_id}")
            if st.button("🗑️ Remove", key=f"{key_prefix}-rm-{layer_id}"):
                layers.pop(idx)
                _request_rerun()
                # Stop rendering this list for the rest of this pass rather than continuing
                # enumerate() over a list we just mutated mid-iteration (deferring the rerun --
                # see _request_rerun() -- means this loop no longer aborts here the way an
                # immediate st.rerun() used to). The deferred rerun re-renders the remaining
                # layers correctly on the very next pass -- true now that each layer's widget
                # keys are id-based rather than index-based (see the setdefault() above): a
                # later layer shifting into a removed layer's old index no longer inherits that
                # index's cached session_state.
                break
        base_mask = _build_mask_image(source, layer["kind"], layer["params"])
        if layer["soften"]:
            base_mask = soften_mask(base_mask, layer["soften"])
        mask_stack.add(MaskLayer(layer["name"], base_mask, layer["enabled"], layer["invert"], layer["opacity"]))

    return mask_stack


def _before_after_html(before: Image.Image, after: Image.Image) -> tuple[str, int]:
    """A draggable before/after comparison: a single range input drives a CSS clip-path on the
    "before" image layered over the "after" image. Pure inline HTML/CSS/JS via
    st.components.v1.html -- no component library dependency, consistent with keeping this app's
    UI free of extra dependencies after removing streamlit-drawable-canvas.

    `before` and `after` must already be the same pixel size (the caller resizes if a crop made
    them differ) -- that's what makes the overlay line up.
    """
    width, height = after.size
    display_height = max(220, min(560, int(700 * height / width))) if width else 320

    def _b64_png(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    before_b64 = _b64_png(before)
    after_b64 = _b64_png(after)
    html = f"""
    <div style="position:relative;width:100%;background:#090d16;border-radius:8px;overflow:hidden;line-height:0;">
      <img src="data:image/png;base64,{after_b64}" style="display:block;width:100%;height:auto;">
      <img id="ba-before" src="data:image/png;base64,{before_b64}"
           style="position:absolute;inset:0;width:100%;height:100%;object-fit:fill;clip-path:inset(0 50% 0 0);">
      <div id="ba-handle" style="position:absolute;top:0;bottom:0;left:50%;width:2px;margin-left:-1px;
           background:#fff;box-shadow:0 0 6px rgba(0,0,0,.7);pointer-events:none;"></div>
      <div style="position:absolute;top:8px;left:12px;color:#fff;font:600 11px system-ui;text-shadow:0 1px 3px rgba(0,0,0,.8);letter-spacing:.05em;">BEFORE</div>
      <div style="position:absolute;top:8px;right:12px;color:#fff;font:600 11px system-ui;text-shadow:0 1px 3px rgba(0,0,0,.8);letter-spacing:.05em;">AFTER</div>
      <input id="ba-range" type="range" min="0" max="100" value="50" aria-label="Before/after comparison position"
             style="position:absolute;left:8px;right:8px;bottom:10px;width:calc(100% - 16px);">
    </div>
    <script>
      const range = document.getElementById('ba-range');
      const before = document.getElementById('ba-before');
      const handle = document.getElementById('ba-handle');
      range.addEventListener('input', () => {{
        const v = range.value;
        before.style.clipPath = `inset(0 ${{100 - v}}% 0 0)`;
        handle.style.left = v + '%';
      }});
    </script>
    """
    return html, display_height


def _curve_editor_html(points: list[tuple[int, int]], bounds: list[tuple[int, int]]) -> str:
    """A draggable RGB tone curve: an inline SVG with five points, each vertically draggable
    within its own [min, max] (matching the five sliders below it -- Blacks/Shadows/Midtones/
    Highlights/Whites). Dragging only updates the curve's own SVG locally; only on release does it
    commit.

    Earlier version of this function wrote the dragged value directly into the real
    `<input type="range">` via a native-value-setter trick, then dispatched `input`/`change`.
    Confirmed live (by comparing the actual rendered image before/after, not just the visible
    slider label) that this does NOT reliably reach Streamlit's backend for range sliders in this
    Streamlit version -- the slider's own displayed number updates as a client-side echo, but the
    value the script actually sees on its next run can still be the old one, so a dragged curve
    could silently fail to affect the photo. Real OS-level key/mouse input (a genuine, trusted
    event) does commit reliably; a script-dispatched one does not, for this specific widget type
    (button clicks and a text_area's Ctrl+Enter were confirmed to work synthetically, just not a
    slider's input/change).

    So instead of touching the sliders at all, commit writes the whole 5-point curve as JSON into
    a hidden `st.text_area` (`st.container(key="curve-hidden")`, same Ctrl+Enter trick used by
    `_paint_editor_html`, confirmed reliable) using the RECIPE_WIDGET_KEYS names
    (curve_black/curve_shadows/curve_mid/curve_highlights/curve_white/curve_enabled). Python reads
    it and stores it in `st.session_state["_pending_restore"]` -- the exact same "write before any
    widget instantiates, then rerun" mechanism Undo/Redo/Reset already use to update a slider's
    value from outside the widget itself, which sidesteps the whole reliability question since it
    never depends on a script-dispatched DOM event reaching a slider at all.
    """
    margin_left, margin_top, plot_size = 34, 12, 210
    svg_w, svg_h = margin_left + plot_size + 12, margin_top + plot_size + 12

    def _to_px(x: float, y: float) -> tuple[float, float]:
        return (margin_left + (x / 255) * plot_size, margin_top + plot_size - (y / 255) * plot_size)

    poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in (_to_px(x, y) for x, y in points))
    circles = "\n".join(
        f'<circle class="pt" data-idx="{i}" cx="{px:.1f}" cy="{py:.1f}" r="7"></circle>'
        for i, (px, py) in enumerate(_to_px(x, y) for x, y in points)
    )
    diag_x0, diag_y0 = _to_px(0, 0)
    diag_x1, diag_y1 = _to_px(255, 255)
    points_json = json.dumps(points)
    bounds_json = json.dumps(bounds)
    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:6px 8px;">
    <svg id="curve-svg" width="100%" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}"
         preserveAspectRatio="xMinYMin meet" style="touch-action:none;display:block;">
      <rect x="{margin_left}" y="{margin_top}" width="{plot_size}" height="{plot_size}"
            fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.15)"/>
      <line x1="{diag_x0:.1f}" y1="{diag_y0:.1f}" x2="{diag_x1:.1f}" y2="{diag_y1:.1f}"
            stroke="rgba(255,255,255,0.2)" stroke-dasharray="3,3"/>
      <polyline id="curve-line" points="{poly}" fill="none" stroke="#fbbf24" stroke-width="2"/>
      {circles}
    </svg>
    </div>
    <style>
      #curve-svg .pt {{ fill:#fbbf24; stroke:#1e293b; stroke-width:2; cursor:ns-resize; }}
      #curve-svg .pt:hover {{ fill:#fde68a; }}
    </style>
    <script>
      (function() {{
        const marginLeft = {margin_left}, marginTop = {margin_top}, plotSize = {plot_size}, svgW = {svg_w};
        const bounds = {bounds_json};
        const points = {points_json};
        const svg = document.getElementById('curve-svg');
        const line = document.getElementById('curve-line');
        function toPx(x, y) {{ return [marginLeft + (x / 255) * plotSize, marginTop + plotSize - (y / 255) * plotSize]; }}
        function fromPy(py) {{ return Math.max(0, Math.min(255, ((marginTop + plotSize - py) / plotSize) * 255)); }}
        function redraw() {{
          line.setAttribute('points', points.map(p => toPx(p[0], p[1]).join(',')).join(' '));
          svg.querySelectorAll('.pt').forEach((c, i) => {{
            const [px, py] = toPx(points[i][0], points[i][1]);
            c.setAttribute('cx', px); c.setAttribute('cy', py);
          }});
        }}
        function commit() {{
          try {{
            const payload = JSON.stringify({{
              curve_black: points[0][1], curve_shadows: points[1][1], curve_mid: points[2][1],
              curve_highlights: points[3][1], curve_white: points[4][1], curve_enabled: true,
            }});
            const container = window.parent.document.querySelector('.st-key-curve-hidden');
            if (!container) return;
            const textarea = container.querySelector('textarea');
            if (!textarea) return;
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            textarea.focus();
            setter.call(textarea, payload);
            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keyup', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.blur();
          }} catch (e) {{}}
        }}
        let dragIdx = null;
        svg.querySelectorAll('.pt').forEach((circle, i) => {{
          circle.addEventListener('pointerdown', (e) => {{
            dragIdx = i;
            circle.setPointerCapture(e.pointerId);
            e.preventDefault();
          }});
        }});
        svg.addEventListener('pointermove', (e) => {{
          if (dragIdx === null) return;
          const rect = svg.getBoundingClientRect();
          const scale = svgW / rect.width;
          const py = (e.clientY - rect.top) * scale;
          let value = Math.round(fromPy(py));
          const [lo, hi] = bounds[dragIdx];
          value = Math.max(lo, Math.min(hi, value));
          points[dragIdx][1] = value;
          redraw();
        }});
        function endDrag() {{
          if (dragIdx === null) return;
          dragIdx = null;
          commit();
        }}
        svg.addEventListener('pointerup', endDrag);
        svg.addEventListener('pointerleave', endDrag);
      }})();
    </script>
    """


def _paint_editor_html(background: Image.Image, key_prefix: str) -> tuple[str, int]:
    """An on-canvas selection editor: brush (paint/erase freehand), radial (drag out a circle),
    and gradient (drag out a direction), all painted onto one HTML canvas as solid white at
    varying alpha -- alpha *is* the mask value, so "Add to selections" just base64-encodes the
    canvas as a PNG and that becomes a new MaskLayer's `params["mask_b64"]` (decoded by
    `_decode_painted_mask`). Committing writes into the hidden `st.text_area` inside
    `st.container(key=f"{key_prefix}-paint-hidden")` (CSS-hidden -- see the app's <style> block)
    via the same native-value-setter trick as `_curve_editor_html`, so Python notices it on the
    next rerun and appends the layer itself (see the "Painted" branch in _render_selection_stack)
    -- there's no separate native Streamlit "Add" button to click after drawing, the in-canvas
    button *is* the commit. No component library dependency, same approach as the other editors
    in this file; avoids a repeat of the streamlit-drawable-canvas dead-weight episode.
    """
    width, height = background.size
    display_w = min(480, width)
    display_h = int(display_w * height / width) if width else 320

    buffer = io.BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    bg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    hidden_selector = f".st-key-{key_prefix}-paint-hidden"

    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:8px;">
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px;">
        <button type="button" class="tool-btn active" data-tool="brush">🖌 Brush</button>
        <button type="button" class="tool-btn" data-tool="radial">⭕ Radial</button>
        <button type="button" class="tool-btn" data-tool="gradient">📐 Gradient</button>
        <button type="button" id="erase-btn" class="tool-btn">🧹 Erase</button>
        <button type="button" id="clear-btn" class="tool-btn">✖ Clear</button>
        <label style="display:flex;align-items:center;gap:4px;color:#cbd5e1;font:12px system-ui;margin-left:4px;">
          Size <input id="brush-size" type="range" min="4" max="80" value="18" style="width:80px;">
        </label>
      </div>
      <div style="position:relative;width:100%;max-width:{display_w}px;line-height:0;touch-action:none;">
        <img src="data:image/png;base64,{bg_b64}" style="display:block;width:100%;height:auto;border-radius:4px;">
        <canvas id="mask-canvas" width="{display_w}" height="{display_h}"
                style="position:absolute;inset:0;width:100%;height:100%;opacity:0.6;cursor:crosshair;"></canvas>
        <canvas id="preview-canvas" width="{display_w}" height="{display_h}"
                style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;"></canvas>
      </div>
      <button type="button" id="commit-btn" style="margin-top:8px;width:100%;padding:8px;border-radius:6px;
              border:none;background:#fbbf24;color:#1e293b;font:600 13px system-ui;cursor:pointer;">
        ➕ Add to selections
      </button>
    </div>
    <style>
      .tool-btn {{ background:rgba(255,255,255,0.08); color:#cbd5e1; border:1px solid rgba(255,255,255,0.15);
                   border-radius:5px; padding:5px 9px; font:12px system-ui; cursor:pointer; }}
      .tool-btn.active {{ background:#fbbf24; color:#1e293b; border-color:#fbbf24; }}
      #erase-btn.active {{ background:#f87171; color:#1e293b; border-color:#f87171; }}
    </style>
    <script>
      (function() {{
        const maskCanvas = document.getElementById('mask-canvas');
        const previewCanvas = document.getElementById('preview-canvas');
        const ctx = maskCanvas.getContext('2d');
        const pctx = previewCanvas.getContext('2d');
        const W = maskCanvas.width, H = maskCanvas.height;
        let tool = 'brush', erasing = false, brushSize = 18;
        let drawing = false, lastX = 0, lastY = 0, startX = 0, startY = 0;

        document.querySelectorAll('.tool-btn[data-tool]').forEach((btn) => {{
          btn.addEventListener('click', () => {{
            tool = btn.dataset.tool;
            document.querySelectorAll('.tool-btn[data-tool]').forEach((b) => b.classList.toggle('active', b === btn));
          }});
        }});
        document.getElementById('erase-btn').addEventListener('click', (e) => {{
          erasing = !erasing;
          e.currentTarget.classList.toggle('active', erasing);
        }});
        document.getElementById('clear-btn').addEventListener('click', () => ctx.clearRect(0, 0, W, H));
        document.getElementById('brush-size').addEventListener('input', (e) => {{ brushSize = Number(e.target.value); }});

        function toCanvasXY(e) {{
          const rect = maskCanvas.getBoundingClientRect();
          return [(e.clientX - rect.left) * (W / rect.width), (e.clientY - rect.top) * (H / rect.height)];
        }}

        function paintDab(x, y) {{
          ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
          ctx.strokeStyle = 'white'; ctx.fillStyle = 'white';
          ctx.lineWidth = brushSize * 2; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
          ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(x, y); ctx.stroke();
          ctx.beginPath(); ctx.arc(x, y, brushSize, 0, Math.PI * 2); ctx.fill();
        }}

        function paintRadial(cx, cy, radius) {{
          ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
          const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(1, radius));
          grad.addColorStop(0, 'rgba(255,255,255,1)');
          grad.addColorStop(0.7, 'rgba(255,255,255,1)');
          grad.addColorStop(1, 'rgba(255,255,255,0)');
          ctx.fillStyle = grad;
          ctx.beginPath(); ctx.arc(cx, cy, Math.max(1, radius), 0, Math.PI * 2); ctx.fill();
        }}

        function paintGradient(x0, y0, x1, y1) {{
          ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
          const grad = ctx.createLinearGradient(x0, y0, x1, y1);
          grad.addColorStop(0, 'rgba(255,255,255,1)');
          grad.addColorStop(1, 'rgba(255,255,255,0)');
          ctx.fillStyle = grad;
          ctx.fillRect(0, 0, W, H);
        }}

        maskCanvas.addEventListener('pointerdown', (e) => {{
          drawing = true;
          maskCanvas.setPointerCapture(e.pointerId);
          [startX, startY] = toCanvasXY(e);
          lastX = startX; lastY = startY;
          if (tool === 'brush') paintDab(startX, startY);
          e.preventDefault();
        }});
        maskCanvas.addEventListener('pointermove', (e) => {{
          if (!drawing) return;
          const [x, y] = toCanvasXY(e);
          if (tool === 'brush') {{
            paintDab(x, y);
            lastX = x; lastY = y;
          }} else {{
            pctx.clearRect(0, 0, W, H);
            pctx.strokeStyle = 'rgba(255,255,255,0.9)'; pctx.lineWidth = 1.5; pctx.setLineDash([4, 3]);
            if (tool === 'radial') {{
              const radius = Math.hypot(x - startX, y - startY);
              pctx.beginPath(); pctx.arc(startX, startY, radius, 0, Math.PI * 2); pctx.stroke();
            }} else {{
              pctx.beginPath(); pctx.moveTo(startX, startY); pctx.lineTo(x, y); pctx.stroke();
            }}
          }}
        }});
        function endStroke(e) {{
          if (!drawing) return;
          drawing = false;
          pctx.clearRect(0, 0, W, H);
          if (tool === 'radial') {{
            const [x, y] = toCanvasXY(e);
            paintRadial(startX, startY, Math.hypot(x - startX, y - startY));
          }} else if (tool === 'gradient') {{
            const [x, y] = toCanvasXY(e);
            paintGradient(startX, startY, x, y);
          }}
        }}
        maskCanvas.addEventListener('pointerup', endStroke);
        maskCanvas.addEventListener('pointerleave', endStroke);

        document.getElementById('commit-btn').addEventListener('click', () => {{
          const base64 = maskCanvas.toDataURL('image/png').split(',')[1];
          try {{
            const container = window.parent.document.querySelector('{hidden_selector}');
            if (!container) return;
            const textarea = container.querySelector('textarea');
            if (!textarea) return;
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            textarea.focus();
            setter.call(textarea, base64);
            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            // Confirmed live: unlike a slider/checkbox (which commit on a plain 'change'),
            // st.text_area only commits -- and reruns the script -- on Ctrl+Enter or on losing
            // focus, and a synthetic .blur() alone wasn't reliably enough to trigger that in
            // this Streamlit version. Ctrl+Enter is Streamlit's own documented text_area
            // shortcut, so dispatch that directly rather than depend on blur semantics.
            textarea.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.dispatchEvent(new KeyboardEvent('keyup', {{ key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true, cancelable: true }}));
            textarea.blur();
          }} catch (e) {{}}
        }});
      }})();
    </script>
    """, display_h + 130


@st.cache_data(show_spinner=False)
def _preset_thumbnails(image_bytes: bytes, filename: str) -> dict[str, bytes]:
    """Small local preview renders for the Preset Look dropdown, so a user can see what each
    preset looks like before picking one instead of guessing from its name. Decorative only --
    generated in-process from a small thumbnail, separate from the real edit pipeline, which
    always goes through app/backend.py regardless of what's previewed here."""
    base = PhotoProcessor.from_bytes(image_bytes, filename).image
    base.thumbnail((96, 96), Image.Resampling.LANCZOS)
    thumbnails = {"None": base}
    for name in PRESETS:
        preview = PhotoProcessor.from_image(base)
        apply_preset(preview, name)
        thumbnails[name] = preview.image
    encoded = {}
    for name, thumb in thumbnails.items():
        buffer = io.BytesIO()
        thumb.save(buffer, format="PNG")
        encoded[name] = buffer.getvalue()
    return encoded


# Keyboard shortcuts for the Library Catalog's "Rate & Flag" panel: 1-5 rate, 0 clears, X toggles
# reject, arrows move Prev/Next. Reaches into window.parent.document (the real Streamlit page --
# this component's iframe is same-origin via `srcdoc`, confirmed live) to `.click()` the real
# buttons inside `st.container(key="lib-rate-controls")`, by fixed position (see the Python code
# that builds that container -- the two must stay in the same order). Skips the parent's normal
# typing targets (inputs/textareas) so it doesn't hijack the Search or Tags fields, and only acts
# when the panel is actually visible (`offsetParent !== null`) so it doesn't fire from a hidden
# tab -- every tab's Python code runs on every rerun in this app, visible or not. A static string,
# not a per-render function, since it has no values to interpolate.
_LIBRARY_KEYBOARD_SHORTCUTS_HTML = """
<script>
  (function() {
    const KEY_TO_BUTTON_INDEX = {
      'ArrowLeft': 0, 'ArrowRight': 1,
      '1': 2, '2': 3, '3': 4, '4': 5, '5': 6,
      '0': 7, 'x': 8, 'X': 8,
    };
    function handler(e) {
      const active = window.parent.document.activeElement;
      const tag = active && active.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      const idx = KEY_TO_BUTTON_INDEX[e.key];
      if (idx === undefined) return;
      const container = window.parent.document.querySelector('.st-key-lib-rate-controls');
      if (!container || container.offsetParent === null) return;
      const buttons = container.querySelectorAll('button');
      const target = buttons[idx];
      if (!target) return;
      target.click();
      e.preventDefault();
    }
    if (!window.parent._photoStudioLibraryShortcutsBound) {
      window.parent.document.addEventListener('keydown', handler);
      window.parent._photoStudioLibraryShortcutsBound = true;
    }
  })();
</script>
"""


@st.cache_data(show_spinner="🎨 Applying edits…")
def _cached_process_via_backend(image_bytes: bytes, filename: str, options: dict, local_mask_bytes: bytes | None, layer_bytes: bytes | None, removal_mask_bytes: bytes | None = None):
    """Cached so that reruns triggered by controls that don't affect the pipeline (zoom, Show
    Original, etc.) don't repeat a full HTTP round trip with the whole image. The spinner message
    is the only feedback the user gets that an edit is being sent over the network to
    app/backend.py -- without it, a slider drag on a large image can look like the app hung."""
    return process_via_backend(image_bytes, filename, options, local_mask_bytes, layer_bytes, removal_mask_bytes)


@st.cache_data(show_spinner=False)
def _lens_corrected_source(image_bytes: bytes, filename: str, distortion: float, vignette_amount: float, vignette_midpoint: float, chromatic_aberration: float) -> Image.Image:
    """The base image the Crop preview and the Masks/Repair on-canvas selection tools paint
    against. app/backend.py::process_image() applies Lens corrections (a real cv2.undistort/remap
    geometric warp for distortion/chromatic aberration) before Crop or any mask, so a crop box or
    mask painted against the raw, uncorrected source would land on the wrong pixels once the
    backend actually warps the image -- worst near edges/corners, where distortion moves content
    the most. Cached (and a no-op fast path when every lens value is still at its default) since
    this recomputes on every Streamlit rerun otherwise, not just when a Lens slider changes."""
    image = PhotoProcessor.from_bytes(image_bytes, filename).image
    if not (distortion or vignette_amount or chromatic_aberration):
        return image
    return apply_lens_corrections(image, distortion, vignette_amount, vignette_midpoint, chromatic_aberration)


@st.cache_data(show_spinner=False)
def _cached_gamut_warning_overlay(image_bytes: bytes, filename: str, icc_bytes: bytes):
    """gamut_warning_overlay() runs two full-image littlecms conversions -- cached so it only
    recomputes when the photo or its embedded profile actually changes, not on every Streamlit
    rerun triggered by an unrelated control while the "Show out-of-gamut pixels" checkbox stays
    on."""
    image = PhotoProcessor.from_bytes(image_bytes, filename).image
    return gamut_warning_overlay(image, icc_bytes)


def _request_rerun() -> None:
    """Use instead of calling st.rerun() directly from ANY widget callback below this point in the
    script -- a tab body (Repair/Layers/Masks/Portrait/Presets/Batch & Library), or canvas_col
    (Undo/Redo/Reset, zoom, split view, pan, export). Streamlit only keeps a keyed widget's
    session_state value if that widget was actually instantiated during the run immediately before
    it; the whole script (every tab, then canvas_col) runs top-to-bottom in one pass regardless of
    which tab is visible, so an immediate st.rerun() from anywhere before the last widget in that
    sequence aborts the script right there -- none of the not-yet-reached widgets get instantiated
    this pass, so Streamlit garbage-collects their session_state and silently resets them to their
    `value=` defaults on the very next run. (Same underlying mechanism as the HSL Color Mixer's
    dropdown-hidden-band bug documented in docs/PROJECT_STATUS.md, just triggered by an early
    rerun instead of conditional rendering -- a real instance of exactly this was found in review:
    the consume-check below used to sit before canvas_col rendered, so any tab's _request_rerun()
    call was already resetting canvas_col's zoom/pan/split-view/export settings, and Library
    Catalog's and canvas_col's own Undo/Redo/Reset buttons called st.rerun() directly and had the
    same effect.) Setting this flag and rerunning only after the very last widget in the script has
    rendered (see the check at the bottom of this file) means nothing gets GC'd."""
    st.session_state["_rerun_pending"] = True


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
    border-bottom: 2px solid rgba(255, 255, 255, 0.12);
    padding: 0.2rem 0 0.5rem 0;
    margin-bottom: 0.6rem;
}
.brand-title {
    font: 800 1.5rem/1.1 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #f1f5f9;
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

/* Workspace tab styling -- wrap onto a second row instead of hiding tabs behind a scroll arrow;
   with 9 tabs in a narrow column, scrolling silently hid Crop/Presets/More from first-time users.
   Streamlit 1.6x renders tabs via React Aria (role="tablist"/"tab"), not the older BaseWeb
   data-baseweb attributes -- these selectors target the current markup (data-testid="stTab(s)"). */
[data-testid="stTab"] {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    padding: 0.35rem 0.6rem !important;
}
[data-testid="stTabs"] [role="tablist"] {
    flex-wrap: wrap !important;
    overflow-x: visible !important;
    row-gap: 0.2rem;
}

/* On a narrow browser window the two-column workspace stacks vertically; without this, the
   whole (long) tool panel renders above the canvas, pushing the actual photo out of view until
   the user scrolls past every slider. Flip the visual order so the canvas -- the thing being
   edited -- appears first, while leaving the underlying script order (and therefore which
   column's widgets execute first) untouched. Scoped to the "workspace-split" keyed container so
   this doesn't affect any other st.columns() row in the app. */
@media (max-width: 900px) {
    .st-key-workspace-split [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    .st-key-workspace-split [data-testid="stColumn"]:nth-of-type(1) { order: 2; }
    .st-key-workspace-split [data-testid="stColumn"]:nth-of-type(2) { order: 1; }
}

/* Holds the real <textarea> the on-canvas paint editor's "Add to selections" button writes a
   base64 mask PNG into (see _paint_editor_html / _render_selection_stack's "Painted" branch), and
   the one the tone curve editor's drag-commit writes curve-point JSON into (see
   _curve_editor_html / the RGB Tone Curve expander) -- plumbing details, not something the user
   should ever see or type into directly. */
.st-key-m-paint-hidden, .st-key-rm-paint-hidden, .st-key-curve-hidden {
    display: none !important;
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
        # A hand-edited or corrupted .photoproject (bad ZIP, malformed JSON manifest, a manifest
        # pointing at a file that isn't actually in the archive, unreadable embedded PNGs, ...)
        # must not crash the whole app with a raw traceback -- every other user-input path in
        # this file (RAW load, backend calls, curve JSON) already degrades to a friendly error.
        try:
            loaded_project = ProjectBundle.load(project_upload.getvalue())
            pending_recipe = dict(loaded_project.recipe)
            workspace = loaded_project.layer_workspace_object()
            overlay_image_b64 = None
            if workspace and len(workspace.layers) > 1:
                overlay_layer = workspace.layers[1]
                overlay_buffer = io.BytesIO()
                overlay_layer.image.save(overlay_buffer, format="PNG")
                overlay_image_b64 = base64.b64encode(overlay_buffer.getvalue()).decode("ascii")
        except (ValueError, KeyError, OSError) as error:
            st.error(f"⚠️ Could not open this project file: {error}")
        else:
            st.session_state["restored_source_bytes"] = loaded_project.source_bytes
            st.session_state["restored_source_name"] = loaded_project.source_name
            # ProjectBundle stores the overlay layer in its own structured `layer_workspace` field
            # (a serialized LayerWorkspace), not base64-embedded in the recipe like the old
            # PhotoProject format -- translate it into the same recipe["layer"] shape
            # _restore_recipe_to_state() already knows how to apply, so that function needs no changes.
            if overlay_image_b64 is not None:
                pending_recipe["layer"] = {
                    "image_b64": overlay_image_b64,
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
except OSError:
    # PIL raises OSError (UnidentifiedImageError, "image file is truncated", ...) for a corrupted,
    # incomplete, or unrecognizable non-RAW file -- this used to propagate as an unhandled
    # exception, showing the end user a raw Python traceback (including local file paths) instead
    # of a plain explanation. The technical message isn't worth surfacing (PIL's own text for an
    # unidentifiable file includes a raw memory address), so this doesn't interpolate `error`.
    st.error(f"⚠️ Couldn't open '{upload_name}' — the file may be corrupted, incomplete, or not a supported image format.")
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
with st.container(key="workspace-split"):
    tools_col, canvas_col = st.columns([1.1, 1.4], gap="large")

    with tools_col:
        st.markdown('<div class="panel-header-title">🛠️ Tool Workspace</div>', unsafe_allow_html=True)
        
        # Crop sits right after Tone in this visual order -- a workflow convenience (see most photo
        # editors: framing first, before fine-tuning tone/color) that's independent of the actual
        # processing order. app/backend.py::process_image() always applies the crop last, after
        # Masks/Portrait, regardless of where this tab sits in the UI -- every tab's code already
        # runs on every rerun no matter which one is visually selected (see the note below the
        # tabs), so reordering this tuple/list is purely cosmetic and doesn't change what gets sent
        # to the backend or when.
        # Repair sits right after Detail, matching where app/backend.py::process_image() applies
        # removal_mask in the actual pipeline (right after Detail, before Layers) -- unlike Crop,
        # there's no workflow reason to decouple this one's visual position from its pipeline
        # position, so they're kept in sync.
        # Lens sits right after Tone, before Crop -- fix distortion/vignette/chromatic aberration
        # before framing the crop against the now-corrected geometry. app/backend.py::process_image()
        # applies lens corrections first in the actual pipeline (before Tone even), same reasoning
        # as Crop above: its visual tab position is a workflow convenience, decoupled from where it
        # actually runs.
        tab_tone, tab_lens, tab_crop, tab_color, tab_detail, tab_repair, tab_layers, tab_masks, tab_portrait, tab_presets, tab_more = st.tabs([
            "☼ Tone", "◎ Lens", "⛶ Crop", "🎨 Color", "🔬 Detail", "🩹 Repair", "◈ Layers", "◉ Masks", "◌ Portrait", "✦ Presets", "☰ Batch & Library"
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
            # Mirrors PhotoProcessor.auto_enhance() (app/image_processor.py) exactly -- that method
            # is otherwise only reachable from the CLI's --auto-enhance flag. Writing straight into
            # _pending_restore reuses the same "one rerun to apply" mechanism as Undo/Redo/Reset,
            # and only touches these five recipe fields, leaving everything else (crop, masks,
            # curve, temperature, ...) exactly as the user left it.
            if st.button("✨ Auto Enhance", key="auto-enhance-btn", help="Apply a quick, balanced exposure/contrast/saturation/denoise/sharpen pass", use_container_width=True):
                st.session_state["_pending_restore"] = {
                    "exposure": 1.12, "contrast": 1.15, "saturation": 1.08, "denoise": 3, "sharpen": True,
                }
                _request_rerun()
            preset_names = ["None", *sorted(PRESETS)]
            preset_thumbs = _preset_thumbnails(upload_bytes, upload_name)
            thumb_cols = st.columns(len(preset_names))
            for thumb_col, name in zip(thumb_cols, preset_names):
                with thumb_col:
                    st.image(preset_thumbs[name], use_container_width=True)
                    st.caption(name)
            preset_choice = st.selectbox("Preset Look", preset_names, key="tone-preset")
            exp_val = st.slider("Exposure", 0.5, 1.5, 1.0, 0.01, key="t-exp")
            cont_val = st.slider("Contrast", 0.5, 1.5, 1.0, 0.01, key="t-cont")
            high_val = st.slider("Highlights", -100, 100, 0, key="t-high")
            shad_val = st.slider("Shadows", -100, 100, 0, key="t-shad")

        # TAB: LENS CORRECTIONS
        with tab_lens:
            st.markdown("**Lens Corrections**")
            st.caption(
                "Not calibrated to a specific camera/lens model -- there's no profile database here, "
                "just typical starting points by lens category. Pick the closest one, then fine-tune "
                "with the sliders below."
            )
            lens_profile_choice = st.selectbox("Lens Profile", list(LENS_PROFILES), key="lens-profile")
            # No help= here: st.button(..., help=...) renders a second, separately-focusable
            # tooltip-trigger <button> in this Streamlit version (see the Library rate/flag panel's
            # star-button fix in docs/PROJECT_STATUS.md) -- harmless for a plain click target like
            # this one, but a caption is simpler and avoids the duplicate DOM node regardless.
            st.caption("Loads this profile's starting values -- you can still adjust the sliders afterward.")
            if st.button("Apply Profile Defaults", key="lens-apply-profile-btn"):
                profile_values = LENS_PROFILES[lens_profile_choice]
                st.session_state["_pending_restore"] = {
                    "lens_distortion": profile_values["distortion"],
                    "lens_vignette_amount": profile_values["vignette_amount"],
                    "lens_vignette_midpoint": profile_values["vignette_midpoint"],
                    "lens_chromatic_aberration": profile_values["chromatic_aberration"],
                }
                _request_rerun()
            lens_distortion = st.slider(
                "Distortion", -100, 100, 0, key="lens-distortion",
                help="Negative corrects barrel distortion (straight lines bowing outward); positive corrects pincushion (bowing inward).",
            )
            lens_vignette_amount = st.slider(
                "Vignette Amount", -100, 100, 0, key="lens-vignette-amt",
                help="Negative darkens the corners (adds a vignette); positive brightens them (removes one).",
            )
            lens_vignette_midpoint = st.slider(
                "Vignette Midpoint", 0, 100, 50, key="lens-vignette-mid",
                help="How far from center the vignette effect starts -- higher concentrates it closer to the corners.",
            )
            lens_chromatic_aberration = st.slider(
                "Chromatic Aberration", -100, 100, 0, key="lens-ca",
                help="Corrects red/cyan and blue/yellow fringing near high-contrast edges.",
            )

        # The Crop preview and the Masks/Repair selection tools below all paint against this,
        # not the raw `source` -- see _lens_corrected_source()'s docstring for why: the backend
        # applies Lens corrections before Crop/Masks/Repair, so those tools need to agree with it
        # on which pixel is where.
        mask_source = _lens_corrected_source(
            upload_bytes, upload_name, lens_distortion, lens_vignette_amount, lens_vignette_midpoint, lens_chromatic_aberration
        )

        # TAB: CROP & ASPECT RATIO -- placed right after Tone for workflow convenience (frame the
        # shot before fine-tuning tone/color); still applied last in the actual pipeline, see the
        # note above the tabs() call.
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
                    preview = mask_source.copy()
                    ImageDraw.Draw(preview).rectangle(crop_box, outline=(255, 60, 40), width=max(2, src_w // 200))
                    st.image(preview, caption=f"Crop preview: {crop_box[2] - crop_box[0]}×{crop_box[3] - crop_box[1]}px", use_container_width=True)
            elif crop_choice != "Original":
                crop_w_ratio, crop_h_ratio = ASPECT_RATIOS[crop_choice]
                st.caption(f"Center-cropping to {crop_choice} ({crop_w_ratio}:{crop_h_ratio}).")
                aspect_code = f"{crop_w_ratio}:{crop_h_ratio}"
            else:
                st.caption("No crop applied — showing the full frame.")

        # TAB 2: COLOR & CURVES
        with tab_color:
            temp_val = st.slider("Temperature (Cool/Warm)", -100, 100, 0, key="c-temp")
            tint_val = st.slider("Tint (Green/Magenta)", -100, 100, 0, key="c-tint")
            sat_val = st.slider("Saturation", 0.0, 2.0, 1.0, 0.01, key="c-sat")

            st.markdown("**Color Mixer (HSL)**")
            st.caption("Adjust Hue, Saturation, and Luminance independently for each color band, like a professional editor's Color Mixer.")
            # Every band's sliders render unconditionally every rerun (not just a "currently
            # selected" one behind a picker) -- an earlier version of this used a selectbox to
            # show only one band's 3 sliders at a time, but Streamlit garbage-collects a widget's
            # session_state value when its key isn't instantiated in a run, so switching away from
            # a band silently reset it back to 0. Always rendering every widget sidesteps that
            # entirely, and matches the same "list of expandable items" pattern already used for
            # Mask Layers below -- collapsed by default, so this isn't much bulkier in practice.
            # (Nested inside an outer expander it would be: st.expander cannot nest inside
            # another st.expander, which is also why this whole section isn't wrapped in one.)
            hsl_bands = {}
            for band in HSL_BAND_ORDER:
                h_key, s_key, l_key = _hsl_widget_key(band, "h"), _hsl_widget_key(band, "s"), _hsl_widget_key(band, "l")
                is_active = any(st.session_state.get(k, 0) for k in (h_key, s_key, l_key))
                with st.expander(f"{'🟢' if is_active else '⚪'} {band}", expanded=False):
                    hue = st.slider("Hue", -100, 100, 0, key=h_key)
                    sat = st.slider("Saturation", -100, 100, 0, key=s_key)
                    lum = st.slider("Luminance", -100, 100, 0, key=l_key)
                hsl_bands[band] = {"hue": hue, "sat": sat, "lum": lum}

            with st.expander("RGB Tone Curve", expanded=False):
                curve_bounds = [(0, 64), (0, 128), (64, 192), (128, 255), (192, 255)]
                preview_points = [
                    (0, st.session_state.get("c-p0", 0)),
                    (64, st.session_state.get("c-p1", 64)),
                    (128, st.session_state.get("c-p2", 128)),
                    (192, st.session_state.get("c-p3", 192)),
                    (255, st.session_state.get("c-p4", 255)),
                ]
                components.html(_curve_editor_html(preview_points, curve_bounds), height=246)
                with st.container(key="curve-hidden"):
                    curve_commit = st.text_area("Curve commit data", key="curve-commit-data", label_visibility="collapsed")
                # See _curve_editor_html's docstring: a script-dispatched event on a range slider
                # doesn't reliably commit in this Streamlit version, so the drag handler writes
                # here instead and this reuses the same "set _pending_restore, then rerun" path
                # Undo/Redo/Reset already use to update a slider from outside the widget itself.
                if curve_commit and curve_commit != st.session_state.get("curve-committed"):
                    try:
                        parsed = json.loads(curve_commit)
                    except (ValueError, TypeError):
                        parsed = None
                    if isinstance(parsed, dict):
                        st.session_state["curve-committed"] = curve_commit
                        st.session_state["_pending_restore"] = parsed
                        _request_rerun()
                with st.container(key="curve-points"):
                    use_curve = st.checkbox("Enable RGB Curve", value=False, key="c-curve-en")
                    c_black = st.slider("Blacks", 0, 64, 0, key="c-p0")
                    c_low = st.slider("Shadows", 0, 128, 64, key="c-p1")
                    c_mid = st.slider("Midtones", 64, 192, 128, key="c-p2")
                    c_high = st.slider("Highlights", 128, 255, 192, key="c-p3")
                    c_white = st.slider("Whites", 192, 255, 255, key="c-p4")
                st.caption("Drag a point on the curve, or use the sliders below for exact values.")

            with st.expander("🎯 Soft Proof: Gamut Warning", expanded=False):
                source_icc_bytes = source.info.get("icc_profile")
                if not source_icc_bytes:
                    st.caption("This photo has no embedded color profile to check against, so there's nothing to proof.")
                else:
                    st.caption(
                        "Highlights, in magenta, pixels that get clipped when this app's own sRGB pipeline converts "
                        "the photo's embedded color profile. Not a simulation of a specific printer/press profile -- "
                        "there's no profile database bundled -- just where this app's own color management can't "
                        "fully represent the source."
                    )
                    if st.checkbox("Show out-of-gamut pixels", key="gamut-warning-toggle"):
                        try:
                            st.image(
                                _cached_gamut_warning_overlay(upload_bytes, upload_name, source_icc_bytes),
                                caption="Magenta = clipped when converted to this app's sRGB working space",
                                use_container_width=True,
                            )
                        except ICC_TRANSFORM_ERRORS:
                            st.caption("Could not read this photo's embedded color profile.")

        # TAB 3: DETAIL & SHARPNESS
        with tab_detail:
            text_val = st.slider("Texture", -100, 100, 0, key="d-text")
            clar_val = st.slider("Clarity", -100, 100, 0, key="d-clar")
            denoise_val = st.slider("Denoise", 0, 9, 0, step=2, key="d-den")
            sharp_en = st.checkbox("Sharpen (Unsharp Mask)", value=False, key="d-sharp")

        # TAB: REPAIR (OBJECT REMOVAL)
        with tab_repair:
            st.markdown("**Remove Objects**")
            st.caption("Select the blemish or object to remove with the same selection tools as Masks (below). The selected area is painted over by blending in nearby texture.")

            removal_stack = _render_selection_stack("removal_mask_layers", "rm", mask_source, include_opacity=False)
            removal_combined_mask = removal_stack.combined()
            removal_active = any(layer["enabled"] for layer in st.session_state.removal_mask_layers)

            if not st.session_state.removal_mask_layers:
                st.caption("No selections yet — nothing will be removed.")
            elif not removal_active:
                st.caption("All selections are disabled — nothing will be removed.")

            removal_radius = st.slider(
                "Repair Softness", 1, 15, 3, key="rm-radius", disabled=not removal_active,
                help="How far the repair blends into the surrounding area -- higher softens a wider edge.",
            )

            if st.checkbox("👁️ Preview removal selection", key="rm-preview-overlay") and st.session_state.removal_mask_layers:
                # Yellow (vs. Masks' red) so the two overlays are never confused with each other.
                st.image(apply_mask_overlay(mask_source, removal_combined_mask, color=(255, 200, 0)), caption="Yellow = will be removed", use_container_width=True)

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

            mask_stack = _render_selection_stack("mask_layers", "m", mask_source)
            # Union of every enabled layer (or an all-zero, no-op mask when there are none).
            combined_mask = mask_stack.combined()

            if not st.session_state.mask_layers:
                st.caption("No mask layers yet — the sliders below have no effect until you add one.")
            elif not any(layer["enabled"] for layer in st.session_state.mask_layers):
                st.caption("All mask layers are disabled — the sliders below have no effect.")

            if st.checkbox("👁️ Preview combined mask overlay", key="m-preview-overlay") and st.session_state.mask_layers:
                # Overlaid on mask_source (lens-corrected, but otherwise untouched) -- the
                # backend-processed preview isn't available locally, so this debug view shows the
                # selection rather than the live edited result.
                st.image(apply_mask_overlay(mask_source, combined_mask), caption="Red = selected by the combined mask", use_container_width=True)

            mask_inert = not any(layer["enabled"] for layer in st.session_state.mask_layers)
            m_exp = st.slider("Target Exposure", 0.5, 1.5, 1.0, 0.01, key="m-texp", disabled=mask_inert)
            m_cont = st.slider("Target Contrast", 0.5, 1.5, 1.0, 0.01, key="m-tcont", disabled=mask_inert)
            m_sat = st.slider("Target Saturation", 0.0, 2.0, 1.0, 0.01, key="m-tsat", disabled=mask_inert)

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

        # TAB 7: PRESETS
        with tab_presets:
            st.markdown("**Preset Library**")
            p_store = PresetStore()
            p_names = p_store.names()
            st.caption(f"Saved: {', '.join(p_names) if p_names else 'None'}")

            # Load/apply/delete a saved custom preset -- p_store.get()/.delete() already existed
            # and were tested, but nothing in the UI ever called them: a preset could be saved,
            # exported, and imported, but never actually reapplied to a photo. Routed through the
            # same "_pending_restore" + _request_rerun() mechanism as Undo/Redo/Load Project, since
            # a preset is just a recipe like any other.
            if p_names:
                st.markdown("**Load a Preset**")
                p_load_name = st.selectbox("Preset to load", p_names, key="p-load-sel", label_visibility="collapsed")
                p_load_col, p_delete_col = st.columns(2)
                if p_load_col.button("📂 Apply", key="p-load-btn", use_container_width=True):
                    st.session_state["_pending_restore"] = p_store.get(p_load_name)
                    st.toast(f"Applied preset '{p_load_name}'", icon="📂")
                    _request_rerun()
                if p_delete_col.button("🗑️ Delete", key="p-delete-btn", use_container_width=True):
                    p_store.delete(p_load_name)
                    st.toast(f"Deleted preset '{p_load_name}'", icon="🗑️")
                    _request_rerun()

            p_new_name = st.text_input("New Preset Name", key="p-name-in")
            if st.button("Save Current Look as Preset", key="p-save-btn"):
                if p_new_name.strip():
                    # The full recipe (every Tone/Lens/Color/Detail/Masks/... control, plus mask
                    # layers), not just 5 basic fields -- a preset saved after building a mask or a
                    # lens correction used to silently drop all of that. Deliberately excludes the
                    # layer overlay image, same as Undo/Redo history (see _build_recipe's
                    # docstring): a preset is a reusable style, not tied to one photo's overlay.
                    p_store.save(p_new_name.strip(), _build_recipe())
                    st.toast(f"Preset '{p_new_name.strip()}' saved", icon="💾")
                    _request_rerun()
                else:
                    st.toast("Type a name before saving a preset", icon="⚠️")

            st.markdown("---")
            st.markdown("**Export / Import Presets**")
            if p_store.names():
                st.download_button(
                    "⬇️ Export all presets",
                    export_presets(p_store),
                    file_name="photo-studio-presets.json",
                    mime="application/json",
                    key="preset-export-btn",
                )
            else:
                st.caption("No saved presets to export yet.")
            preset_import_file = st.file_uploader("Import presets (.json)", type=["json"], key="preset-import-upload")
            if preset_import_file and st.button("Import", key="preset-import-btn"):
                try:
                    imported = import_presets(p_store, preset_import_file.getvalue())
                except ValueError as error:
                    st.toast(f"Could not import presets: {error}", icon="⚠️")
                else:
                    if imported:
                        st.toast(f"Imported {len(imported)} preset(s)", icon="📥")
                        _request_rerun()
                    else:
                        # Existing names win (import_presets(overwrite=False)) -- every name in the
                        # file already exists locally, so nothing changed and nothing needs a rerun.
                        st.toast("No new presets to import — names already exist", icon="ℹ️")

            st.markdown("---")
            if "edit_history" not in st.session_state:
                st.session_state.edit_history = EditHistory()
            history = st.session_state.edit_history
            history.record(_build_recipe())
            st.caption("↶ Undo / ↷ Redo are always available in the bar above the canvas, not just here.")

            st.markdown("---")
            st.markdown("**Project File**")
            st.caption("Save every adjustment (plus the source image and any layer) into one file you can reopen later.")
            if st.button("💾 Save Project", key="proj-save-btn"):
                layer_workspace_json = None
                layer_readable = True
                if layer_bytes:
                    try:
                        overlay = Image.open(io.BytesIO(layer_bytes)).convert("RGBA")
                    except OSError:
                        # Same "don't show a raw traceback for a bad file" fix as the main upload
                        # above -- deliberately doesn't call st.stop() here, unlike that one: this
                        # runs deep inside tools_col, and stopping the script early would abort
                        # before canvas_col's widgets are instantiated this pass, resetting them
                        # (the exact bug class _request_rerun() exists to prevent). Skipping just
                        # the save, below, keeps the rest of the script running normally instead.
                        st.error("⚠️ Couldn't read the layer image — it may be corrupted. Remove it in the Layers tab and try again.")
                        layer_readable = False
                    else:
                        workspace = LayerWorkspace(source).add(WorkspaceLayer("Overlay", overlay, l_opacity, l_blend))
                        layer_workspace_json = workspace.to_json()
                if layer_readable:
                    project_recipe = _build_recipe()
                    project = ProjectBundle(source_name=upload_name, source_bytes=upload_bytes, recipe=project_recipe, layer_workspace=layer_workspace_json)
                    proj_buf = io.BytesIO()
                    project.save(proj_buf)
                    st.toast("Project packaged — click Download to save it", icon="💾")
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
                    st.session_state["lib-cursor"] = 0
                    st.toast(f"Indexed {len(st.session_state.records)} photo(s)", icon="✅")
                rec = st.session_state.get("records", [])
                q = st.text_input("Search", key="lib-q-in")
                r_min = st.slider("Min Stars", 0, 5, 0, key="lib-r-in")
                st.dataframe(PhotoCatalog(lib_db).search(q, r_min) if rec else [], use_container_width=True)

                if rec:
                    st.markdown("**Rate & Flag**")
                    cursor = max(0, min(st.session_state.get("lib-cursor", 0), len(rec) - 1))
                    current_path = rec[cursor]["path"]
                    catalog = PhotoCatalog(lib_db)
                    existing = catalog.get(current_path)
                    is_rejected = "rejected" in existing["tags"]

                    st.caption(f"{cursor + 1} / {len(rec)} — {Path(current_path).name}" + (" — 🚫 rejected" if is_rejected else ""))
                    try:
                        st.image(current_path, width=240)
                    except Exception:
                        st.caption("Preview unavailable.")

                    with st.container(key="lib-rate-controls"):
                        nav_prev, nav_next = st.columns(2)
                        if nav_prev.button("⬅ Prev", key="lib-prev-btn", use_container_width=True):
                            st.session_state["lib-cursor"] = (cursor - 1) % len(rec)
                            _request_rerun()
                        if nav_next.button("Next ➡", key="lib-next-btn", use_container_width=True):
                            st.session_state["lib-cursor"] = (cursor + 1) % len(rec)
                            _request_rerun()

                        st.caption("Rating (keys 1-5, 0 to clear, X to reject, ←/→ to move)")
                        star_cols = st.columns(5)
                        for i in range(5):
                            label = "⭐" if i < existing["rating"] else "☆"
                            if star_cols[i].button(label, key=f"lib-star-{i + 1}", use_container_width=True):
                                catalog.upsert(current_path, rating=i + 1, tags=existing["tags"])
                                _request_rerun()

                        clear_col, reject_col = st.columns(2)
                        if clear_col.button("Clear rating", key="lib-clear-btn", use_container_width=True):
                            catalog.upsert(current_path, rating=0, tags=existing["tags"])
                            _request_rerun()
                        reject_label = "🚫 Un-reject" if is_rejected else "🚫 Reject"
                        if reject_col.button(reject_label, key="lib-reject-btn", use_container_width=True):
                            new_tags = [t for t in existing["tags"] if t != "rejected"] if is_rejected else existing["tags"] + ["rejected"]
                            catalog.upsert(current_path, rating=existing["rating"], tags=new_tags)
                            _request_rerun()

                        visible_tags = ", ".join(t for t in existing["tags"] if t != "rejected")
                        tags_text = st.text_input("Tags (comma-separated)", value=visible_tags, key=f"lib-tags-{cursor}")
                        if st.button("Save tags", key="lib-tags-save-btn"):
                            new_tags = [t.strip() for t in tags_text.split(",") if t.strip()]
                            if is_rejected:
                                new_tags.append("rejected")
                            catalog.upsert(current_path, rating=existing["rating"], tags=new_tags)
                            st.toast("Tags saved", icon="🏷️")

                    components.html(_LIBRARY_KEYBOARD_SHORTCUTS_HTML, height=0)
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
        # (Lens -> Tone -> Color -> Detail -> Repair -> Layers -> Masks -> Portrait -> Crop, matching
        # the tab order above) runs once, here, on app/backend.py -- see that module's process_image()
        # for the mirrored order.
        #
        # Note: the deferred-rerun consume check for _request_rerun() (used by tab bodies above)
        # deliberately does NOT live here -- canvas_col, below, still has its own widgets left to
        # instantiate this pass (zoom, split view, pan, export, Undo/Redo/Reset), and rerunning
        # before they render would garbage-collect them exactly like the bug _request_rerun() exists
        # to prevent (a real instance of this was found in review). See the check at the very bottom
        # of this file, after canvas_col.

        options = {
            "lens_distortion": lens_distortion,
            "lens_vignette_amount": lens_vignette_amount,
            "lens_vignette_midpoint": lens_vignette_midpoint,
            "lens_chromatic_aberration": lens_chromatic_aberration,
            "exposure": exp_val,
            "contrast": cont_val,
            "highlights": high_val,
            "shadows": shad_val,
            "preset": preset_choice,
            "temperature": temp_val,
            "tint": tint_val,
            "saturation": sat_val,
            "hsl_bands": hsl_bands,
            "curve": [(0, c_black), (64, c_low), (128, c_mid), (192, c_high), (255, c_white)] if use_curve else None,
            "texture": text_val,
            "clarity": clar_val,
            "denoise": denoise_val,
            "sharpen": sharp_en,
            "removal_radius": removal_radius,
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

        removal_mask_bytes = None
        if removal_active:
            removal_buffer = io.BytesIO()
            removal_combined_mask.save(removal_buffer, format="PNG")
            removal_mask_bytes = removal_buffer.getvalue()

        try:
            edited_image = _cached_process_via_backend(upload_bytes, upload_name, options, local_mask_bytes, layer_bytes, removal_mask_bytes)
        except RuntimeError as error:
            st.error(f"⚠️ {error}")
            st.stop()

    with canvas_col:
        # Canvas Header Bar
        top_c1, top_undo, top_redo, top_reset, top_c2, top_c3 = st.columns([0.95, 0.18, 0.18, 0.18, 0.75, 0.95], gap="small")
        with top_c1:
            st.markdown(f'<div class="panel-header-title">🖼️ Live View: <span>{upload_name} ({edited_image.width}×{edited_image.height})</span></div>', unsafe_allow_html=True)
        with top_undo:
            if st.button("↶", key="hist-undo-btn", disabled=not history.can_undo, help="Undo", use_container_width=True):
                st.session_state["_pending_restore"] = history.undo() or {}
                _request_rerun()
        with top_redo:
            if st.button("↷", key="hist-redo-btn", disabled=not history.can_redo, help="Redo", use_container_width=True):
                st.session_state["_pending_restore"] = history.redo() or {}
                _request_rerun()
        with top_reset:
            # Resets exactly the same "recipe" fields Undo/Redo operate on (see _build_recipe), so it
            # deliberately leaves an uploaded layer/RAW source alone -- only the adjustment values reset.
            if st.button("↺", key="hist-reset-btn", help="Reset all adjustments to defaults", use_container_width=True):
                st.session_state["_pending_restore"] = {**RECIPE_DEFAULTS, "mask_layers": [], "removal_mask_layers": []}
                _request_rerun()
        with top_c2:
            zoom_level = st.selectbox("Zoom View", [1.0, 0.5, 1.5, 2.0, 4.0], format_func=lambda z: f"🔍 {int(z*100)}%" if z>=1 else f"🔍 {int(z*100)}% (Fit)", key="cv-zoom", label_visibility="collapsed")
        with top_c3:
            compare_mode = st.checkbox("⇄ Split View", value=False, key="cv-split", help="Drag to compare before/after")
            show_original = st.checkbox("👁️ Original", value=False, key="cv-orig", disabled=compare_mode)

        if zoom_level > 1.0:
            pan_c1, pan_c2 = st.columns(2)
            with pan_c1:
                pan_x = st.slider("Pan Horizontal", 0.0, 1.0, 0.5, 0.01, key="cv-panx")
            with pan_c2:
                pan_y = st.slider("Pan Vertical", 0.0, 1.0, 0.5, 0.01, key="cv-pany")

        # Main Image Canvas
        if compare_mode:
            # Split View: a draggable before/after slider (see _before_after_html) rather than a
            # static side-by-side -- takes priority over the plain "Original" toggle since it
            # already shows both. The two previews share the same zoom/pan as the normal view; if
            # a crop left them different sizes, resize "before" to match so the overlay lines up.
            before_preview = zoom_preview(source, zoom_level, (pan_x, pan_y))
            after_preview = zoom_preview(edited_image, zoom_level, (pan_x, pan_y))
            if before_preview.size != after_preview.size:
                before_preview = before_preview.resize(after_preview.size, Image.Resampling.LANCZOS)
            compare_html, compare_height = _before_after_html(before_preview, after_preview)
            components.html(compare_html, height=compare_height + 10)
        else:
            st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
            if show_original:
                st.image(preview_image(source), caption="Original Untouched", use_container_width=True)
            else:
                st.image(zoom_preview(edited_image, zoom_level, (pan_x, pan_y)), caption="Current Edited Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Histogram -- always reflects edited_image (the backend-processed result), not whichever
        # view (Original / Split) happens to be toggled above, since that's the reference exposure
        # decisions are actually made against.
        histogram_bins = 64
        histogram = compute_histogram(edited_image, bins=histogram_bins)
        # st.line_chart plots a dict's values against a plain 0..N-1 index by default, which would
        # label this axis in bin numbers instead of the pixel brightness (0-255) each bin covers --
        # give it an explicit "Pixel Value" column (each bin's center) to chart against instead.
        histogram["Pixel Value"] = [round(255 * (i + 0.5) / histogram_bins) for i in range(histogram_bins)]
        st.line_chart(histogram, x="Pixel Value", y=["Red", "Green", "Blue"], height=90, color=["#f87171", "#4ade80", "#60a5fa"])

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

# A tab body or canvas_col control above may have called _request_rerun() instead of st.rerun()
# directly (adding/removing a mask selection, Auto Enhance, committing the curve editor, importing
# presets, Undo/Redo/Reset, Library Catalog's Prev/Next/rating buttons -- see that function's
# docstring). Actually rerunning only here, at the very end of the script, now that every widget in
# every tab AND in canvas_col has been instantiated this pass, is what stops Streamlit from
# garbage-collecting a not-yet-rendered widget's session_state and silently resetting it to its
# `value=` default.
if st.session_state.pop("_rerun_pending", False):
    st.rerun()
