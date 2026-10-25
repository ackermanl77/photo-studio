"""The on-canvas editors -- the before/after slider, tone curve, paint/brush, click picker (Magic
Wand, white balance, zoom navigator), free-form crop, Pen tool and zoomed pan viewer -- as plain
functions returning (markup, height) for st.components.v1.html().

Each editor is a static component in app/ui/static/editors/ (F14): `<name>.html` holds its markup and
CSS with `{{field}}` slots, `<name>.js` its script, and `commit.js` the commit helper they share. A
builder here only computes the values: HTML fields (already escaped -- labels go through `_tx()`,
i.e. app.i18n's `t()` plus HTML escaping) and one JSON `CONFIG` object the script reads (initial
state, sizes, the hidden field's selector, translated strings it builds at run time). `_render()`
puts them together as one self-contained page, because components.html() can't load files next to
it; the markup itself never contains Python-generated JavaScript any more.

Every editor commits back to Python through a hidden st.text_area plus a synthetic Ctrl+Enter
(`commitToStreamlit()` in commit.js) -- see CLAUDE.md for why that is the one reliable cross-iframe
channel in this Streamlit version. Streamlit's bidirectional component API (a component folder with
its own frontend build) was considered for F14 and not adopted: it would need a JS build step and a
second commit path, while this channel is verified live and covered under Node.

Keyboard (F11): every editor is focusable (role="application" plus an aria label). Arrow keys nudge
the selected crop corner or box, Pen anchor (Alt: its curve handle), curve point, or the last
radial/gradient in the paint editor (Shift = 10x); [ and ] change the paint brush size; arrow keys
pan the zoomed canvas viewer. A burst of key presses commits once, KEY_COMMIT_DELAY_MS after the
last one, through the same hidden text_area as a pointer edit -- the Python side sees the same
payloads either way.
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image

from app.i18n import t
from app.ui.theme import editor_style

# One commit per burst of arrow-key presses rather than one Streamlit rerun per key press.
KEY_COMMIT_DELAY_MS = 350

EDITORS_DIR = Path(__file__).resolve().parent / "ui" / "static" / "editors"
_FIELD = re.compile(r"\{\{(\w+)\}\}")


def _tx(text: str) -> str:
    """`text` in the interface language (app.i18n), escaped for HTML text and attributes."""
    return html.escape(t(text))


@lru_cache(maxsize=None)
def _asset(name: str) -> str:
    return (EDITORS_DIR / name).read_text(encoding="utf-8")


def _render(name: str, fields: dict[str, object], config: dict | None = None) -> str:
    """`<name>.html` with its `{{field}}` slots filled from `fields` (inserted as given: callers pass
    escaped text), then one script: `const CONFIG = ...`, commit.js and `<name>.js`. Every slot must
    be filled and every field used, so a renamed slot fails loudly instead of rendering `{{x}}`."""
    markup = _asset(f"{name}.html")
    used = set(_FIELD.findall(markup))
    missing, unused = used - set(fields), set(fields) - used
    if missing or unused:
        raise KeyError(f"{name}.html: missing {sorted(missing)}, unused {sorted(unused)}")
    markup = _FIELD.sub(lambda match: str(fields[match.group(1)]), markup)
    payload = json.dumps({"keyCommitDelay": KEY_COMMIT_DELAY_MS, **(config or {})}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")  # never closes the <script> early
    script = f"const CONFIG = {payload};\n{_asset('commit.js')}\n{_asset(f'{name}.js')}"
    return editor_style() + markup + f"<script>\n{script}</script>\n"


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _display_size(background: Image.Image) -> tuple[int, int]:
    """The editor's drawing size: the background's own, which the caller already fitted to where the
    editor goes (480 px in the sidebar, larger over the photo -- app.ui.state.editor_box())."""
    width, height = background.size
    return width, height if width else 320


def _image_view_html(image: Image.Image, label: str) -> tuple[str, int]:
    """A still image drawn over the photo like an editor (the Masks and Repair overlay previews),
    centred at its own size -- already fitted to the box by the caller."""
    width, height = _display_size(image)
    markup = (f'<img src="data:image/png;base64,{_png_b64(image)}" alt="{_tx(label)}" '
              f'style="display:block;width:{width}px;max-width:100%;height:auto;margin:0 auto;border-radius:4px;">')
    return editor_style() + markup, height + 20


def _look_gallery_html(groups: list[tuple[str, list[dict]]], key_prefix: str) -> tuple[str, int]:
    """The look gallery (UI5): `groups` are (title, cards); a card is {"kind", "name", "title",
    "thumb", "preview", "current"} with base64 JPEGs. Hovering a card shows its preview on the
    canvas, clicking applies it -- committed as {"kind", "name", "at"} into the hidden text_area in
    st.container(key=f"{key_prefix}-hidden")."""
    fields = {
        "label_root": _tx("Looks and presets"), "label_grid": _tx("Show as a grid"), "label_list": _tx("Show as a list"),
        "label_hint": _tx("Point at one to preview it on the photo; click to apply."),
    }
    config = {
        "hiddenSelector": f".st-key-{key_prefix}-hidden", "previewLabel": t("Preview: {0}"),
        "groups": [{"title": t(title), "cards": cards} for title, cards in groups if cards],
    }
    # Tall enough for the list view (the taller one past a few cards): a row each, plus titles.
    count = sum(len(cards) for _title, cards in groups)
    height = 40 + 30 * sum(1 for _title, cards in groups if cards) + max(84 * -(-count // 3), 50 * count)
    return _render("looks", fields, config), height


def _before_after_html(before: Image.Image, after: Image.Image) -> tuple[str, int]:
    """A draggable before/after comparison: a single range input (or a drag on the photo) drives a
    CSS clip-path on the "before" image layered over the "after" image, split side by side or top
    and bottom. Static before_after.html/.js via
    st.components.v1.html -- no component library dependency, consistent with keeping this app's
    UI free of extra dependencies after removing streamlit-drawable-canvas.

    `before` and `after` must already be the same pixel size (the caller resizes if a crop made
    them differ) -- that's what makes the overlay line up.
    """
    # Sized for the photo's place (about 1000 px wide, at most 600 tall), and no wider than that
    # height allows -- sized by width alone, a landscape photo ran past the iframe's bottom.
    width, height = after.size
    display_height = max(220, min(600, int(1000 * height / width))) if width else 320
    display_w = int(display_height * width / height) if height else 1000
    fields = {
        "after_b64": _png_b64(after), "before_b64": _png_b64(before), "display_w": display_w,
        "label_before": _tx("BEFORE"), "label_after": _tx("AFTER"),
        "label_range": _tx("Before/after comparison position"),
        "label_dir": _tx("Split side by side or top and bottom"),
    }
    return _render("before_after", fields), display_height


_MASTER_CURVE_KEYS = ("curve_black", "curve_shadows", "curve_mid", "curve_highlights", "curve_white")


def _curve_editor_html(points: list[tuple[int, int]], bounds: list[tuple[int, int]], extra_points: list[tuple[int, int]] | None = None,
                       keys: tuple[str, ...] = _MASTER_CURVE_KEYS, color: str = "var(--ps-accent)") -> str:
    """A draggable RGB tone curve: an inline SVG with five fixed points (Blacks/Shadows/Midtones/
    Highlights/Whites, matching the five sliders below it), each vertically draggable within its
    own [min, max], plus any number of user-added "extra" points -- click empty space on the curve
    to add one, drag it freely in both X and Y, double-click to remove it. `PhotoProcessor.
    apply_curves()` already accepted an arbitrary point list; the 5-fixed-point limit was purely a
    UI one (see "Known Issues" in docs/PROJECT_STATUS.md before this). Dragging/adding/removing
    only updates the curve's own SVG locally; each of those three actions commits on its own
    (point release, the add-click itself, the remove-double-click).

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

    So instead of touching the sliders at all, commit writes the whole curve as JSON into a hidden
    `st.text_area` (`st.container(key="curve-hidden")`, same Ctrl+Enter trick used by
    `_paint_editor_html`, confirmed reliable) using the RECIPE_WIDGET_KEYS names
    (curve_black/curve_shadows/curve_mid/curve_highlights/curve_white/curve_enabled) plus
    `curve_extra_points` (a plain list, handled like `mask_layers` rather than through
    RECIPE_WIDGET_KEYS -- see `_build_recipe`/`_restore_recipe_to_state`). Python reads it and
    stores it in `st.session_state["_pending_restore"]` -- the exact same "write before any widget
    instantiates, then rerun" mechanism Undo/Redo/Reset already use to update a slider's value from
    outside the widget itself, which sidesteps the whole reliability question since it never
    depends on a script-dispatched DOM event reaching a slider at all.

    `keys` are the recipe keys of the five fixed points (default: the master curve's); a per-channel
    curve (F13) passes its own and `color`, and gets no free-form extra points (they belong to the
    master curve, `curve_extra_points`).
    """
    allow_extra = tuple(keys) == _MASTER_CURVE_KEYS
    extra_points = (extra_points or []) if allow_extra else []
    margin_left, margin_top, plot_size = 34, 12, 210
    svg_w, svg_h = margin_left + plot_size + 12, margin_top + plot_size + 12

    def _to_px(x: float, y: float) -> tuple[float, float]:
        return (margin_left + (x / 255) * plot_size, margin_top + plot_size - (y / 255) * plot_size)

    all_points_sorted = sorted(list(points) + list(extra_points), key=lambda point: point[0])
    poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in (_to_px(x, y) for x, y in all_points_sorted))
    fixed_circles = "\n".join(
        f'<circle class="pt fixed" data-idx="{i}" cx="{px:.1f}" cy="{py:.1f}" r="7" tabindex="0" role="slider" '
        f'aria-label="{html.escape(t("Curve point {0} of 5 -- Up/Down arrows adjust it").format(i + 1))}"></circle>'
        for i, (px, py) in enumerate(_to_px(x, y) for x, y in points)
    )
    diag_x0, diag_y0 = _to_px(0, 0)
    diag_x1, diag_y1 = _to_px(255, 255)
    hint = (_tx("Click an empty spot on the curve to add a point, drag it, double-click to remove it.") if allow_extra
            else _tx("Drag a point up or down."))
    fields = {
        "color": html.escape(color), "svg_w": svg_w, "svg_h": svg_h,
        "label_editor": _tx("Tone curve editor. Tab to a point, then use the Up and Down arrows."),
        "margin_left": margin_left, "margin_top": margin_top, "plot_size": plot_size,
        "bg_cursor": "copy" if allow_extra else "default",
        "diag_x0": f"{diag_x0:.1f}", "diag_y0": f"{diag_y0:.1f}", "diag_x1": f"{diag_x1:.1f}", "diag_y1": f"{diag_y1:.1f}",
        "poly": poly, "fixed_circles": fixed_circles,
        "hint": f"{hint} {_tx('Keyboard: Tab to a point, Up/Down (Shift = x10).')}",
    }
    config = {
        "marginLeft": margin_left, "marginTop": margin_top, "plotSize": plot_size, "svgW": svg_w,
        "bounds": [list(b) for b in bounds], "points": [list(p) for p in points],
        "extraPoints": [list(p) for p in extra_points], "keys": list(keys), "allowExtra": allow_extra,
    }
    return _render("curve", fields, config)


def _paint_editor_html(background: Image.Image, key_prefix: str, preload_mask_b64: str | None = None, commit_label: str = "➕ Add to selections") -> tuple[str, int]:
    """An on-canvas selection editor: brush (paint/erase freehand), radial (drag out a circle),
    and gradient (drag out a direction), all painted onto one HTML canvas as solid white at
    varying alpha -- alpha *is* the mask value, so committing just base64-encodes the canvas as a
    PNG and that becomes a MaskLayer's `params["mask_b64"]` (decoded by `_decode_painted_mask`).
    Committing writes into the hidden `st.text_area` inside `st.container(key=f"{key_prefix}-
    paint-hidden")` (CSS-hidden -- see the app's <style> block) via the same native-value-setter
    trick as `_curve_editor_html`, so Python notices it on the next rerun (see the "Painted"
    branch in _render_selection_stack for a new selection, or the per-layer "Edit Painting"
    section for an existing one) -- there's no separate native Streamlit "Add" button to click
    after drawing, the in-canvas button *is* the commit. No component library dependency, same
    approach as the other editors in this file; avoids a repeat of the streamlit-drawable-canvas
    dead-weight episode.

    `preload_mask_b64`, when given (re-editing an existing "Painted" selection instead of
    starting a fresh one), is a base64 PNG at this same canvas resolution -- the exact bytes
    `_decode_painted_mask` would read back out -- drawn onto the canvas before any user input, so
    painting continues from what was already there instead of a blank slate. `commit_label` lets
    the re-edit path read "💾 Update Painting" instead of "➕ Add to selections" without a second,
    near-duplicate template.
    """
    display_w, display_h = _display_size(background)
    fields = {
        "label_root": _tx("Selection painter. [ and ] change the brush size, Ctrl+Z undoes the last stroke; arrow keys move the last radial or gradient (Alt: its end point, Shift: 10x)."),
        "label_brush": _tx("🖌 Brush"), "label_radial": _tx("⭕ Radial"), "label_gradient": _tx("📐 Gradient"),
        "label_erase": _tx("🧹 Erase"), "label_clear": _tx("✖ Clear"), "label_size": _tx("Size"), "label_undo": _tx("↶ Undo stroke"),
        "label_size_input": _tx("Brush size ([ and ] keys)"),
        "display_w": display_w, "display_h": display_h, "bg_b64": _png_b64(background),
        "label_commit": _tx(commit_label),
        "label_keys": _tx("Keys: [ / ] brush size, Ctrl+Z undoes the last stroke. After a radial or gradient: arrows move it, Alt+arrows move its end point, Shift = x10."),
    }
    config = {"hiddenSelector": f".st-key-{key_prefix}-paint-hidden", "preloadMaskB64": preload_mask_b64 or ""}
    return _render("paint", fields, config), display_h + 130


def _wand_picker_html(background: Image.Image, key_prefix: str, hint: str = "Click anywhere on the photo to sample that color.") -> tuple[str, int]:
    """Magic Wand's on-canvas "click to sample a color" picker: a single click on the preview
    reports its fractional (x, y) position back through the same hidden-`st.text_area` commit
    pattern `_paint_editor_html`/`_curve_editor_html` use (see `_curve_editor_html`'s docstring for
    why a script-dispatched event can't just touch a native widget directly). One click, no
    drawing state to track, so this is its own small component rather than another branch bolted
    onto `_paint_editor_html`.
    """
    display_w, display_h = _display_size(background)
    fields = {"display_w": display_w, "bg_b64": _png_b64(background), "hint": _tx(hint)}
    return _render("picker", fields, {"hiddenSelector": f".st-key-{key_prefix}-wand-hidden"}), display_h + 60


def _crop_editor_html(
    background: Image.Image, key_prefix: str, box: tuple[float, float, float, float], full_size: tuple[int, int],
    snap_x: list[float] | None = None, snap_y: list[float] | None = None,
) -> tuple[str, int]:
    """A draggable free-form crop rectangle directly on the photo -- replaces the old pair of
    plain X/Y range sliders (see "Known Issues" in docs/PROJECT_STATUS.md, which explicitly called
    that a deliberate stand-in to avoid reintroducing a third-party UI component dependency after
    streamlit-drawable-canvas was removed as dead weight). No new dependency here either: two
    round corner-drag handles plus a move-by-dragging-the-body rectangle, all plain absolutely-
    positioned `<div>`s over the preview image, following the same "commit into a hidden
    st.text_area via the native-value-setter trick" pattern as `_paint_editor_html`/
    `_wand_picker_html`/`_curve_editor_html`.

    `box` is `(x0, y0, x1, y1)` as fractions of the image (0..1 each) -- deliberately not pixel
    coordinates, so the stored crop survives switching to a differently-sized photo (the exact gap
    the old range-slider version's Known Issues entry called out: its pixel-bounded slider values
    reset to the full frame on reload because they couldn't safely be interpreted against a new
    image's dimensions). `full_size` is the *source* image's real (width, height) -- distinct from
    `background`'s own (possibly downscaled-for-display) size -- purely so the on-canvas dimensions
    label reads real pixel counts rather than the preview's.

    `snap_x`/`snap_y` are the caller's active grid-line/guide fractions (see the Guides & Grid panel
    in canvas_col) -- only the two corner handles snap to them, not a body drag/move; snapping a
    translate against two independent edges per axis is a meaningfully bigger piece of geometry for
    comparatively little extra value over "frame this edge exactly on a guide," which the corners
    already cover, so it's left as a known, deliberate gap (see "Known Issues").
    """
    display_w, display_h = _display_size(background)
    full_w, full_h = full_size
    x0, y0, x1, y1 = box
    fields = {
        "label_wrap": _tx("Crop box. Tab to the box or a corner, then use the arrow keys (Shift = 10 pixels)."),
        "display_w": display_w, "bg_b64": _png_b64(background),
        "label_rect": _tx("Crop box -- arrow keys move it"),
        "label_tl": _tx("Top-left crop corner -- arrow keys move it"),
        "label_br": _tx("Bottom-right crop corner -- arrow keys move it"),
    }
    config = {
        "hiddenSelector": f".st-key-{key_prefix}-hidden",
        "box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, "fullW": full_w, "fullH": full_h,
        "snapX": list(snap_x or []), "snapY": list(snap_y or []),
        "text": {"crop": t("Crop:"), "hint": t("drag a corner to resize, drag inside to move.")},
    }
    return _render("crop", fields, config), display_h + 40


def _pen_editor_html(background: Image.Image, key_prefix: str, points: list[dict], closed: bool) -> tuple[str, int]:
    """The Pen tool: click to add anchor points on the photo, drag a point to move it, drag its
    amber handle to curve the path through that point (a symmetric "smooth point" handle -- see
    app.vector_shapes's module docstring for why it's not independent in/out handles), double-click
    a point to remove it. Same "commit into a hidden st.text_area via the native-value-setter
    trick" pattern as every other on-canvas editor in this file (_paint_editor_html/
    _wand_picker_html/_curve_editor_html/_crop_editor_html) -- confirmed the only reliable
    cross-iframe commit mechanism in this Streamlit version.

    Anchor/handle dots are plain absolutely-positioned `<div>`s (percentage left/top, exactly like
    _crop_editor_html's corner handles) rather than SVG circles inside a percentage-viewBox SVG --
    a 100x100 viewBox stretched with `preserveAspectRatio="none"` onto a non-square display box
    would visually squash circles into ellipses. The connecting path/handle-guide-lines don't need
    to be perfect circles, so they stay as an overlaid SVG using that same simple 0-100 coordinate
    space, avoiding any pixel/fraction conversion math for either layer.

    The "add a point" click listener is attached to the background `<img>` specifically, not the
    whole wrapper div -- anchor/handle dots are siblings layered on top of it (not descendants), so
    a click landing on one of them never bubbles into "add a new point at this spot" the way
    attaching the listener to a shared ancestor would.
    """
    display_w, display_h = _display_size(background)
    fields = {
        "label_wrap": _tx("Pen path. Tab to a point; arrow keys move it, Alt+arrows move its curve handle, Shift = 10x, Delete removes it."),
        "display_w": display_w, "bg_b64": _png_b64(background),
        "label_help": _tx("Click the photo to add a point. Drag a point to move it, drag its amber handle to curve the path. Double-click a point to remove it. Keys: Tab to a point, arrows nudge it (Alt: its handle, Shift: x10), Delete removes it."),
    }
    config = {
        "hiddenSelector": f".st-key-{key_prefix}-hidden", "points": list(points), "closed": bool(closed),
        "text": {"point": t("Path point {0} of {1}")},
    }
    return _render("pen", fields, config), display_h + 60


def _liquify_editor_html(background: Image.Image, key_prefix: str, size: float, max_points: int) -> tuple[str, int]:
    """Liquify (F23): drag on the photo to record one stroke, committed as {"points": [[x, y], ...]}
    (fractions of the frame) when the pointer lifts -- the tool, size and strength come from the
    widgets beside the editor, and the warp itself is rendered by the backend (app.liquify), so the
    editor only draws the brush outline (`size`, a fraction of the short edge) and the stroke being
    recorded. Arrow keys move the brush (Shift = 10x); Space/Enter applies one stamp there."""
    display_w, display_h = _display_size(background)
    fields = {
        "label_wrap": _tx("Liquify brush. Drag on the photo to warp it; arrow keys move the brush (Shift = 10x), Space applies it there."),
        "display_w": display_w, "display_h": display_h, "bg_b64": _png_b64(background),
        "label_help": _tx("Drag on the photo to warp it with the chosen tool. Each stroke can be undone."),
    }
    config = {"hiddenSelector": f".st-key-{key_prefix}-hidden", "size": float(size), "maxPoints": int(max_points)}
    return _render("liquify", fields, config), display_h + 40


def _pan_viewer_html(
    view: Image.Image, key_prefix: str, center: tuple[float, float], zoom: float, image_size: tuple[int, int],
    max_display_height: int = 640,
) -> tuple[str, int]:
    """The zoomed (100 % and above) canvas with drag-to-pan (F3): drag the photo and, on release,
    the view's new center is committed as {"x", "y"} fractions into the hidden text_area of
    `key_prefix`'s click picker (the zoom navigator's own field, `.st-key-{key_prefix}-wand-hidden`),
    so the Python side is exactly the navigator's -- same payload, same handling. While dragging
    the image just follows the pointer; the rerun then renders the newly centered region.

    `view` is the zoom_preview() region (already enlarged for 200 %/400 %), `center` the region's
    real center as fractions of the full image (after zoom_viewport()'s edge clamping, so a drag
    from a view pinned against an edge starts from where it really is), `image_size` the full
    image's pixel size. Arrow keys pan by a tenth of the view (Shift: half), committed once per
    burst of presses.
    """
    width, height = view.size
    display_h = min(max_display_height, height)
    image_w, image_h = (max(1, int(value)) for value in image_size)
    fields = {
        "label_wrap": _tx("Zoomed photo. Drag to pan, or use the arrow keys (Shift = faster)."),
        "display_h": display_h, "view_b64": _png_b64(view),
        "label_help": _tx("Drag the photo to pan (arrow keys work too)."),
    }
    config = {
        "hiddenSelector": f".st-key-{key_prefix}-wand-hidden", "viewW": width, "viewH": height,
        "zoom": max(1.0, float(zoom)), "imageW": image_w, "imageH": image_h,
        "center": [float(center[0]), float(center[1])],
    }
    return _render("pan", fields, config), display_h + 30
