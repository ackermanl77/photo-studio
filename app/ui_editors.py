"""The on-canvas editors' HTML/CSS/JS -- the before/after slider, tone curve, paint/brush, click
picker (Magic Wand, white balance, zoom navigator), free-form crop and Pen tool -- as plain
functions returning (markup, height) for st.components.v1.html().

Split out of app/frontend_unified_app.py (F14 in the improvement report), which is a Streamlit
script and can't be imported without rendering; these builders have no Streamlit dependency at all.
Every editor commits back to Python through a hidden st.text_area plus a synthetic Ctrl+Enter --
see CLAUDE.md for why that is the one reliable cross-iframe channel in this Streamlit version.
"""
from __future__ import annotations

import base64
import html
import io
import json

from PIL import Image


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


def _curve_editor_html(points: list[tuple[int, int]], bounds: list[tuple[int, int]], extra_points: list[tuple[int, int]] | None = None) -> str:
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
    """
    extra_points = extra_points or []
    margin_left, margin_top, plot_size = 34, 12, 210
    svg_w, svg_h = margin_left + plot_size + 12, margin_top + plot_size + 12

    def _to_px(x: float, y: float) -> tuple[float, float]:
        return (margin_left + (x / 255) * plot_size, margin_top + plot_size - (y / 255) * plot_size)

    all_points_sorted = sorted(list(points) + list(extra_points), key=lambda point: point[0])
    poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in (_to_px(x, y) for x, y in all_points_sorted))
    fixed_circles = "\n".join(
        f'<circle class="pt fixed" data-idx="{i}" cx="{px:.1f}" cy="{py:.1f}" r="7"></circle>'
        for i, (px, py) in enumerate(_to_px(x, y) for x, y in points)
    )
    diag_x0, diag_y0 = _to_px(0, 0)
    diag_x1, diag_y1 = _to_px(255, 255)
    points_json = json.dumps(points)
    bounds_json = json.dumps(bounds)
    extra_points_json = json.dumps(extra_points)
    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:6px 8px;">
    <svg id="curve-svg" width="100%" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}"
         preserveAspectRatio="xMinYMin meet" style="touch-action:none;display:block;">
      <rect id="curve-bg" x="{margin_left}" y="{margin_top}" width="{plot_size}" height="{plot_size}"
            fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.15)" style="cursor:copy;"/>
      <line x1="{diag_x0:.1f}" y1="{diag_y0:.1f}" x2="{diag_x1:.1f}" y2="{diag_y1:.1f}"
            stroke="rgba(255,255,255,0.2)" stroke-dasharray="3,3"/>
      <polyline id="curve-line" points="{poly}" fill="none" stroke="#fbbf24" stroke-width="2"/>
      {fixed_circles}
      <g id="extra-points"></g>
    </svg>
    </div>
    <div style="color:#94a3b8;font:11px system-ui;margin-top:4px;padding:0 2px;">
      Click an empty spot on the curve to add a point, drag it, double-click to remove it.
    </div>
    <style>
      #curve-svg .pt.fixed {{ fill:#fbbf24; stroke:#1e293b; stroke-width:2; cursor:ns-resize; }}
      #curve-svg .pt.fixed:hover {{ fill:#fde68a; }}
      #curve-svg .pt.extra {{ fill:#60a5fa; stroke:#1e293b; stroke-width:2; cursor:move; }}
      #curve-svg .pt.extra:hover {{ fill:#93c5fd; }}
    </style>
    <script>
      (function() {{
        const marginLeft = {margin_left}, marginTop = {margin_top}, plotSize = {plot_size}, svgW = {svg_w};
        const bounds = {bounds_json};
        const points = {points_json};
        let extraPoints = {extra_points_json};
        const svg = document.getElementById('curve-svg');
        const bg = document.getElementById('curve-bg');
        const line = document.getElementById('curve-line');
        const extraGroup = document.getElementById('extra-points');
        function toPx(x, y) {{ return [marginLeft + (x / 255) * plotSize, marginTop + plotSize - (y / 255) * plotSize]; }}
        function fromPy(py) {{ return Math.max(0, Math.min(255, ((marginTop + plotSize - py) / plotSize) * 255)); }}
        function fromPx(px) {{ return Math.max(0, Math.min(255, ((px - marginLeft) / plotSize) * 255)); }}
        function redraw() {{
          const all = points.concat(extraPoints).slice().sort((a, b) => a[0] - b[0]);
          line.setAttribute('points', all.map(p => toPx(p[0], p[1]).join(',')).join(' '));
          svg.querySelectorAll('.pt.fixed').forEach((c, i) => {{
            const [px, py] = toPx(points[i][0], points[i][1]);
            c.setAttribute('cx', px); c.setAttribute('cy', py);
          }});
          extraGroup.innerHTML = '';
          extraPoints.forEach((p, i) => {{
            const [px, py] = toPx(p[0], p[1]);
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('class', 'pt extra');
            circle.setAttribute('data-idx', i);
            circle.setAttribute('cx', px); circle.setAttribute('cy', py); circle.setAttribute('r', 7);
            extraGroup.appendChild(circle);
          }});
        }}
        function commit() {{
          try {{
            const payload = JSON.stringify({{
              curve_black: points[0][1], curve_shadows: points[1][1], curve_mid: points[2][1],
              curve_highlights: points[3][1], curve_white: points[4][1], curve_enabled: true,
              curve_extra_points: extraPoints,
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

        // Fixed points: Y-only drag within their own [min, max] -- unchanged from before extra
        // points existed. Their count never changes, so listeners attach once, directly.
        let dragIdx = null;
        svg.querySelectorAll('.pt.fixed').forEach((circle, i) => {{
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

        // Extra points: added/removed dynamically, so handled via event delegation on `svg`
        // (checking e.target) instead of per-circle listeners that would need re-attaching every
        // time redraw() rebuilds the #extra-points group's children.
        let dragExtraIdx = null;
        svg.addEventListener('pointerdown', (e) => {{
          const el = e.target.closest('.pt.extra');
          if (!el) return;
          dragExtraIdx = Number(el.dataset.idx);
          svg.setPointerCapture(e.pointerId);
          e.preventDefault();
          e.stopPropagation();
        }});
        svg.addEventListener('pointermove', (e) => {{
          if (dragExtraIdx === null) return;
          const rect = svg.getBoundingClientRect();
          const scale = svgW / rect.width;
          const px = (e.clientX - rect.left) * scale;
          const py = (e.clientY - rect.top) * scale;
          extraPoints[dragExtraIdx] = [Math.round(fromPx(px)), Math.round(fromPy(py))];
          redraw();
        }});
        function endExtraDrag() {{
          if (dragExtraIdx === null) return;
          dragExtraIdx = null;
          commit();
        }}
        svg.addEventListener('pointerup', endExtraDrag);
        svg.addEventListener('pointerleave', endExtraDrag);
        svg.addEventListener('dblclick', (e) => {{
          const el = e.target.closest('.pt.extra');
          if (!el) return;
          extraPoints.splice(Number(el.dataset.idx), 1);
          redraw();
          commit();
        }});

        // Click empty background (not a drag-release from a point, which targets a circle
        // element instead of the rect) adds a new extra point at the clicked position.
        bg.addEventListener('click', (e) => {{
          const rect = svg.getBoundingClientRect();
          const scale = svgW / rect.width;
          const px = (e.clientX - rect.left) * scale;
          const py = (e.clientY - rect.top) * scale;
          extraPoints.push([Math.round(fromPx(px)), Math.round(fromPy(py))]);
          redraw();
          commit();
        }});

        redraw();
      }})();
    </script>
    """


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
    width, height = background.size
    display_w = min(480, width)
    display_h = int(display_w * height / width) if width else 320

    buffer = io.BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    bg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    hidden_selector = f".st-key-{key_prefix}-paint-hidden"
    preload_js = (
        f"""const preload = new Image();
          preload.onload = () => ctx.drawImage(preload, 0, 0, W, H);
          preload.src = 'data:image/png;base64,{preload_mask_b64}';"""
        if preload_mask_b64
        else ""
    )

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
        {commit_label}
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
        {preload_js}

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


def _wand_picker_html(background: Image.Image, key_prefix: str, hint: str = "Click anywhere on the photo to sample that color.") -> tuple[str, int]:
    """Magic Wand's on-canvas "click to sample a color" picker: a single click on the preview
    reports its fractional (x, y) position back through the same hidden-`st.text_area` commit
    pattern `_paint_editor_html`/`_curve_editor_html` use (see `_curve_editor_html`'s docstring for
    why a script-dispatched event can't just touch a native widget directly). One click, no
    drawing state to track, so this is its own small component rather than another branch bolted
    onto `_paint_editor_html`.
    """
    width, height = background.size
    display_w = min(480, width)
    display_h = int(display_w * height / width) if width else 320

    buffer = io.BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    bg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    hidden_selector = f".st-key-{key_prefix}-wand-hidden"

    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:8px;">
      <div style="position:relative;width:100%;max-width:{display_w}px;line-height:0;">
        <img id="wand-img" src="data:image/png;base64,{bg_b64}" role="button" tabindex="0" aria-label="{html.escape(hint)}"
             style="display:block;width:100%;height:auto;border-radius:4px;cursor:crosshair;">
        <div id="wand-marker" style="position:absolute;width:16px;height:16px;margin:-8px 0 0 -8px;
             border:2px solid #fbbf24;border-radius:50%;box-shadow:0 0 0 1px rgba(0,0,0,.6);
             display:none;pointer-events:none;"></div>
      </div>
      <div style="color:#94a3b8;font:12px system-ui;margin-top:6px;">{html.escape(hint)}</div>
    </div>
    <script>
      (function() {{
        const img = document.getElementById('wand-img');
        const marker = document.getElementById('wand-marker');
        img.addEventListener('click', function(e) {{
          const rect = img.getBoundingClientRect();
          const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
          const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
          marker.style.left = (fx * rect.width) + 'px';
          marker.style.top = (fy * rect.height) + 'px';
          marker.style.display = 'block';
          try {{
            const payload = JSON.stringify({{x: fx, y: fy}});
            const container = window.parent.document.querySelector('{hidden_selector}');
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
          }} catch (err) {{}}
        }});
      }})();
    </script>
    """, display_h + 60


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
    width, height = background.size
    display_w = min(480, width)
    display_h = int(display_w * height / width) if width else 320
    full_w, full_h = full_size
    x0, y0, x1, y1 = box

    buffer = io.BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    bg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    hidden_selector = f".st-key-{key_prefix}-hidden"

    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:8px;">
      <div id="crop-wrap" style="position:relative;width:100%;max-width:{display_w}px;line-height:0;touch-action:none;">
        <img src="data:image/png;base64,{bg_b64}" style="display:block;width:100%;height:auto;border-radius:4px;">
        <div id="crop-rect" style="position:absolute;border:2px solid #fbbf24;background:rgba(251,191,36,0.15);cursor:move;box-sizing:border-box;"></div>
        <div id="crop-tl" class="crop-handle" style="cursor:nwse-resize;"></div>
        <div id="crop-br" class="crop-handle" style="cursor:nwse-resize;"></div>
      </div>
      <div id="crop-dims" style="color:#94a3b8;font:12px system-ui;margin-top:6px;"></div>
    </div>
    <style>
      .crop-handle {{ position:absolute; width:16px; height:16px; margin:-8px 0 0 -8px;
                      background:#fbbf24; border:2px solid #1e293b; border-radius:50%; }}
    </style>
    <script>
      (function() {{
        const wrap = document.getElementById('crop-wrap');
        const rectEl = document.getElementById('crop-rect');
        const tl = document.getElementById('crop-tl');
        const br = document.getElementById('crop-br');
        const dimsLabel = document.getElementById('crop-dims');
        const box = {{x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}}};
        const snapX = {json.dumps(list(snap_x or []))};
        const snapY = {json.dumps(list(snap_y or []))};
        const SNAP_THRESHOLD = 0.012;
        let dragMode = null, moveStart = null;

        function snapValue(value, candidates) {{
          for (const candidate of candidates) {{
            if (Math.abs(value - candidate) < SNAP_THRESHOLD) return candidate;
          }}
          return value;
        }}

        function redraw() {{
          rectEl.style.left = (box.x0 * 100) + '%';
          rectEl.style.top = (box.y0 * 100) + '%';
          rectEl.style.width = ((box.x1 - box.x0) * 100) + '%';
          rectEl.style.height = ((box.y1 - box.y0) * 100) + '%';
          tl.style.left = (box.x0 * 100) + '%';
          tl.style.top = (box.y0 * 100) + '%';
          br.style.left = (box.x1 * 100) + '%';
          br.style.top = (box.y1 * 100) + '%';
          const pxW = Math.round((box.x1 - box.x0) * {full_w});
          const pxH = Math.round((box.y1 - box.y0) * {full_h});
          dimsLabel.textContent = 'Crop: ' + pxW + '×' + pxH + 'px -- drag a corner to resize, drag inside to move.';
        }}
        redraw();

        function fractionalPos(e) {{
          const rect = wrap.getBoundingClientRect();
          const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
          const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
          return [fx, fy];
        }}

        function startDrag(mode) {{
          return function(e) {{
            dragMode = mode;
            wrap.setPointerCapture(e.pointerId);
            if (mode === 'move') {{
              const [fx, fy] = fractionalPos(e);
              moveStart = {{fx: fx, fy: fy, x0: box.x0, y0: box.y0, x1: box.x1, y1: box.y1}};
            }}
            e.preventDefault();
            e.stopPropagation();
          }};
        }}
        tl.addEventListener('pointerdown', startDrag('tl'));
        br.addEventListener('pointerdown', startDrag('br'));
        rectEl.addEventListener('pointerdown', startDrag('move'));

        const MIN_SIZE = 0.02;
        wrap.addEventListener('pointermove', function(e) {{
          if (!dragMode) return;
          let [fx, fy] = fractionalPos(e);
          if (dragMode === 'tl' || dragMode === 'br') {{
            // Snap only the corner handles -- not a body drag/move, see the docstring above.
            fx = snapValue(fx, snapX);
            fy = snapValue(fy, snapY);
          }}
          if (dragMode === 'tl') {{
            box.x0 = Math.min(fx, box.x1 - MIN_SIZE);
            box.y0 = Math.min(fy, box.y1 - MIN_SIZE);
          }} else if (dragMode === 'br') {{
            box.x1 = Math.max(fx, box.x0 + MIN_SIZE);
            box.y1 = Math.max(fy, box.y0 + MIN_SIZE);
          }} else if (dragMode === 'move') {{
            const w = moveStart.x1 - moveStart.x0, h = moveStart.y1 - moveStart.y0;
            let nx0 = Math.max(0, Math.min(1 - w, moveStart.x0 + (fx - moveStart.fx)));
            let ny0 = Math.max(0, Math.min(1 - h, moveStart.y0 + (fy - moveStart.fy)));
            box.x0 = nx0; box.y0 = ny0; box.x1 = nx0 + w; box.y1 = ny0 + h;
          }}
          redraw();
        }});

        function commit() {{
          try {{
            const payload = JSON.stringify(box);
            const container = window.parent.document.querySelector('{hidden_selector}');
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
          }} catch (err) {{}}
        }}
        function endDrag() {{
          if (!dragMode) return;
          dragMode = null;
          commit();
        }}
        wrap.addEventListener('pointerup', endDrag);
        wrap.addEventListener('pointerleave', endDrag);
      }})();
    </script>
    """, display_h + 40


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
    width, height = background.size
    display_w = min(480, width)
    display_h = int(display_w * height / width) if width else 320

    buffer = io.BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    bg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    hidden_selector = f".st-key-{key_prefix}-hidden"
    points_json = json.dumps(points)
    closed_json = json.dumps(bool(closed))

    return f"""
    <div style="background:#0f172a;border-radius:8px;padding:8px;">
      <div id="pen-wrap" style="position:relative;width:100%;max-width:{display_w}px;line-height:0;touch-action:none;">
        <img id="pen-bg" src="data:image/png;base64,{bg_b64}" style="display:block;width:100%;height:auto;border-radius:4px;">
        <svg id="pen-svg" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;" viewBox="0 0 100 100" preserveAspectRatio="none">
          <path id="pen-path" d="" fill="none" stroke="#60a5fa" stroke-width="0.6"></path>
          <g id="pen-handle-lines" stroke="rgba(251,191,36,0.7)" stroke-width="0.3"></g>
        </svg>
        <div id="pen-anchors"></div>
        <div id="pen-handles"></div>
      </div>
      <div style="color:#94a3b8;font:12px system-ui;margin-top:6px;">Click the photo to add a point. Drag a point to move it, drag its amber handle to curve the path. Double-click a point to remove it.</div>
    </div>
    <style>
      #pen-wrap .pen-anchor {{ position:absolute; width:14px; height:14px; margin:-7px 0 0 -7px;
                      background:#60a5fa; border:2px solid #1e293b; border-radius:50%; cursor:move; }}
      #pen-wrap .pen-handle {{ position:absolute; width:10px; height:10px; margin:-5px 0 0 -5px;
                      background:#fbbf24; border:2px solid #1e293b; border-radius:50%; cursor:crosshair; }}
    </style>
    <script>
      (function() {{
        const wrap = document.getElementById('pen-wrap');
        const bg = document.getElementById('pen-bg');
        const pathEl = document.getElementById('pen-path');
        const handleLines = document.getElementById('pen-handle-lines');
        const anchorLayer = document.getElementById('pen-anchors');
        const handleLayer = document.getElementById('pen-handles');
        let points = {points_json};
        const closed = {closed_json};
        let dragMode = null;  // {{type: 'anchor'|'handle', idx: number}}

        function fractionalPos(e) {{
          const rect = wrap.getBoundingClientRect();
          const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
          const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
          return [fx * 100, fy * 100];
        }}

        // Mirrors app.vector_shapes.path_outline_points()/_cubic_bezier_points() exactly -- the
        // preview drawn while editing and the layer actually rendered by the backend must trace
        // the same curve, or a point could look right here and land somewhere else once applied.
        function cubicPoints(p0, c1, c2, p1, steps) {{
          const pts = [];
          for (let i = 0; i <= steps; i++) {{
            const t = i / steps, mt = 1 - t;
            const x = mt*mt*mt*p0[0] + 3*mt*mt*t*c1[0] + 3*mt*t*t*c2[0] + t*t*t*p1[0];
            const y = mt*mt*mt*p0[1] + 3*mt*mt*t*c1[1] + 3*mt*t*t*c2[1] + t*t*t*p1[1];
            pts.push([x, y]);
          }}
          return pts;
        }}

        function outline() {{
          if (points.length < 2) return points.map(p => [p.x_pct, p.y_pct]);
          const anchors = points.map(p => [p.x_pct, p.y_pct]);
          const handles = points.map(p => [p.handle_dx_pct || 0, p.handle_dy_pct || 0]);
          const n = anchors.length;
          const segCount = closed ? n : n - 1;
          let out = [anchors[0]];
          for (let i = 0; i < segCount; i++) {{
            const p0 = anchors[i], p1 = anchors[(i + 1) % n];
            const h0 = handles[i], h1 = handles[(i + 1) % n];
            const c1 = [p0[0] + h0[0], p0[1] + h0[1]];
            const c2 = [p1[0] - h1[0], p1[1] - h1[1]];
            out = out.concat(cubicPoints(p0, c1, c2, p1, 20).slice(1));
          }}
          return out;
        }}

        function redraw() {{
          const out = outline();
          const d = out.length ? ('M ' + out.map(p => p[0].toFixed(2) + ' ' + p[1].toFixed(2)).join(' L ')) + (closed ? ' Z' : '') : '';
          pathEl.setAttribute('d', d);
          pathEl.setAttribute('fill', closed ? 'rgba(96,165,250,0.25)' : 'none');

          anchorLayer.innerHTML = '';
          handleLayer.innerHTML = '';
          handleLines.innerHTML = '';
          points.forEach((p, i) => {{
            const anchor = document.createElement('div');
            anchor.className = 'pen-anchor';
            anchor.style.left = p.x_pct + '%';
            anchor.style.top = p.y_pct + '%';
            anchor.dataset.idx = i;
            anchorLayer.appendChild(anchor);

            const hx = p.x_pct + (p.handle_dx_pct || 0);
            const hy = p.y_pct + (p.handle_dy_pct || 0);
            const handle = document.createElement('div');
            handle.className = 'pen-handle';
            handle.style.left = hx + '%';
            handle.style.top = hy + '%';
            handle.dataset.idx = i;
            handleLayer.appendChild(handle);

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', p.x_pct); line.setAttribute('y1', p.y_pct);
            line.setAttribute('x2', hx); line.setAttribute('y2', hy);
            handleLines.appendChild(line);
          }});
        }}
        redraw();

        function commit() {{
          try {{
            const payload = JSON.stringify(points);
            const container = window.parent.document.querySelector('{hidden_selector}');
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
          }} catch (err) {{}}
        }}

        anchorLayer.addEventListener('pointerdown', (e) => {{
          const el = e.target.closest('.pen-anchor');
          if (!el) return;
          dragMode = {{type: 'anchor', idx: Number(el.dataset.idx)}};
          wrap.setPointerCapture(e.pointerId);
          e.preventDefault();
          e.stopPropagation();
        }});
        handleLayer.addEventListener('pointerdown', (e) => {{
          const el = e.target.closest('.pen-handle');
          if (!el) return;
          dragMode = {{type: 'handle', idx: Number(el.dataset.idx)}};
          wrap.setPointerCapture(e.pointerId);
          e.preventDefault();
          e.stopPropagation();
        }});
        wrap.addEventListener('pointermove', (e) => {{
          if (!dragMode) return;
          const [fx, fy] = fractionalPos(e);
          const p = points[dragMode.idx];
          if (dragMode.type === 'anchor') {{
            p.x_pct = fx; p.y_pct = fy;
          }} else {{
            p.handle_dx_pct = fx - p.x_pct;
            p.handle_dy_pct = fy - p.y_pct;
          }}
          redraw();
        }});
        function endDrag() {{
          if (!dragMode) return;
          dragMode = null;
          commit();
        }}
        wrap.addEventListener('pointerup', endDrag);
        wrap.addEventListener('pointerleave', endDrag);

        anchorLayer.addEventListener('dblclick', (e) => {{
          const el = e.target.closest('.pen-anchor');
          if (!el) return;
          points.splice(Number(el.dataset.idx), 1);
          redraw();
          commit();
        }});

        // Attached to the background image itself, not `wrap` -- see the docstring above for why.
        bg.addEventListener('click', (e) => {{
          const [fx, fy] = fractionalPos(e);
          points.push({{x_pct: fx, y_pct: fy, handle_dx_pct: 0, handle_dy_pct: 0}});
          redraw();
          commit();
        }});
      }})();
    </script>
    """, display_h + 60
