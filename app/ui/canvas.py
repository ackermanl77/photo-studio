"""The canvas column: header (file label, Undo/Redo/Reset, Named Snapshots, zoom, compare toggles),
keyboard-shortcut buttons, refine-on-pause status, Guides & Grid, the canvas itself and the scopes.

The canvas view (navigator, pan, the image) and the scopes (histogram, waveform, clipping) run as
st.fragment()s (F1): panning or toggling a scope reruns only that fragment -- not the tool panels,
not the pipeline. Anything that changes what is rendered (zoom, which changes the preview
resolution; Original/Split; the grid, which the Crop editor also reads) stays outside them, so it
triggers a normal full run.
"""
from __future__ import annotations

import html
import json
import uuid

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from app.professional_color import clipping_warning_overlay
from app.professional_tools import GRID_PRESETS, compute_histogram, compute_waveform, draw_guides_and_grid, draw_viewport_box, preview_image, zoom_preview, zoom_viewport
from app.refine_preview import POLL_SECONDS as REFINE_POLL_SECONDS
from app.ui.processing import refine_status_view
from app.ui.state import RECIPE_DEFAULTS, build_recipe, consume_new_commit, request_rerun, run_fragment
from app.ui_editors import _before_after_html, _pan_viewer_html, _wand_picker_html


def _toggle_one_to_one() -> None:
    """Shortcut Z: Fit <-> a true 1:1 view (a callback runs before the script, so it may set the
    zoom widget's value)."""
    st.session_state["cv-zoom"] = 0.0 if float(st.session_state.get("cv-zoom", 0.0)) >= 1.0 else 1.0


def _toggle_original() -> None:
    """Shortcut B: before/after."""
    st.session_state["cv-orig"] = not st.session_state.get("cv-orig", False)


def _toggle_grid() -> None:
    """Shortcut G: grid on/off (remembering the last grid chosen)."""
    current = st.session_state.get("grid-type", "None")
    if current != "None":
        st.session_state["_last-grid-type"] = current
        st.session_state["grid-type"] = "None"
    else:
        st.session_state["grid-type"] = st.session_state.get("_last-grid-type", "Rule of Thirds")


def _snapshots_popover(history) -> None:
    with st.popover(
        "📌", use_container_width=True,
        help="History and Named Snapshots -- jump to any earlier step, or bookmark the current look under a name",
    ):
        st.markdown("**Named Snapshots**")
        st.caption(
            "Bookmark the current look under a name and jump back to it any time -- unlike "
            "Undo/Redo, restoring a snapshot never discards it (or any other saved snapshot)."
        )
        snap_new_name = st.text_input(
            "Snapshot name", key="snap-name-in", label_visibility="collapsed", placeholder="Snapshot name",
            help="Name for the bookmark you're about to save.",
        )
        if st.button("📌 Save Snapshot", key="snap-save-btn", use_container_width=True):
            if snap_new_name.strip():
                history.save_snapshot(snap_new_name.strip(), build_recipe())
                st.toast(f"Snapshot '{snap_new_name.strip()}' saved", icon="📌")
                request_rerun()
            else:
                st.toast("Type a name before saving a snapshot", icon="⚠️")
        if not history.snapshot_names:
            st.caption("No snapshots saved yet.")
        # Keyed by the snapshot's own (stable) name; `break` after a mutation like every list.
        for snap_name in history.snapshot_names:
            snap_row1, snap_row2, snap_row3 = st.columns([2, 1, 0.6])
            snap_row1.caption(snap_name)
            if snap_row2.button("↩️ Restore", key=f"snap-restore-{snap_name}", use_container_width=True, help=f"Restore snapshot '{snap_name}'"):
                st.session_state["_pending_restore"] = history.restore_snapshot(snap_name) or {}
                st.toast(f"Restored snapshot '{snap_name}'", icon="↩️")
                request_rerun()
                break
            if snap_row3.button("🗑️", key=f"snap-delete-{snap_name}", use_container_width=True, help=f"Delete snapshot '{snap_name}'"):
                history.delete_snapshot(snap_name)
                request_rerun()
                break
        _history_list(history)


# Steps shown in the History list (newest first); older ones stay reachable with Undo.
HISTORY_LIST_STEPS = 30


def _history_list(history) -> None:
    """The Undo steps as a list (F13): click one to jump there, like pressing Undo/Redo until it."""
    st.divider()
    st.markdown("**History**")
    steps = history.steps()
    if len(steps) <= 1:
        st.caption("No edits yet.")
        return
    first_shown = max(0, len(steps) - HISTORY_LIST_STEPS)
    for index in range(len(steps) - 1, first_shown - 1, -1):
        current = index == history.cursor
        label = f"{'▶ ' if current else ''}{index}. {steps[index]}"
        # Keyed by step number: the list is rebuilt from history each run, and a jump never
        # renumbers steps (only a new edit after Undo drops the redo branch).
        if st.button(label, key=f"hist-step-{index}", disabled=current, use_container_width=True, type="tertiary"):
            st.session_state["_pending_restore"] = history.jump(index) or {}
            request_rerun()
            break
    if first_shown:
        st.caption(f"{first_shown} older step(s) -- reachable with Undo.")


def canvas_header(ctx) -> None:
    history = ctx.history
    top_c1, top_undo, top_redo, top_reset, top_snap, top_c2, top_c3 = st.columns([0.85, 0.18, 0.18, 0.18, 0.22, 0.65, 0.95], gap="small")
    with top_c1:
        # html.escape(): the file name is user-controlled and lands in an attribute and in markup.
        live_view_label = html.escape(f"{ctx.upload_name} ({ctx.output_size[0]}×{ctx.output_size[1]})")
        st.markdown(
            f'<div class="panel-header-title canvas-live-view" title="{live_view_label}">🖼️ Live View: <span>{live_view_label}</span></div>',
            unsafe_allow_html=True,
        )
    with top_undo:
        if st.button("↶", key="hist-undo-btn", disabled=not history.can_undo, help="Undo (Ctrl+Z)", use_container_width=True, shortcut="Ctrl+Z"):
            st.session_state["_pending_restore"] = history.undo() or {}
            request_rerun()
    with top_redo:
        if st.button("↷", key="hist-redo-btn", disabled=not history.can_redo, help="Redo (Ctrl+Shift+Z)", use_container_width=True, shortcut="Ctrl+Shift+Z"):
            st.session_state["_pending_restore"] = history.redo() or {}
            request_rerun()
    with top_reset:
        # The same recipe fields Undo/Redo cover; layers and the opened photo stay.
        if st.button("↺", key="hist-reset-btn", help="Reset all adjustments to defaults", use_container_width=True):
            st.session_state["_pending_restore"] = {**RECIPE_DEFAULTS, "mask_layers": [], "removal_mask_layers": [], "curve_extra_points": []}
            request_rerun()
    with top_snap:
        _snapshots_popover(history)
    with top_c2:
        # Fit shows the whole photo; 100 % is a true 1:1 view of its pixels (full resolution even
        # for a proxy-previewed photo); 200 %/400 % enlarge those pixels.
        st.selectbox(
            "Zoom View", [0.0, 1.0, 2.0, 4.0], key="cv-zoom", label_visibility="collapsed",
            format_func=lambda z: "🔍 Fit" if z < 1 else ("🔍 100% (1:1)" if z == 1 else f"🔍 {int(z * 100)}%"),
        )
    with top_c3:
        compare_mode = st.checkbox("⇄ Split View", value=False, key="cv-split", help="Drag to compare before/after")
        st.checkbox("👁️ Original", value=False, key="cv-orig", disabled=compare_mode)

    # Keyboard shortcuts (native, F11): Z = 1:1 / Fit, B = before/after, G = grid. Ctrl+Z /
    # Ctrl+Shift+Z are on Undo/Redo, Ctrl+E on the Save button.
    key_c1, key_c2, key_c3 = st.columns(3)
    key_c1.button("🔍 1:1 / Fit", key="kb-zoom-btn", on_click=_toggle_one_to_one, shortcut="Z", use_container_width=True, help="Toggle a true 1:1 view (Z)")
    key_c2.button("👁 Before / After", key="kb-orig-btn", on_click=_toggle_original, shortcut="B", use_container_width=True, help="Show the original (B)")
    key_c3.button("# Grid", key="kb-grid-btn", on_click=_toggle_grid, shortcut="G", use_container_width=True, help="Toggle the composition grid (G)")

    # Refine-on-pause status (P1): a fragment that polls only while a full-resolution render is
    # pending for a proxy preview, so a settled photo costs nothing.
    st.fragment(refine_status_view, run_every=REFINE_POLL_SECONDS if ctx.refine_tracker.pending() else None)()


def guides_panel() -> None:
    """Guides & Grid: a view-only composition aid (never exported). Outside the canvas fragment
    because the free-form Crop editor snaps to these lines too."""
    if "guides" not in st.session_state:
        st.session_state.guides = []
    guides = st.session_state.guides
    grid_on = st.session_state.get("grid-type", "None") != "None" or bool(guides)
    with st.expander("🧭 Guides & Grid" + (" ●" if grid_on else ""), expanded=False, key="sec-guides"):
        st.caption("View-only -- never affects the exported image. The free-form Crop tool's corner handles snap to these lines.")
        st.selectbox("Grid overlay", list(GRID_PRESETS), key="grid-type")
        for guide in guides:
            guide.setdefault("id", uuid.uuid4().hex)
        add_col1, add_col2, add_col3 = st.columns([1, 1.4, 0.8])
        with add_col1:
            new_guide_orientation = st.selectbox("Orientation", ["Horizontal", "Vertical"], key="guide-new-orientation", label_visibility="collapsed")
        with add_col2:
            new_guide_position = st.slider("Position %", 0.0, 100.0, 50.0, key="guide-new-position", label_visibility="collapsed")
        with add_col3:
            if st.button("➕ Add", key="guide-add-btn"):
                guides.append({"id": uuid.uuid4().hex, "orientation": new_guide_orientation.lower(), "position_pct": new_guide_position})
                request_rerun()
        for idx, guide in enumerate(guides):
            guide_col1, guide_col2 = st.columns([3, 1])
            with guide_col1:
                st.caption(f"{guide['orientation'].capitalize()} guide at {guide['position_pct']:.0f}%")
            with guide_col2:
                if st.button("🗑️ Remove", key=f"guide-rm-{guide['id']}"):
                    guides.pop(idx)
                    request_rerun()
                    break


def canvas_view(edited_image: Image.Image, oriented_source: Image.Image) -> None:
    """The navigator and pan sliders (when zoomed) and the canvas image -- a fragment: panning
    reruns only this. Zoom, Original/Split and the grid are read from session_state (their widgets
    render earlier in the same full run)."""
    zoom_level = float(st.session_state.get("cv-zoom", 0.0))
    compare_mode = bool(st.session_state.get("cv-split", False))
    show_original = bool(st.session_state.get("cv-orig", False)) and not compare_mode
    grid_type = st.session_state.get("grid-type", "None")
    guides = st.session_state.get("guides", [])
    pan_x = pan_y = 0.5
    if zoom_level >= 1.0:
        # Navigator: click the overview to center the zoomed view there (the hidden text_area
        # commit pattern). Handled before the pan sliders render, so it may set their values.
        nav_view = preview_image(edited_image, (360, 360))
        nav_box = zoom_viewport(edited_image.size, zoom_level, (st.session_state.get("cv-panx", 0.5), st.session_state.get("cv-pany", 0.5)))
        scale_x, scale_y = nav_view.width / edited_image.width, nav_view.height / edited_image.height
        nav_view = draw_viewport_box(nav_view, tuple(int(v * s) for v, s in zip(nav_box, (scale_x, scale_y, scale_x, scale_y))))
        nav_html, nav_height = _wand_picker_html(nav_view, "cv-nav", hint="Click to center the zoomed view there.")
        components.html(nav_html, height=nav_height)
        with st.container(key="cv-nav-wand-hidden"):
            pending_nav = st.text_area("Navigator click", key="cv-nav-wand-data", label_visibility="collapsed")
        if consume_new_commit(pending_nav, "cv-nav-committed"):
            try:
                nav_click = json.loads(pending_nav)
                # 4 decimals: a drag-pan at 400 % moves the view by a few source pixels.
                st.session_state["cv-panx"] = round(min(1.0, max(0.0, float(nav_click["x"]))), 4)
                st.session_state["cv-pany"] = round(min(1.0, max(0.0, float(nav_click["y"]))), 4)
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                pass
        pan_c1, pan_c2 = st.columns(2)
        with pan_c1:
            pan_x = st.slider("Pan Horizontal", 0.0, 1.0, 0.5, 0.0001, format="%.3f", key="cv-panx")
        with pan_c2:
            pan_y = st.slider("Pan Vertical", 0.0, 1.0, 0.5, 0.0001, format="%.3f", key="cv-pany")

    if compare_mode:
        # Split View: a draggable before/after slider sharing the zoom/pan of the normal view.
        before_preview = zoom_preview(oriented_source, zoom_level, (pan_x, pan_y))
        after_preview = zoom_preview(edited_image, zoom_level, (pan_x, pan_y))
        if before_preview.size != after_preview.size:
            before_preview = before_preview.resize(after_preview.size, Image.Resampling.LANCZOS)
        compare_html, compare_height = _before_after_html(before_preview, after_preview)
        components.html(compare_html, height=compare_height + 10)
        return
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    if show_original:
        canvas_display, canvas_caption = preview_image(oriented_source), "Original Untouched"
    else:
        canvas_display, canvas_caption = zoom_preview(edited_image, zoom_level, (pan_x, pan_y)), "Current Edited Image"
    if grid_type != "None" or guides:
        canvas_display = draw_guides_and_grid(canvas_display, grid_type, guides)
    if zoom_level >= 1.0 and not show_original:
        # Drag-to-pan (F3): the zoomed view commits a new center through the navigator's field.
        view_box = zoom_viewport(edited_image.size, zoom_level, (pan_x, pan_y))
        view_center = ((view_box[0] + view_box[2]) / 2 / edited_image.width, (view_box[1] + view_box[3]) / 2 / edited_image.height)
        pan_html, pan_height = _pan_viewer_html(canvas_display, "cv-nav", view_center, zoom_level, edited_image.size)
        components.html(pan_html, height=pan_height)
        st.caption(f"{canvas_caption} -- {int(zoom_level * 100)}%, drag to pan")
    else:
        st.image(canvas_display, caption=canvas_caption, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def scopes_panel(edited_image: Image.Image) -> None:
    """Histogram, waveform and clipping warning of the edited result (not of whichever view is
    toggled) -- a fragment: toggling a scope reruns only this. Computed on a display-sized copy."""
    scope_image = preview_image(edited_image, (1024, 1024))
    histogram_bins = 64
    histogram = compute_histogram(scope_image, bins=histogram_bins)
    # Plot against each bin's pixel value (0-255), not the bin index.
    histogram["Pixel Value"] = [round(255 * (i + 0.5) / histogram_bins) for i in range(histogram_bins)]
    st.line_chart(histogram, x="Pixel Value", y=["Red", "Green", "Blue"], height=90, color=["#f87171", "#4ade80", "#60a5fa"])

    st.caption("〰️ **Waveform monitor**: an extra scope for checking exposure across the frame, next to the histogram above.")
    waveform_toggle_col, waveform_mode_col = st.columns([1, 1])
    with waveform_toggle_col:
        show_waveform = st.checkbox(
            "〰️ Show waveform monitor", key="waveform-toggle",
            help="Horizontal position vs. luminance/RGB intensity, like a video editor's waveform scope.",
        )
    with waveform_mode_col:
        # Always rendered (disabled when off) so the channel choice survives toggling the scope.
        waveform_mode = st.radio(
            "Waveform channels", ["Luma", "RGB"], key="waveform-mode", horizontal=True,
            label_visibility="collapsed", disabled=not show_waveform,
        )
    if show_waveform:
        st.image(compute_waveform(scope_image, mode=waveform_mode), caption=f"Waveform ({waveform_mode})", use_container_width=True)
    if st.checkbox("⚠️ Show clipping", key="clip-warning-toggle", help="Highlights blocked shadows (blue) and blown highlights (red)."):
        st.image(
            clipping_warning_overlay(scope_image),
            caption="Blue = shadows clipped to black, Red = highlights clipped to white",
            use_container_width=True,
        )


def render_canvas_column(ctx, oriented_source: Image.Image) -> None:
    canvas_header(ctx)
    guides_panel()
    run_fragment(canvas_view, ctx.edited_image, oriented_source)
    run_fragment(scopes_panel, ctx.edited_image)
