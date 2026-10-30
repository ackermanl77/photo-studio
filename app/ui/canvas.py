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
import re
import uuid

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from app.color_profiles import is_srgb_name, profile_name
from app.professional_color import clipping_warning_overlay
from app.professional_tools import GRID_PRESETS, compute_histogram, compute_waveform, draw_guides_and_grid, draw_viewport_box, preview_image, zoom_preview, zoom_viewport
from app.refine_preview import POLL_SECONDS as REFINE_POLL_SECONDS
from app.ui.processing import refine_status_view
from app.ui.tool_search import jump_to
from app.ui.state import (
    CANVAS_MODE_KEY, CANVAS_PHOTO, RECIPE_DEFAULTS, build_on_canvas, build_recipe, consume_new_commit, request_rerun,
    run_fragment, take_canvas_editors,
)
from app.ui_editors import _before_after_html, _pan_viewer_html, _wand_picker_html
from app.i18n import t, t_format, t_option


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
        t("History"), icon=":material/history:", key="history-popover", use_container_width=True,
        help=t("Reset, History and Named Snapshots -- jump to any earlier step, or bookmark the current look under a name"),
    ):
        # The same recipe fields Undo/Redo cover; layers and the opened photo stay. In here rather
        # than on the header row (UI13): it's rare, and one stray click wipes every adjustment.
        if st.button(t("↺ Reset all adjustments"), key="hist-reset-btn", help=t("Reset all adjustments to defaults (Undo brings them back)"), use_container_width=True):
            st.session_state["_pending_restore"] = {**RECIPE_DEFAULTS, "mask_layers": [], "removal_mask_layers": [], "curve_extra_points": [], "liquify_strokes": [], "plugin_filters": []}
            request_rerun()
        st.divider()
        st.markdown(t("**Named Snapshots**"))
        st.caption(
            t("Bookmark the current look under a name and jump back to it any time -- unlike "
            "Undo/Redo, restoring a snapshot never discards it (or any other saved snapshot).")
        )
        snap_new_name = st.text_input(
            t("Snapshot name"), key="snap-name-in", label_visibility="collapsed", placeholder=t("Snapshot name"),
            help=t("Name for the bookmark you're about to save."),
        )
        if st.button(t("📌 Save Snapshot"), key="snap-save-btn", use_container_width=True):
            if snap_new_name.strip():
                history.save_snapshot(snap_new_name.strip(), build_recipe())
                st.toast(t("Snapshot '{0}' saved").format(t(snap_new_name.strip())), icon="📌")
                request_rerun()
            else:
                st.toast(t("Type a name before saving a snapshot"), icon="⚠️")
        if not history.snapshot_names:
            st.caption(t("No snapshots saved yet."))
        # Keyed by the snapshot's own (stable) name; `break` after a mutation like every list.
        for snap_name in history.snapshot_names:
            snap_row1, snap_row2, snap_row3 = st.columns([2, 1, 0.6])
            snap_row1.caption(t(snap_name))
            if snap_row2.button(t("↩️ Restore"), key=f"snap-restore-{snap_name}", use_container_width=True, help=t("Restore snapshot '{0}'").format(t(snap_name))):
                st.session_state["_pending_restore"] = history.restore_snapshot(snap_name) or {}
                st.toast(t("Restored snapshot '{0}'").format(t(snap_name)), icon="↩️")
                request_rerun()
                break
            if snap_row3.button(t("🗑️"), key=f"snap-delete-{snap_name}", use_container_width=True, help=t("Delete snapshot '{0}'").format(t(snap_name))):
                history.delete_snapshot(snap_name)
                request_rerun()
                break
        _history_list(history)


# Steps shown in the History list (newest first); older ones stay reachable with Undo.
HISTORY_LIST_STEPS = 30


# Every keyboard shortcut, for the "⌨" list (Photoshop's Keyboard Shortcuts). Keep in sync with the
# shortcut= buttons, app/ui/static/library_shortcuts.html and the editors' key handlers.
SHORTCUTS = (
    ("Ctrl+Z", "Undo"), ("Ctrl+Shift+Z", "Redo"), ("Ctrl+E", "Save (export) the photo"),
    ("Z", "1:1 / Fit"), ("B", "Before / after"), ("G", "Composition grid"),
    ("[ / ]", "Brush size (Paint, Repair brush)"), ("Ctrl+Z", "Undo the last stroke (Paint, Repair brush)"),
    ("Arrows", "Nudge the crop box, a Pen point, the Liquify brush, the last radial or gradient (Shift: x10)"),
    ("Space", "Liquify: apply the brush where it is"), ("Delete", "Pen: remove the selected point"),
    ("Ctrl+K or /", "Find a tool"),
    ("1-5 / 0", "Library: rate / clear the rating"), ("6-9", "Library: color label"), ("X", "Library: reject"),
    ("← / →", "Library: previous / next photo"),
)


def _shortcuts_popover() -> None:
    # "⌨" plus VS16 (U+FE0F): the emoji glyph -- the bare character has none in the UI font.
    with st.popover("⌨️", help=t("Keyboard shortcuts"), use_container_width=True):
        st.markdown(t("**Keyboard shortcuts**"))
        rows = [f"| `{keys}` | {t(action)} |" for keys, action in SHORTCUTS]
        st.markdown("\n".join([f"| {t('Keys')} | {t('Action')} |", "|---|---|", *rows]))
        st.caption(t("Single-letter keys work when you aren't typing in a field."))


def _history_list(history) -> None:
    """The Undo steps as a list (F13): click one to jump there, like pressing Undo/Redo until it."""
    st.divider()
    st.markdown(t("**History**"))
    steps = history.steps()
    if len(steps) <= 1:
        st.caption(t("No edits yet."))
        return
    first_shown = max(0, len(steps) - HISTORY_LIST_STEPS)
    for index in range(len(steps) - 1, first_shown - 1, -1):
        current = index == history.cursor
        label = f"{'▶ ' if current else ''}{index}. {steps[index]}"
        # Keyed by step number: the list is rebuilt from history each run, and a jump never
        # renumbers steps (only a new edit after Undo drops the redo branch).
        if st.button(t(label), key=f"hist-step-{index}", disabled=current, use_container_width=True, type="tertiary"):
            st.session_state["_pending_restore"] = history.jump(index) or {}
            request_rerun()
            break
    if first_shown:
        st.caption(t("{0} older step(s) -- reachable with Undo.").format(t(first_shown)))


def canvas_header(ctx, editors: list) -> str:
    """One row above the photo, like Photoshop's options bar (UI13): the file, the "On the photo"
    tool choice (when the open tab has photo editors), Undo/Redo and History, then the view --
    zoom, 1:1, before/after, grid and a View popover with Split, Original and Guides & Grid.
    Returns the "On the photo" choice. Short labels with the full name and key in the tooltip;
    the buttons' native key hints are hidden here (theme.css) -- they printed over the labels.
    Keyed for the stylesheet: on a phone its columns wrap into a compact grid, not one per line."""
    history = ctx.history
    # Icon buttons: the word stays as the button's text for screen readers, hidden on screen
    # (theme.css), with the name and key in the tooltip.
    widths = [1.5] + ([0.95 * (len(editors) + 1)] if editors else []) + [0.42, 0.42, 0.62, 0.95, 0.48, 0.42, 0.42, 0.78, 0.55]
    with st.container(key="canvas-controls"):
        columns = iter(st.columns(widths, gap="small", vertical_alignment="center"))
        with next(columns):
            # html.escape(): the file name is user-controlled and lands in an attribute and in markup.
            live_view_label = html.escape(f"{ctx.upload_name} ({ctx.output_size[0]}×{ctx.output_size[1]})")
            st.markdown(
                t("<div class=\"panel-header-title canvas-live-view\" title=\"{0}\">🖼️ Live View: <span>{1}</span></div>").format(t(live_view_label), t(live_view_label)),
                unsafe_allow_html=True,
            )
        choice = CANVAS_PHOTO
        if editors:
            with next(columns):
                choice = canvas_editor_choice(editors)
        with next(columns):
            if st.button(t("Undo"), icon=":material/undo:", key="hist-undo-btn", disabled=not history.can_undo, help=t("Undo (Ctrl+Z)"), use_container_width=True, shortcut="Ctrl+Z"):
                st.session_state["_pending_restore"] = history.undo() or {}
                request_rerun()
        with next(columns):
            if st.button(t("Redo"), icon=":material/redo:", key="hist-redo-btn", disabled=not history.can_redo, help=t("Redo (Ctrl+Shift+Z)"), use_container_width=True, shortcut="Ctrl+Shift+Z"):
                st.session_state["_pending_restore"] = history.redo() or {}
                request_rerun()
        with next(columns):
            _snapshots_popover(history)

        # Keyboard shortcuts (native, F11): Z = 1:1 / Fit, B = before/after, G = grid. Ctrl+Z /
        # Ctrl+Shift+Z are on Undo/Redo, Ctrl+E on the Save button.
        with next(columns):
            # Fit shows the whole photo; 100 % is a true 1:1 view of its pixels (full resolution even
            # for a proxy-previewed photo); 200 %/400 % enlarge those pixels.
            st.selectbox(
                t("Zoom View"), [0.0, 1.0, 2.0, 4.0], key="cv-zoom", label_visibility="collapsed",
                format_func=t_format(lambda z: "Fit" if z < 1 else f"{int(z * 100)}%"),
                help=t("Zoom: Fit shows the whole photo, 100% is a true 1:1 view of its pixels"),
            )
        next(columns).button(t("1:1"), key="kb-zoom-btn", on_click=_toggle_one_to_one, shortcut="Z", use_container_width=True, help=t("Toggle a true 1:1 view (Z)"))
        next(columns).button(t("Before / After"), icon=":material/compare:", key="kb-orig-btn", on_click=_toggle_original, shortcut="B", use_container_width=True, help=t("Show the original (B)"))
        next(columns).button(t("Grid"), icon=":material/grid_on:", key="kb-grid-btn", on_click=_toggle_grid, shortcut="G", use_container_width=True, help=t("Toggle the composition grid (G)"))
        with next(columns):
            _view_popover()
        with next(columns):
            _shortcuts_popover()

    # Refine-on-pause status (P1): a fragment that polls only while a full-resolution render is
    # pending for a proxy preview, so a settled photo costs nothing.
    st.fragment(refine_status_view, run_every=REFINE_POLL_SECONDS if ctx.refine_tracker.pending() else None)()
    return choice


def _view_popover() -> None:
    """View options (UI13): Split View, Original, and Guides & Grid -- marked "●" while any is on.
    Keyed, so the popover stays open when that mark appears or goes."""
    on = (st.session_state.get("cv-split", False) or st.session_state.get("cv-orig", False)
          or st.session_state.get("cv-mirror", False) or st.session_state.get("grid-type", "None") != "None"
          or bool(st.session_state.get("guides")))
    with st.popover(t("View" + (" ●" if on else "")), key="view-popover", use_container_width=True,
                    help=t("Split view, the original, guides and grid")):
        compare_mode = st.checkbox(t("⇄ Split View"), value=False, key="cv-split", help=t("Drag on the photo to compare before/after; ↕ in the view splits it top and bottom"))
        st.checkbox(t("👁️ Original"), value=False, key="cv-orig", disabled=compare_mode)
        st.divider()
        guides_panel()


def _show_clipped_colors() -> None:
    jump_to("🎯 Soft Proof: Gamut Warning")
    st.session_state["gamut-warning-toggle"] = True


def profile_notice(ctx) -> None:
    """Photoshop stops to ask what to do with a photo whose embedded profile isn't its working space
    ("Embedded Profile Mismatch"). Here the conversion is automatic (run_pipeline()'s ICC step), so
    it's said above the photo instead, without a dialog: which profile, that editing happens in
    sRGB, and a button to Soft Proof, which shows what that clips."""
    name = profile_name(ctx.source.info.get("icc_profile"))
    if name is None or is_srgb_name(name):
        return
    with st.container(key="profile-notice"):
        text_col, button_col = st.columns([5, 1.5], vertical_alignment="center")
        if not name:
            text_col.caption(t("🎨 This photo's color profile couldn't be read, so its colors are shown as sRGB."))
            return
        # The name comes from the file: escaped, so it can't add Markdown of its own.
        shown_name = re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~:])", r"\\\1", name)
        text_col.caption(t("🎨 Color profile **{0}**: edited in sRGB, so colors outside sRGB are clipped. Export as Display P3 to keep more of them.").format(shown_name))
        button_col.button(t("Show clipped colors"), key="profile-proof-btn", on_click=_show_clipped_colors, use_container_width=True,
                          help=t("Opens Soft Proof (🎨 Curves & Color) with the out-of-gamut pixels shown."))


def guides_panel() -> None:
    """Guides & Grid: a view-only composition aid (never exported). Outside the canvas fragment
    because the free-form Crop editor snaps to these lines too."""
    if "guides" not in st.session_state:
        st.session_state.guides = []
    guides = st.session_state.guides
    st.markdown(t("**🧭 Guides & Grid**"))
    st.caption(t("View-only -- never affects the exported image. The free-form Crop tool's corner handles snap to these lines."))
    # Photoshop's View > Flip Horizontal: a fresh look at a composition's balance.
    st.checkbox(t("⇋ Flip the view (mirror)"), key="cv-mirror",
                help=t("Shows the photo mirrored left to right, to judge its composition with fresh eyes. View-only: the edit and the export aren't flipped. Not applied to the zoomed (100 %+) view."))
    st.selectbox(t("Grid overlay"), list(GRID_PRESETS), key="grid-type", format_func=t_option)
    for guide in guides:
        guide.setdefault("id", uuid.uuid4().hex)
    add_col1, add_col2, add_col3 = st.columns([1, 1.4, 0.8])
    with add_col1:
        new_guide_orientation = st.selectbox(t("Orientation"), ["Horizontal", "Vertical"], key="guide-new-orientation", label_visibility="collapsed", format_func=t_option)
    with add_col2:
        new_guide_position = st.slider(t("Position %"), 0.0, 100.0, 50.0, key="guide-new-position", label_visibility="collapsed")
    with add_col3:
        if st.button(t("➕ Add"), key="guide-add-btn"):
            guides.append({"id": uuid.uuid4().hex, "orientation": new_guide_orientation.lower(), "position_pct": new_guide_position})
            request_rerun()
    for idx, guide in enumerate(guides):
        guide_col1, guide_col2 = st.columns([3, 1])
        with guide_col1:
            st.caption(t("{0} guide at {1:.0f}%").format(t(guide['orientation'].capitalize()), t(guide['position_pct'])))
        with guide_col2:
            if st.button(t("🗑️ Remove"), key=f"guide-rm-{guide['id']}"):
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
    # Flip view (Guides & Grid): mirrored last, after the guides, so they stay on the same content.
    # Not at 100 %+, where dragging pans the view and a mirror would turn the drag around.
    mirror = ImageOps.mirror if st.session_state.get("cv-mirror", False) and zoom_level < 1.0 else (lambda image: image)
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
            pending_nav = st.text_area(t("Navigator click"), key="cv-nav-wand-data", label_visibility="collapsed")
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
            pan_x = st.slider(t("Pan Horizontal"), 0.0, 1.0, 0.5, 0.0001, format="%.3f", key="cv-panx")
        with pan_c2:
            pan_y = st.slider(t("Pan Vertical"), 0.0, 1.0, 0.5, 0.0001, format="%.3f", key="cv-pany")

    if compare_mode:
        # Split View: a draggable before/after slider sharing the zoom/pan of the normal view.
        before_preview = zoom_preview(oriented_source, zoom_level, (pan_x, pan_y))
        after_preview = zoom_preview(edited_image, zoom_level, (pan_x, pan_y))
        if before_preview.size != after_preview.size:
            before_preview = before_preview.resize(after_preview.size, Image.Resampling.LANCZOS)
        compare_html, compare_height = _before_after_html(mirror(before_preview), mirror(after_preview))
        components.html(compare_html, height=compare_height + 10)
        return
    st.markdown(t('<div class="canvas-container">'), unsafe_allow_html=True)
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
        st.caption(t("{0} -- {1}%, drag to pan").format(t(canvas_caption), t(int(zoom_level * 100))))
    else:
        # Keyed so theme.css can fit the photo to the window's height too (a tall photo used to
        # run past the bottom of the screen once the canvas had the page's full width).
        if mirror is ImageOps.mirror:
            canvas_display, canvas_caption = mirror(canvas_display), t("{0} (flipped view)").format(t(canvas_caption))
        with st.container(key="canvas-stage"):
            st.image(canvas_display, caption=canvas_caption, use_container_width=True)
    st.markdown(t('</div>'), unsafe_allow_html=True)


def scopes_panel(edited_image: Image.Image) -> None:
    """Histogram, waveform and clipping warning of the edited result (not of whichever view is
    toggled) -- a fragment: toggling a scope reruns only this. Computed on a display-sized copy."""
    scope_image = preview_image(edited_image, (1024, 1024))
    histogram_bins = 64
    histogram = compute_histogram(scope_image, bins=histogram_bins)
    # Plot against each bin's pixel value (0-255), not the bin index.
    histogram["Pixel Value"] = [round(255 * (i + 0.5) / histogram_bins) for i in range(histogram_bins)]
    st.line_chart(histogram, x="Pixel Value", y=["Red", "Green", "Blue"], height=90, color=["#f87171", "#4ade80", "#60a5fa"])

    st.caption(t("〰️ **Waveform monitor**: an extra scope for checking exposure across the frame, next to the histogram above."))
    waveform_toggle_col, waveform_mode_col = st.columns([1, 1])
    with waveform_toggle_col:
        show_waveform = st.checkbox(
            t("〰️ Show waveform monitor"), key="waveform-toggle",
            help=t("Horizontal position vs. luminance/RGB intensity, like a video editor's waveform scope."),
        )
    with waveform_mode_col:
        # Always rendered (disabled when off) so the channel choice survives toggling the scope.
        waveform_mode = st.radio(
            t("Waveform channels"), ["Luma", "RGB"], key="waveform-mode", horizontal=True,
            label_visibility="collapsed", disabled=not show_waveform, format_func=t_option,
        )
    if show_waveform:
        st.image(compute_waveform(scope_image, mode=waveform_mode), caption=f"Waveform ({waveform_mode})", use_container_width=True)
    show_clipping = st.checkbox(t("⚠️ Show clipping"), key="clip-warning-toggle", help=t("Highlights blocked shadows (blue) and blown highlights (red)."))
    # Always rendered (a skipped widget loses its value), disabled while the warning is off.
    margin = st.slider(
        t("Clipping margin"), 0, 20, 0, key="clip-warning-margin", disabled=not show_clipping,
        help=t("Also flag tones within this many levels of pure black or white (0 = only fully clipped pixels)."),
    )
    if show_clipping:
        st.image(
            clipping_warning_overlay(scope_image, shadow_threshold=margin, highlight_threshold=255 - margin),
            caption=t("Blue = shadows clipped to black, Red = highlights clipped to white"),
            use_container_width=True,
        )


# The box an editor's background fits in when it is drawn over the photo: the page's width at
# 1440 px with the sidebar open, and short enough to leave room for the editor's own buttons.
CANVAS_EDITOR_BOX = (1000, 600)


# The status line over a tool that works on the photo (Photoshop's per-tool status-bar hint): what to
# do, and where its settings are -- the sidebar may be collapsed. Keyed by the editor's label.
CANVAS_HINTS = {
    "✂ Crop": ("Drag a corner to resize the box, drag inside it to move it", "⛶ Geometry › Crop / Aspect Ratio"),
    "🌀 Liquify": ("Drag on the photo to warp it", "⛶ Geometry › 🌀 Liquify"),
    "🖌 Paint": ("Paint what to select, then click Add to selections", "◉ Local › Masks"),
    "🩹 Repair brush": ("Paint over what to remove, then click Remove", "◉ Local › 🩹 Repair"),
    "🎨 Edit Painting": ("Paint or erase, then click Update Painting", "◉ Local"),
    "🪄 Magic Wand": ("Click a point to select similar colors around it", "◉ Local"),
    "🎯 White Balance": ("Click a spot that should be neutral gray or white", "☼ Basic › Color"),
    "✒ Pen": ("Click to add points; drag a point to move it, its handle to curve the path", "◈ Layers"),
    "🟨 Removal preview": ("Yellow = what will be removed", "◉ Local › 🩹 Repair"),
    "🟥 Mask preview": ("Highlighted = selected by the combined mask", "◉ Local › Masks"),
}


def canvas_hint(label: str) -> None:
    hint = CANVAS_HINTS.get(re.sub(r" \d+$", "", label))  # a second editor of a kind is "<label> 2"
    if hint:
        st.caption(t("{0}. Its settings: {1}.").format(t(hint[0]), t(hint[1])))


def _back_to_photo() -> None:
    st.session_state[CANVAS_MODE_KEY] = CANVAS_PHOTO


def canvas_action_bar(editor) -> None:
    """The bar over a tool that works on the photo, like Photoshop's contextual task bar: what to do
    (the hint), the tool's one-click actions, and "✓ Done" back to the photo."""
    with st.container(key="canvas-action-bar"):
        columns = st.columns([4.2] + [1.25] * len(editor.actions) + [1], vertical_alignment="center")
        with columns[0]:
            canvas_hint(editor.label)
        for column, action in zip(columns[1:], editor.actions):
            column.button(t(action.label), key=action.key, on_click=action.on_click, args=action.args, disabled=action.disabled,
                          help=t(action.help) if action.help else None, use_container_width=True)
        columns[-1].button(t("✓ Done"), key="canvas-done-btn", on_click=_back_to_photo, type="primary", use_container_width=True,
                           help=t("Back to the photo (🖼 Photo)."))


def canvas_editor_choice(editors: list) -> str:
    """"On the photo": the photo itself, or one of the open tab's photo editors (crop, liquify,
    paint, magic wand, pen...), drawn over the photo instead of at sidebar width -- Photoshop's
    model, where a tool works on the document. Only rendered when the open tab offers an editor;
    switching tabs (other editors) falls back to the photo."""
    if not editors:
        return CANVAS_PHOTO
    options = [CANVAS_PHOTO] + [editor.label for editor in editors]
    if st.session_state.get(CANVAS_MODE_KEY) not in options:
        st.session_state[CANVAS_MODE_KEY] = CANVAS_PHOTO
    choice = st.segmented_control(
        t("On the photo"), options, key=CANVAS_MODE_KEY, format_func=t_option, label_visibility="collapsed",
        help=t("Work on the photo with one of this tab's tools, or just view it."),
    )
    return choice or CANVAS_PHOTO


def render_canvas_column(ctx, oriented_source: Image.Image) -> None:
    editors = take_canvas_editors()
    choice = canvas_header(ctx, editors)
    profile_notice(ctx)
    if choice == CANVAS_PHOTO:
        run_fragment(canvas_view, ctx.edited_image, oriented_source)
    else:
        editor = next(editor for editor in editors if editor.label == choice)
        canvas_action_bar(editor)
        markup, height = build_on_canvas(editor.build, CANVAS_EDITOR_BOX)
        components.html(markup, height=height)
    run_fragment(scopes_panel, ctx.edited_image)
