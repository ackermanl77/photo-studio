"""Geometry tab: Rotate & Straighten, Lens Corrections, Liquify, Crop.

The pipeline applies rotation/flip, Lens, straighten and constrain-crop first and the crop last
(app.pipeline); compute_geometry() builds the same geometry locally so Crop and the Masks/Repair
tools draw against the frame the backend will actually produce.
"""
from __future__ import annotations

import json

import streamlit as st

from app import geometry, lens_lookup, liquify
from app.controls import CONTROLS_BY_KEY, section_keys
from app.lens_corrections import LENS_PROFILES
from app.professional_tools import draw_guides_and_grid, grid_line_fractions, preview_image
from app.ui.processing import lens_corrected_source
from app.ui.state import consume_new_commit, editor_box, request_rerun, section_header, section_reset_button, show_component
from app.ui.widgets import render_control
from app.ui_editors import _crop_editor_html, _liquify_editor_html
from app.i18n import t, t_format

GEOMETRY_KEYS = (*section_keys("geometry"), *(key for key in section_keys("lens") if CONTROLS_BY_KEY[key].option))


def rotate_section(ctx) -> None:
    section_header("Rotate & Straighten", "geometry")
    values = ctx.values
    rot_left_col, rot_right_col, rot_value_col = st.columns([1, 1, 1.2])
    current_rotation = int(st.session_state.get("geo-rotate", 0))
    if rot_left_col.button(t("⟲ Rotate left"), key="geo-rot-left-btn", use_container_width=True, help=t("Rotate 90° counter-clockwise")):
        st.session_state["_pending_restore"] = {"rotate": (current_rotation - 90) % 360}
        request_rerun()
    if rot_right_col.button(t("⟳ Rotate right"), key="geo-rot-right-btn", use_container_width=True, help=t("Rotate 90° clockwise")):
        st.session_state["_pending_restore"] = {"rotate": (current_rotation + 90) % 360}
        request_rerun()
    values["rotate"] = render_control("rotate", rot_value_col, format_func=lambda angle: f"{angle}° clockwise", label_visibility="collapsed")
    flip_h_col, flip_v_col = st.columns(2)
    values["flip_horizontal"] = render_control("flip_horizontal", flip_h_col)
    values["flip_vertical"] = render_control("flip_vertical", flip_v_col)
    straighten_col, auto_col = st.columns([3, 1])
    values["straighten"] = render_control("straighten", straighten_col)
    ctx.auto_level_requested = auto_col.button(t("Auto"), key="geo-auto-btn", use_container_width=True, help=t("Detect the photo's dominant horizontal/vertical lines and level them"))
    values["constrain_crop"] = render_control("constrain_crop")
    section_reset_button("geometry", "Rotate & Straighten")


def lens_section(ctx) -> None:
    section_header("Lens Corrections", "lens")
    values = ctx.values
    available = lens_lookup.lensfun_available()
    st.caption(
        t("📷 **From the photo's lens** looks the camera and lens up in the lensfun database (from the "
        "photo's EXIF) and fits the sliders below to its calibration at this focal length and aperture."
        if available else
        "The lens database (lensfunpy) isn't installed, so only the category starting points below are available.")
    )
    if st.button(t("📷 From the photo's lens"), key="lens-auto-btn", disabled=not available):
        match, message = lens_lookup.find_lens(ctx.source)
        # Kept with the photo it describes, so another photo doesn't show a stale match.
        st.session_state["_lens-auto-message"] = (ctx.source_digest, message)
        if match is not None:
            st.session_state["_pending_restore"] = {**match.sliders, "lens_profile": "None"}
            request_rerun()
    lens_message = st.session_state.get("_lens-auto-message")
    if lens_message and lens_message[0] == ctx.source_digest:
        st.caption(t(lens_message[1]))
    st.caption(t("Or start from a typical lens category, then fine-tune with the sliders below."))
    values["lens_profile"] = render_control("lens_profile")
    # A caption rather than help=: st.button(help=...) adds a second focusable tooltip button.
    st.caption(t("Loads this profile's starting values -- you can still adjust the sliders afterward."))
    if st.button(t("Apply Profile Defaults"), key="lens-apply-profile-btn"):
        profile_values = LENS_PROFILES[values["lens_profile"]]
        st.session_state["_pending_restore"] = {
            "lens_distortion": profile_values["distortion"],
            "lens_vignette_amount": profile_values["vignette_amount"],
            "lens_vignette_midpoint": profile_values["vignette_midpoint"],
            "lens_chromatic_aberration": profile_values["chromatic_aberration"],
        }
        request_rerun()
    for key in ("lens_distortion", "lens_vignette_amount", "lens_vignette_midpoint", "lens_chromatic_aberration"):
        values[key] = render_control(key)
    st.markdown(t("**Perspective (Keystone)**"))
    st.caption(t("Not part of a lens profile -- this depends on the shooting angle, not the lens, so it isn't filled in by 'Apply Profile Defaults' above."))
    for key in ("lens_perspective_v", "lens_perspective_h"):
        values[key] = render_control(key)
    section_reset_button("lens", "Lens")


def compute_geometry(ctx) -> None:
    """The frame Crop/Masks/Repair work against (after rotate_section and lens_section ran)."""
    ctx.geometry_options = {key: ctx.values[key] for key in GEOMETRY_KEYS}
    # Liquify strokes (F23) are drawn on the geometry frame and applied last in the same step.
    strokes = st.session_state.setdefault("liquify_strokes", [])
    if strokes:
        ctx.geometry_options["liquify"] = list(strokes)
    geometry_json = json.dumps(ctx.geometry_options, sort_keys=True)
    ctx.mask_source = lens_corrected_source(ctx.source_digest, ctx.source, geometry_json)
    # Cache key for everything built against mask_source: the photo plus the geometry that warped it.
    ctx.mask_source_key = f"{ctx.source_digest}|{geometry_json}"
    if ctx.auto_level_requested:
        # Measured on the frame before straightening (the angle replaces the current one).
        level_base = geometry.apply_geometry(preview_image(ctx.source, (1024, 1024)), {**ctx.geometry_options, "straighten": 0.0, "constrain_crop": False})
        detected_angle = geometry.estimate_straighten_angle(level_base)
        st.session_state["_pending_restore"] = {"straighten": detected_angle}
        st.toast(t("Straighten set to {0:+.2f}°").format(detected_angle) if detected_angle else t("No dominant lines found to level"), icon="📐")
        request_rerun()


def liquify_section(ctx) -> None:
    """Liquify (F23): a mesh-warp brush. Strokes are parametric (app.liquify) -- kept in
    st.session_state["liquify_strokes"], part of the recipe (Undo/Redo, projects) but never synced
    to other photos -- and drawn on the geometry frame, already showing the strokes so far."""
    strokes = st.session_state.setdefault("liquify_strokes", [])
    section_header("🌀 Liquify", extra_modified=bool(strokes))
    tool_col, size_col, strength_col = st.columns([1.3, 1, 1])
    tool = tool_col.selectbox(t("Tool"), liquify.TOOLS, key="liq-tool", format_func=t_format(lambda name: liquify.TOOL_LABELS[name]))
    size = size_col.slider(t("Brush size (%)"), 1, 50, 10, key="liq-size", help=t("The brush radius, as a percentage of the photo's short edge."))
    strength = strength_col.slider(t("Strength"), 5, 100, 50, key="liq-strength")
    mask_source = ctx.mask_source

    def build_editor():
        return _liquify_editor_html(preview_image(mask_source, editor_box()), "liquify-editor", size / 100.0, liquify.MAX_POINTS)

    show_component(build_editor, canvas_label="🌀 Liquify", slot_id="liquify")
    with st.container(key="liquify-editor-hidden"):
        pending = st.text_area(t("Liquify stroke"), key="liquify-editor-data", label_visibility="collapsed")
    if consume_new_commit(pending, "liquify-editor-committed"):
        try:
            points = json.loads(pending)["points"]
        except (json.JSONDecodeError, TypeError, KeyError):
            points = None
        stroke = liquify.clean_stroke({"tool": tool, "size": size / 100.0, "strength": strength / 100.0, "points": points})
        if stroke is not None and len(strokes) < liquify.MAX_STROKES:
            strokes.append(stroke)
            request_rerun()
    undo_col, clear_col = st.columns(2)
    if undo_col.button(t("↶ Undo last stroke"), key="liq-undo-btn", disabled=not strokes, use_container_width=True):
        strokes.pop()
        request_rerun()
    if clear_col.button(t("✖ Clear Liquify"), key="liq-clear-btn", disabled=not strokes, use_container_width=True):
        strokes.clear()
        request_rerun()
    st.caption(t("{0} stroke(s). Liquify belongs to this photo: it isn't synced to other photos.").format(len(strokes)))


def _numeric_crop_changed() -> None:
    """on_change of the Free-form crop's pixel fields: store the box as the fractions the crop
    editor and the recipe use (crop-free-*). A callback runs before the script, so writing those
    (non-widget) keys is allowed."""
    width, height = st.session_state.get("_crop_frame_size", (1, 1))
    left = min(max(0, int(st.session_state.get("crop-num-left", 0))), width - 1)
    top = min(max(0, int(st.session_state.get("crop-num-top", 0))), height - 1)
    box_w = min(max(1, int(st.session_state.get("crop-num-width", width))), width - left)
    box_h = min(max(1, int(st.session_state.get("crop-num-height", height))), height - top)
    st.session_state["crop-free-x0"], st.session_state["crop-free-y0"] = left / width, top / height
    st.session_state["crop-free-x1"], st.session_state["crop-free-y1"] = (left + box_w) / width, (top + box_h) / height


def crop_section(ctx) -> None:
    section_header("Crop / Aspect Ratio", "crop")
    values = ctx.values
    mask_source = ctx.mask_source
    crop_choice = values["crop_aspect"] = render_control("crop_aspect")
    values["crop_swap"] = render_control("crop_swap", disabled=crop_choice in ("Original", "Free-form"))
    # The custom ratio fields only matter for "Custom", but always render (disabled otherwise), so
    # their values survive switching ratios (a widget that isn't rendered loses its state).
    custom_w_col, custom_h_col = st.columns(2)
    values["crop_custom_w"] = render_control("crop_custom_w", custom_w_col, disabled=crop_choice != "Custom")
    values["crop_custom_h"] = render_control("crop_custom_h", custom_h_col, disabled=crop_choice != "Custom")
    for axis in ("x0", "y0", "x1", "y1"):
        values[f"crop_free_{axis}"] = st.session_state.get(f"crop-free-{axis}", CONTROLS_BY_KEY[f"crop_free_{axis}"].default)
    if crop_choice == "Free-form":
        # Fractions of the frame (not pixels), so the box scales to any image size and persists
        # through Undo/Redo and projects like every other control.
        crop_box_fractional = tuple(values[f"crop_free_{axis}"] for axis in ("x0", "y0", "x1", "y1"))
        # The Guides & Grid panel renders later (canvas column); its last values snap the box.
        crop_grid_type = st.session_state.get("grid-type", "None")
        crop_guides = st.session_state.get("guides", [])
        snap_x = list(grid_line_fractions(crop_grid_type)) + [g["position_pct"] / 100.0 for g in crop_guides if g.get("orientation") == "vertical"]
        snap_y = list(grid_line_fractions(crop_grid_type)) + [g["position_pct"] / 100.0 for g in crop_guides if g.get("orientation") == "horizontal"]

        def build_editor():
            background = preview_image(mask_source, editor_box())
            if crop_grid_type != "None" or crop_guides:
                background = draw_guides_and_grid(background, crop_grid_type, crop_guides)
            return _crop_editor_html(background, "crop-editor", crop_box_fractional, mask_source.size, snap_x=snap_x, snap_y=snap_y)

        show_component(build_editor, canvas_label="✂ Crop", slot_id="crop")
        with st.container(key="crop-editor-hidden"):
            pending_crop = st.text_area(t("Free-form crop box"), key="crop-editor-data", label_visibility="collapsed")
        if consume_new_commit(pending_crop, "crop-editor-committed"):
            try:
                parsed_box = json.loads(pending_crop)
                fx0, fy0, fx1, fy1 = (float(parsed_box[k]) for k in ("x0", "y0", "x1", "y1"))
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                pass
            else:
                st.session_state["crop-free-x0"], st.session_state["crop-free-y0"] = fx0, fy0
                st.session_state["crop-free-x1"], st.session_state["crop-free-y1"] = fx1, fy1
                crop_box_fractional = (fx0, fy0, fx1, fy1)
                request_rerun()
        src_w, src_h = mask_source.size
        fx0, fy0, fx1, fy1 = crop_box_fractional
        crop_box = (int(fx0 * src_w), int(fy0 * src_h), round(fx1 * src_w), round(fy1 * src_h))
        # Numeric entry (F4): the same box in pixels. Set from the box before they render (so they
        # show the editor's latest box); typing calls _numeric_crop_changed().
        st.session_state["_crop_frame_size"] = (src_w, src_h)
        st.session_state["crop-num-left"], st.session_state["crop-num-top"] = crop_box[0], crop_box[1]
        st.session_state["crop-num-width"] = max(1, crop_box[2] - crop_box[0])
        st.session_state["crop-num-height"] = max(1, crop_box[3] - crop_box[1])
        num_cols = st.columns(4)
        for num_col, (num_key, num_label, num_max) in zip(num_cols, (
            ("crop-num-left", "Left (px)", src_w - 1), ("crop-num-top", "Top (px)", src_h - 1),
            ("crop-num-width", "Width (px)", src_w), ("crop-num-height", "Height (px)", src_h),
        )):
            num_col.number_input(t(num_label), min_value=0, max_value=max(1, num_max), step=1, key=num_key, on_change=_numeric_crop_changed)
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            crop_box = None
            st.warning(t("Select a non-empty region."))
        ctx.crop_box = crop_box
    elif crop_choice != "Original":
        from app.controls import aspect_code_from

        ctx.aspect_code = aspect_code_from(values)
        st.caption(t("Center-cropping to {0}.").format(t(ctx.aspect_code)))
    else:
        st.caption(t("No crop applied — showing the full frame."))
    section_reset_button("crop", "Crop")
