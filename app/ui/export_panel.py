"""The Export tab (export settings and presets, watermark, AI Upscale, several sizes at once, Save
Project) and the Save footer under the canvas.

Every widget renders on every rerun; nothing is encoded until a button is clicked -- the Save and
Save Project buttons' `data` are callables Streamlit runs on a separate thread (no st.* there,
every input bound with functools.partial).
"""
from __future__ import annotations

import functools
import io
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from app import ai_upscale, backend_client
from app.batch_release import export_multiple
from app.color_profiles import COLOR_SPACES
from app.export import DEFAULT_NAME_TEMPLATE, EXPORT_FORMATS, FLOAT_FORMATS, HIGH_BIT_FORMATS, LOSSY_FORMATS, METADATA_POLICIES, METADATA_POLICY_LABELS, OUTPUT_SHARPENING, ExportPresetStore, cached_export, render_export_name
from app.painted_masks import inline_painted_masks, session_painted_masks
from app.pipeline import check_upscale_output, expected_output_size
from app.professional_tools import EXPORT_PROFILES, WATERMARK_POSITIONS
from app.reliability import image_digest
from app.ui.layers import font_select, layers_payload
from app.ui.processing import encode_rendered_export, render_full_resolution_export, render_layered_psd
from app.ui.project_io import PROJECT_MAX_BYTES, estimated_project_bytes, project_archive
from app.ui.selections import combined_mask_png, local_masks_payload
from app.ui.state import build_recipe, request_rerun, section_header
from app.i18n import t, t_format, t_option

EXPORT_WIDGET_KEYS = {
    "format": "exp-fmt", "quality": "exp-qual", "long_edge": "exp-long-edge", "sharpen": "exp-sharpen",
    "metadata": "exp-metadata", "remove_gps": "exp-remove-gps", "name_template": "exp-name-template",
    "bit_depth": "exp-bit-depth", "gpu_export": "exp-gpu-export", "color_space": "exp-color-space",
}


def _apply_export_preset(preset: dict) -> None:
    """on_click of "Apply": a callback runs before the script, so it may set widget values."""
    for field, widget_key in EXPORT_WIDGET_KEYS.items():
        if field in preset:
            st.session_state[widget_key] = preset[field]


def export_settings_section(ctx) -> None:
    section_header("Export settings")
    st.caption(t("Used by the 💾 Save button under the canvas (format and quality sit next to it)."))
    export_store = ExportPresetStore()
    saved_presets = export_store.load()
    if saved_presets:
        preset_pick_col, preset_apply_col, preset_delete_col = st.columns([2, 1, 1])
        export_preset_name = preset_pick_col.selectbox(t("Export preset"), sorted(saved_presets), key="exp-preset-sel", label_visibility="collapsed", format_func=t_option)
        preset_apply_col.button(
            t("Apply"), key="exp-preset-apply-btn", use_container_width=True,
            on_click=_apply_export_preset, args=(saved_presets[export_preset_name],),
        )
        if preset_delete_col.button(t("🗑️"), key="exp-preset-delete-btn", use_container_width=True, help=t("Delete this export preset")):
            export_store.delete(export_preset_name)
            request_rerun()
    settings = ctx.export
    exp_set1, exp_set2 = st.columns(2)
    with exp_set1:
        settings["long_edge"] = st.selectbox(
            t("Resize (long edge)"), [0, 4096, 3000, 2048, 1600, 1080], key="exp-long-edge",
            format_func=t_format(lambda edge: "Original size" if not edge else f"{edge} px"),
        )
        settings["sharpen"] = st.selectbox(
            t("Output sharpening"), list(OUTPUT_SHARPENING), key="exp-sharpen",
            help=t("Applied after the resize, for where the image will be seen."), format_func=t_option,
        )
    with exp_set2:
        settings["metadata"] = st.selectbox(
            t("Metadata"), list(METADATA_POLICIES), format_func=t_format(lambda policy: METADATA_POLICY_LABELS[policy]), key="exp-metadata",
            help=t("EXIF from the original photo (camera, lens, capture time, copyright...). The color profile is always kept."),
        )
        # On by default (R4): a location shouldn't leave the machine unless asked for.
        settings["remove_gps"] = st.checkbox(t("Remove GPS location"), value=True, key="exp-remove-gps", disabled=settings["metadata"] != "all")
        # B4: genuine 16-bit PNG/TIFF. Always rendered (only disabled for JPEG/WEBP) so its value
        # survives a format switch; the format selectbox renders later, in the footer.
        settings["bit_depth"] = st.selectbox(
            t("Bit depth"), [8, 16], key="exp-bit-depth", format_func=t_format(lambda bits: f"{bits}-bit"),
            disabled=st.session_state.get("exp-fmt", "JPEG") not in HIGH_BIT_FORMATS,
            help=t("16-bit (PNG/TIFF only) keeps smooth skies and gradients free of banding for further editing or print. Renders at full resolution when you click Save. EXR (half float) and PFM (32-bit float) are always floating point, in linear light, for compositing and HDR tools."),
        )
        settings["color_space"] = st.selectbox(
            t("Color space"), list(COLOR_SPACES), key="exp-color-space", format_func=t_option,
            help=t("sRGB suits the web and most screens. Display P3 is the wider space of current phones, Macs and "
                   "many monitors: the colors look the same in a color-managed viewer, with more room for later edits."),
        )
        settings["gpu_export"] = st.checkbox(
            t("Use GPU for export resize/sharpen"),
            value=False,
            key="exp-gpu-export",
            help=t("Optional CUDA path for final resize and output sharpening. Needs the optional GPU pack or CUDA-enabled PyTorch; file encoding stays on the CPU. If CUDA is missing or fails, export falls back to the CPU."),
        )
    # AI Upscale runs once, on the final full-resolution result, when Save is clicked.
    # ai_upscale_installed() only checks the packages exist (the app's optional GPU pack, see
    # app.gpu_pack), without importing torch into this process.
    if ai_upscale.ai_upscale_installed():
        settings["upscale"] = st.selectbox(
            t("AI Upscale (Real-ESRGAN)"), [0, 2, 4], key="exp-ai-upscale",
            format_func=t_format(lambda factor: "Off" if not factor else f"{factor}x"),
            help=t("Neural super-resolution -- the one deliberate exception to this app's otherwise classical, "
            "CPU-only adjustments (see docs/PROJECT_STATUS.md). Runs on the processing backend (GPU when "
            "available, much slower on CPU) after every other edit, when you click Save."),
        )
    else:
        settings["upscale"] = 0
        st.caption(t(ai_upscale.ai_upscale_install_hint()))
    settings["name_template"] = st.text_input(
        t("File name"), DEFAULT_NAME_TEMPLATE, key="exp-name-template",
        help=t("Fields: {stem} (original name), {date}, {width}, {height}, {format}. The extension is added for you."),
    )
    save_preset_col, save_preset_btn_col = st.columns([2, 1])
    new_export_preset = save_preset_col.text_input(t("Save these settings as"), key="exp-preset-name", placeholder=t("Export preset name"), label_visibility="collapsed")
    if save_preset_btn_col.button(t("💾 Save preset"), key="exp-preset-save-btn", use_container_width=True):
        if new_export_preset.strip():
            export_store.save(new_export_preset.strip(), {
                "format": st.session_state.get("exp-fmt", "JPEG"), "quality": st.session_state.get("exp-qual", 92),
                "long_edge": settings["long_edge"], "sharpen": settings["sharpen"], "metadata": settings["metadata"],
                "remove_gps": settings["remove_gps"], "name_template": settings["name_template"], "bit_depth": settings["bit_depth"],
                "gpu_export": settings["gpu_export"], "color_space": settings["color_space"],
            })
            st.toast(t("Export preset '{0}' saved").format(t(new_export_preset.strip())), icon="💾")
            request_rerun()
        else:
            st.toast(t("Type a name before saving an export preset"), icon="⚠️")


def watermark_section(ctx) -> None:
    """An optional text stamp applied only to exported files (Save and Export Multiple Sizes),
    never to the canvas preview."""
    wm_text = st.session_state.get("wm-text-in", "")
    with st.expander(t("🏷️ Watermark") + (" ●" if wm_text.strip() else ""), key="sec-watermark"):
        st.caption(t("Optional text stamp applied only to exported files, not the canvas preview."))
        text = st.text_input(t("Watermark text (blank = none)"), key="wm-text-in")
        wm_pos_col, wm_op_col, wm_size_col = st.columns(3)
        with wm_pos_col:
            position = st.selectbox(t("Position"), list(WATERMARK_POSITIONS), index=3, key="wm-pos-sel", format_func=t_option)
        with wm_op_col:
            opacity = st.slider(t("Opacity %"), 5, 100, 50, key="wm-opacity-sl")
        with wm_size_col:
            font_size = st.slider(t("Font size"), 12, 120, 32, key="wm-size-sl")
        font = font_select(t("Font"), st.session_state.get("wm-font-sel", ""), key="wm-font-sel")
    ctx.export["watermark"] = {"text": text, "position": position, "opacity": opacity, "font_size": font_size, "font": font}


def _full_options(ctx) -> dict:
    full_options = {key: value for key, value in ctx.options.items() if key != "preview_scale"}
    if ctx.export.get("upscale"):
        full_options.update(ai_upscale=True, ai_upscale_factor=ctx.export["upscale"])
    return full_options


def export_multiple_section(ctx) -> None:
    """"Export As" several EXPORT_PROFILES (Web/Social/Print/Archive) of this one photo at once --
    distinct from the Batch Processor (one recipe, many photos)."""
    with st.expander(t("📦 Export Multiple Sizes")):
        selected_profiles = st.multiselect(t("Sizes to include"), list(EXPORT_PROFILES), default=["Web", "Social"], key="exp-multi-profiles", format_func=t_option)
        multi_remove_gps = st.checkbox(
            t("Remove GPS location"), value=True, key="exp-multi-remove-gps",
            help=t("Web and Social carry no EXIF at all; this also strips the location from Print and Archive."),
        )
        if st.button(t("Prepare ZIP"), key="exp-multi-btn", disabled=not selected_profiles):
            multi_source = ctx.edited_image
            if ctx.use_proxy or ctx.export.get("gpu_export"):
                # Print/Archive are full-resolution: render it (lossless) on the backend first.
                mask_source, mask_source_key = ctx.mask_source, ctx.mask_source_key
                with st.spinner(t("Rendering at full resolution…")):
                    multi_source = Image.open(io.BytesIO(backend_client.export_via_backend(
                        ctx.upload_bytes, ctx.upload_name, _full_options(ctx) | {"ai_upscale": False}, {"format": "PNG", "metadata": "all"},
                        combined_mask_png(mask_source_key, mask_source, ctx.mask_signature) if ctx.shared_mask_layers else None,
                        layers_payload(ctx.layers_signature, tuple(ctx.full_canvas), 1.0, mask_source_key, mask_source) if ctx.layers_signature else None,
                        combined_mask_png(mask_source_key, mask_source, ctx.removal_signature) if ctx.removal_active else None,
                        **({"local_masks_bytes": local_masks_payload([combined_mask_png(mask_source_key, mask_source, signature) for signature in ctx.own_signatures])} if ctx.own_signatures else {}),
                    )))
            watermark = ctx.export["watermark"]
            zip_bytes = export_multiple(
                multi_source, selected_profiles, Path(ctx.upload_name).stem,
                watermark_text=watermark["text"] if watermark["text"].strip() else None,
                watermark_position=watermark["position"],
                watermark_opacity=watermark["opacity"],
                watermark_font_size=watermark["font_size"],
                watermark_font=watermark.get("font", ""),
                exif=ctx.source_exif,
                remove_gps=multi_remove_gps,
                gpu_export=ctx.export.get("gpu_export", False),
            )
            st.session_state["_exp_multi_zip"] = zip_bytes
            request_rerun()
        if st.session_state.get("_exp_multi_zip"):
            st.download_button(
                t("⬇️ Download ZIP"),
                st.session_state["_exp_multi_zip"],
                file_name=f"{Path(ctx.upload_name).stem}-exports.zip",
                mime="application/zip",
                key="exp-multi-dl-btn",
            )


def layered_psd_section(ctx) -> None:
    """⬇ Layered PSD (F22): the edited photo as the Background plus every Layers-tab layer as its own
    Photoshop layer, built only when the button is pressed (a data callable on the download thread)."""
    st.markdown(t("**Layered PSD**"))
    layers = st.session_state.get("image_layers", [])
    st.caption(t("For Photoshop, GIMP, Krita or Affinity: the edited photo as the Background and each layer from the Layers tab as its own layer (rasterized), with its name, opacity, blend mode and visibility."))
    producer = functools.partial(
        render_layered_psd, ctx.upload_bytes, ctx.upload_name, _full_options(ctx), ctx.mask_source,
        ctx.mask_signature if ctx.shared_mask_layers else None, ctx.removal_signature if ctx.removal_active else None,
        ctx.layers_signature, ctx.own_signatures, tuple(ctx.full_canvas), session_painted_masks(),
    )
    st.download_button(
        t("⬇ Download layered PSD"),
        data=producer,
        file_name=f"{Path(ctx.upload_name).stem}.psd",
        mime="image/vnd.adobe.photoshop",
        key="exp-psd-btn",
        help=None if layers else t("No layers yet: the PSD holds the edited photo as a single Background layer."),
    )


def project_file_section(ctx) -> None:
    st.markdown(t("**Project File**"))
    st.caption(t("Save every adjustment (plus the source image and any layers) into one file you can reopen later."))
    # One click: the archive is built only when the button is pressed (a data callable), from the
    # recipe (painted masks inlined) and the layer stack as they are on this rerun.
    project_layers = json.loads(json.dumps(st.session_state.get("image_layers", [])))
    project_estimate = estimated_project_bytes(ctx.upload_bytes, project_layers)
    project_too_big = project_estimate > PROJECT_MAX_BYTES
    if project_too_big:
        st.warning(
            t("⚠️ This project would be about {0:.0f} MB -- over the {1} MB a project can be reopened with. Remove or shrink some image layers.").format(t(project_estimate / 1024 ** 2), t(PROJECT_MAX_BYTES // 1024 ** 2))
        )
    st.download_button(
        t("💾 Save Project (.photoproject)"),
        data=functools.partial(project_archive, ctx.upload_name, ctx.upload_bytes, json.loads(json.dumps(inline_painted_masks(build_recipe()))), project_layers),
        file_name=f"{Path(ctx.upload_name).stem}.photoproject",
        mime="application/zip",
        key="proj-dl-btn",
        disabled=project_too_big,
    )
    st.caption(t("Text, Shape and Path layers stay editable when the project is reopened."))


def save_footer(ctx) -> None:
    """Format, quality and the 💾 Save button (Ctrl+E) under the canvas."""
    settings = ctx.export
    st.markdown(t('<div class="export-panel">'), unsafe_allow_html=True)
    exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 1.2], gap="small")
    with exp_c1:
        out_format = st.selectbox(t("Export Format"), list(EXPORT_FORMATS), key="exp-fmt", label_visibility="collapsed", format_func=t_option)
    with exp_c2:
        out_quality = st.slider(t("Quality"), 50, 100, 92, key="exp-qual", disabled=out_format not in LOSSY_FORMATS, label_visibility="collapsed")
    with exp_c3:
        watermark = settings["watermark"]
        export_settings = {
            "format": out_format, "quality": out_quality, "long_edge": settings["long_edge"], "sharpen": settings["sharpen"],
            "metadata": settings["metadata"], "remove_gps": settings["remove_gps"],
            "watermark": watermark if watermark["text"].strip() else None,
            # EXR/PFM (F24) are always float, rendered from the high-precision path.
            "bit_depth": 16 if out_format in FLOAT_FORMATS else settings["bit_depth"] if out_format in HIGH_BIT_FORMATS else 8,
            "gpu_export": settings["gpu_export"],
            "color_space": settings["color_space"],
        }
        export_extension, export_mime = EXPORT_FORMATS[out_format]
        # A full-resolution backend render (which also encodes the file) is needed for a proxy
        # preview, a 16-bit file, AI Upscale, or Display P3 (rendered in a P3 working space from the
        # original file, B14); otherwise the rendered result is encoded here.
        full_options = _full_options(ctx)
        upscale = settings.get("upscale", 0)
        upscale_error = None
        if upscale:
            try:
                check_upscale_output(expected_output_size(ctx.source.size, ctx.options), upscale)
            except ValueError as error:
                upscale_error = str(error)
                st.warning(t("⚠️ {0}").format(t(error)))
        export_key = image_digest(json.dumps(
            [ctx.source_digest, full_options, ctx.mask_source_key, ctx.mask_signature, ctx.own_signatures,
             ctx.removal_signature if ctx.removal_active else None, ctx.layers_key, export_settings, ctx.use_proxy],
            sort_keys=True, default=str,
        ).encode("utf-8"))
        # A finished refine-on-pause render already *is* the full-resolution result.
        if ((ctx.use_proxy and ctx.refined_image is None) or upscale or export_settings["bit_depth"] == 16
                or export_settings["gpu_export"] or export_settings["color_space"] == "Display P3"):
            export_producer = functools.partial(
                render_full_resolution_export, ctx.upload_bytes, ctx.upload_name, full_options, export_settings, ctx.mask_source,
                ctx.mask_signature if ctx.shared_mask_layers else None, ctx.removal_signature if ctx.removal_active else None, ctx.layers_signature,
                ctx.own_signatures, tuple(ctx.full_canvas), session_painted_masks(),
            )
        else:
            export_producer = functools.partial(encode_rendered_export, ctx.edited_image, export_settings, ctx.source_exif)
        st.download_button(
            label=t("💾 Save {0}").format(t(out_format)),
            data=functools.partial(cached_export, export_key, export_producer),
            file_name=render_export_name(settings["name_template"], Path(ctx.upload_name).stem, ctx.output_size, out_format, export_extension),
            mime=export_mime,
            use_container_width=True,
            key="exp-download-btn",
            shortcut="Ctrl+E",
            disabled=upscale_error is not None,
        )
    st.markdown(t('</div>'), unsafe_allow_html=True)
