"""Turning the panels' values into a rendered edit: the two-stage backend calls and their caches,
the proxy preview, refine-on-pause, and the full-resolution producers the Save button and the
refine thread run.

The frontend never edits pixels itself (app/backend.py does, through app.pipeline.run_pipeline);
the local `source` is only decoded here, for previews, mask geometry and face detection.

Cache rules: every st.cache_data has max_entries (a full-resolution result is ~70 MB), the full-
resolution ones a TTL too (P5); large inputs are passed as an unhashed `_argument` next to a short
hashed key -- the photo's digest (source_digest, hashed once per upload) or a derived digest.
Backend functions are looked up on app.backend_client at call time (tests replace them).
"""
from __future__ import annotations

import functools
import io
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app import backend_client, export as export_module, portrait_tools
from app.backend import process_image
from app.batch_release import FileLike
from app.controls import controls_to_options, curve_channels_from, curve_fixed_points
from app.edit_schema import DETAIL_STAGE_OPTION_KEYS, sanitize_recipe
from app.geometry import apply_geometry, geometry_output_size
from app.image_processor import PhotoProcessor
from app.painted_masks import session_painted_masks
from app.pipeline import expected_output_size
from app.presets import PRESETS, apply_preset
from app.professional_color import gamut_warning_overlay
from app.recipe_options import merge_curve_points, recipe_to_options
from app.refine_preview import RefineTracker
from app.reliability import image_digest
from app.ui.layers import build_layers_payload, layers_payload
from app.ui.selections import combined_mask_png, combined_mask_png_uncached, local_masks_payload
from app.ui.state import in_fragment_rerun, request_rerun
from app.i18n import t

# The full-resolution caches keep a few recent entries (enough for Undo/Redo or nudging a slider
# back) and drop them after an hour unused; thumbnail-sized ones can afford more.
FULL_RES_CACHE_ENTRIES = 4
FULL_RES_CACHE_TTL = 3600
THUMBNAIL_CACHE_ENTRIES = 16

# Interactive previews of a photo larger than PROXY_TRIGGER_LONG_EDGE are rendered from a proxy
# whose long edge is PROXY_LONG_EDGE (P1): the canvas never shows more than ~1200x900. Full
# resolution is rendered for export, for the 1:1+ zoom views, and after a pause (refine-on-pause).
PROXY_LONG_EDGE = 2048
PROXY_TRIGGER_LONG_EDGE = 2560
# A request option both stages need.
BOTH_STAGES_OPTION_KEYS = ("preview_scale",)


def png_bytes(image: Image.Image, **extra) -> bytes:
    buffer = io.BytesIO()
    # compress_level=1: every PNG built here is an internal hand-off, not a delivered file.
    image.save(buffer, format="PNG", compress_level=1, **extra)
    return buffer.getvalue()


def source_digest(identity: object, content: bytes) -> str:
    """SHA-256 of the open photo, computed once per distinct upload rather than per rerun --
    `identity` is the uploader's file_id (or the restored bytes object's id())."""
    cached = st.session_state.get("_source_digest")
    if cached and cached[0] == identity:
        return cached[1]
    digest = image_digest(content)
    st.session_state["_source_digest"] = (identity, digest)
    return digest


@st.cache_resource(show_spinner="Opening photo…", max_entries=2)
def decoded_source(digest: str, _content: bytes, filename: str) -> Image.Image:
    """The decoded source photo (RAW included), decoded once per upload. st.cache_resource hands
    back the same object each time, so every caller treats it as read-only."""
    return PhotoProcessor.from_bytes(_content, filename).image


@st.cache_data(show_spinner=False, max_entries=THUMBNAIL_CACHE_ENTRIES)
def preset_thumbnails(digest: str, _source: Image.Image) -> dict[str, bytes]:
    """Small previews of each built-in look, rendered in-process from a thumbnail -- decorative
    only; the real edit always goes through the backend."""
    base = _source.copy()
    base.thumbnail((96, 96), Image.Resampling.LANCZOS)
    thumbnails = {"None": base}
    for name in PRESETS:
        preview = PhotoProcessor.from_image(base)
        apply_preset(preview, name)
        thumbnails[name] = preview.image
    return {name: png_bytes(thumb) for name, thumb in thumbnails.items()}


LOOK_PREVIEW_EDGE = 640


def _jpeg_b64(image: Image.Image, quality: int) -> str:
    import base64

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@st.cache_data(show_spinner=False, max_entries=48)
def look_preview(digest: str, _source: Image.Image, recipe_json: str) -> tuple[str, str]:
    """A look gallery card (UI5): `recipe` applied to the photo, as base64 JPEGs -- a 160 px card
    thumbnail and a 640 px preview the gallery shows over the canvas on hover. Rendered in-process
    on a small copy, like custom_preset_thumbnail(): decorative only, the real edit goes through
    the backend. Falls back to the plain photo if the recipe can't be rendered."""
    base = _source.copy()
    base.thumbnail((LOOK_PREVIEW_EDGE, LOOK_PREVIEW_EDGE), Image.Resampling.LANCZOS)
    try:
        preview = process_image(FileLike("preview.png", png_bytes(base)), recipe_to_options(sanitize_recipe(json.loads(recipe_json))))
    except (KeyError, ValueError, OSError, RuntimeError, cv2.error):
        preview = base
    thumb = preview.copy()
    thumb.thumbnail((160, 160), Image.Resampling.LANCZOS)
    return _jpeg_b64(thumb, 80), _jpeg_b64(preview, 82)


@st.cache_data(show_spinner=False, max_entries=THUMBNAIL_CACHE_ENTRIES)
def custom_preset_thumbnail(digest: str, _source: Image.Image, recipe: dict) -> bytes:
    """A preview of one saved custom preset: a full recipe, so only the real pipeline (called
    in-process on a thumbnail) reproduces it. Falls back to the plain thumbnail if it fails."""
    base = _source.copy()
    base.thumbnail((96, 96), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    base.save(buffer, format="PNG")
    try:
        preview = process_image(FileLike("thumbnail.png", buffer.getvalue()), recipe_to_options(sanitize_recipe(recipe)))
    except (KeyError, ValueError, OSError, RuntimeError, cv2.error):
        preview = base
    encoded = io.BytesIO()
    preview.convert("RGB").save(encoded, format="PNG")
    return encoded.getvalue()


@st.cache_data(show_spinner=False, max_entries=2)
def proxy_source_png(digest: str, _source: Image.Image, long_edge: int) -> bytes:
    """The reduced-size stand-in the backend edits for the interactive preview, keeping the
    source's ICC profile and EXIF."""
    proxy = _source.copy()
    proxy.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    extra = {}
    if _source.info.get("icc_profile"):
        extra["icc_profile"] = _source.info["icc_profile"]
    exif = _source.getexif()
    if len(exif):
        extra["exif"] = exif.tobytes()
    return png_bytes(proxy, **extra)


@st.cache_data(show_spinner="🎨 Applying edits (tone/color/detail)…", max_entries=FULL_RES_CACHE_ENTRIES, ttl=FULL_RES_CACHE_TTL)
def cached_process_through_detail(source_key: str, _image_bytes: bytes, filename: str, detail_options: dict) -> tuple[bytes, str | None]:
    """Stage 1 of 2 (Geometry/Lens -> Tone -> Color -> Detail; see app.pipeline), cached apart from
    every Repair-onward option so a Mask slider doesn't rerun the Non-Local Means denoise. Returns
    the backend's own PNG (ICC and EXIF kept, P3) and its handle for this result."""
    options = dict(detail_options)
    options["stop_after"] = "detail"
    result = backend_client.process_via_backend(_image_bytes, filename, options)
    png = result.info.get("response_png")
    if png is None:
        extra = {key: result.info[key] for key in ("icc_profile", "exif") if result.info.get(key)}
        png = png_bytes(result, **extra)
    return png, result.info.get("stage_handle")


@st.cache_data(show_spinner="🎨 Applying edits…", max_entries=FULL_RES_CACHE_ENTRIES, ttl=FULL_RES_CACHE_TTL)
def cached_process_from_repair_onward(
    stage_a_key: str, _stage_a_png_bytes: bytes, remaining_options: dict,
    local_mask_key: str | None, _local_mask_bytes: bytes | None,
    layers_key: str | None, _layers_bytes: bytes | None,
    removal_mask_key: str | None, _removal_mask_bytes: bytes | None,
    _stage_handle: str | None = None,
    local_masks_key: str | None = None, _local_masks_bytes: bytes | None = None,
) -> Image.Image:
    """Stage 2 of 2 -- Repair through Effects, resuming from Stage 1 (by handle; the PNG is only
    re-sent when the backend no longer holds it). Each large input arrives as a short hashed key
    plus unhashed bytes."""
    options = dict(remaining_options)
    options["resume_from_detail"] = True
    extra = {}
    if _stage_handle:
        extra["stage_handle"] = _stage_handle
    if _local_masks_bytes:
        extra["local_masks_bytes"] = _local_masks_bytes
    return backend_client.process_via_backend(_stage_a_png_bytes, "stage_a.png", options, _local_mask_bytes, _layers_bytes, _removal_mask_bytes, **extra)


@st.cache_resource(show_spinner=False, max_entries=2)
def lens_corrected_source(digest: str, _source: Image.Image, geometry_json: str) -> Image.Image:
    """What Crop and the Masks/Repair selection tools work against: the source after the
    pipeline's Geometry step (app.geometry.apply_geometry, the function the backend runs) -- a box
    or mask drawn on the raw source would land on the wrong pixels once the backend warps it.
    Returned by reference (read-only); the shared source itself when nothing changes geometry."""
    geometry = json.loads(geometry_json)
    if not any(value for key, value in geometry.items() if key not in ("lens_vignette_midpoint", "constrain_crop")):
        return _source
    return apply_geometry(_source, geometry)


@st.cache_data(show_spinner=False, max_entries=FULL_RES_CACHE_ENTRIES, ttl=FULL_RES_CACHE_TTL)
def cached_gamut_warning_overlay(digest: str, _source: Image.Image, icc_bytes: bytes):
    """Two full-image littlecms conversions -- only when the photo or its profile changes."""
    return gamut_warning_overlay(_source, icc_bytes)


@st.cache_data(show_spinner=False, max_entries=4)
def face_count(digest: str, _source: Image.Image) -> int:
    """Face detection for the Portrait caption -- once per photo, on a 1024 px proxy."""
    return len(portrait_tools.detect_faces(_source))


def encode_rendered_export(image: Image.Image, settings: dict, exif: bytes | None) -> bytes:
    """Save-button producer when the canvas already holds the full-resolution result."""
    return export_module.encode_export(image, settings, exif=exif)[0]


def _producer_masks(mask_source, local_signature, removal_signature, own_signatures, painted) -> tuple:
    local = combined_mask_png_uncached(mask_source, local_signature, painted) if local_signature else None
    removal = combined_mask_png_uncached(mask_source, removal_signature, painted) if removal_signature else None
    extra = {}
    if own_signatures:
        extra["local_masks_bytes"] = local_masks_payload([combined_mask_png_uncached(mask_source, signature, painted) for signature in own_signatures])
    return local, removal, extra


def _producer_layers(mask_source, layers_signature, layers_canvas, painted) -> bytes | None:
    """The Layers payload at full resolution for a producer thread; layer masks (F13) are built
    uncached from the bound painted-mask table."""
    if not layers_signature:
        return None
    return build_layers_payload(layers_signature, layers_canvas, mask_builder=lambda signature, _size: combined_mask_png_uncached(mask_source, signature, painted))


def render_full_resolution_export(
    content: bytes, filename: str, options: dict, settings: dict, mask_source: Image.Image,
    local_signature: str | None, removal_signature: str | None, layers_signature: str | None,
    own_signatures: list[str] | None = None, layers_canvas: tuple[int, int] | None = None, painted: dict | None = None,
) -> bytes:
    """Save-button producer for a proxy-previewed photo, a 16-bit file or an AI Upscale export:
    one full-resolution backend render that also encodes the delivery file. Runs on Streamlit's
    download thread (no st.* calls; every input bound up front with functools.partial)."""
    local, removal, extra = _producer_masks(mask_source, local_signature, removal_signature, own_signatures, painted)
    return backend_client.export_via_backend(
        content, filename, options, settings, local,
        _producer_layers(mask_source, layers_signature, layers_canvas, painted), removal, **extra,
    )


def render_layered_psd(
    content: bytes, filename: str, options: dict, mask_source: Image.Image,
    local_signature: str | None, removal_signature: str | None, layers_signature: str | None,
    own_signatures: list[str] | None = None, layers_canvas: tuple[int, int] | None = None, painted: dict | None = None,
) -> bytes:
    """Layered PSD producer (F22): the full-resolution edit *without* the Layers stack as the
    Background, then every layer rasterized at the full frame (styles and layer masks applied),
    cropped like the photo, as its own PSD layer with its name, opacity, blend mode and visibility.
    The composite is the edit with the layers. Runs on the download thread (no st.* calls)."""
    from app.layer_workspace import layers_from_json
    from app.pipeline import crop_rect
    from app.psd import PsdLayer, write_psd

    local, removal, extra = _producer_masks(mask_source, local_signature, removal_signature, own_signatures, painted)
    options = {**options, "ai_upscale": False}
    png = {"format": "PNG", "metadata": "none"}
    background = Image.open(io.BytesIO(backend_client.export_via_backend(content, filename, options, png, local, None, removal, **extra))).convert("RGB")
    layers_bytes = _producer_layers(mask_source, layers_signature, layers_canvas, painted)
    composite = background
    if layers_bytes:
        composite = Image.open(io.BytesIO(backend_client.export_via_backend(content, filename, options, png, local, layers_bytes, removal, **extra))).convert("RGB")
    psd_layers = []
    rect = crop_rect(tuple(layers_canvas), options) if layers_canvas else None
    for layer in layers_from_json(layers_bytes) if layers_bytes else []:
        if layer.is_adjustment:
            continue  # no pixels: its effect is in the composite only (a PSD adjustment layer isn't written)
        image = layer.image.convert("RGBA")
        if layers_canvas and image.size != tuple(layers_canvas):
            image = image.resize(tuple(layers_canvas), Image.Resampling.LANCZOS)
        if layer.mask is not None:
            alpha = np.asarray(image.getchannel("A"), dtype=np.uint16) * np.asarray(layer.mask.convert("L").resize(image.size), dtype=np.uint16) // 255
            image.putalpha(Image.fromarray(alpha.astype(np.uint8)))
        if rect:
            image = image.crop(rect)
        if image.size != background.size:
            image = image.resize(background.size, Image.Resampling.LANCZOS)
        psd_layers.append(PsdLayer(layer.name, image, 0, 0, layer.opacity, layer.blend_mode, layer.visible))
    return write_psd(composite, [PsdLayer("Background", background.convert("RGBA")), *psd_layers])


def render_layers_svg(
    content: bytes, filename: str, options: dict, mask_source: Image.Image,
    local_signature: str | None, removal_signature: str | None, own_signatures: list[str] | None,
    layers: list[dict], frame: tuple[int, int], painted: dict | None, include_photo: bool,
) -> bytes:
    """SVG producer (F27): the Layers stack as vectors (app.svg_export), cropped like the photo, over
    the full-resolution edit without the layers when `include_photo`. Download thread: no st.*."""
    from app.pipeline import crop_rect
    from app.svg_export import layers_to_svg

    background = None
    if include_photo:
        local, removal, extra = _producer_masks(mask_source, local_signature, removal_signature, own_signatures, painted)
        png = backend_client.export_via_backend(content, filename, {**options, "ai_upscale": False}, {"format": "PNG", "metadata": "none"},
                                                local, None, removal, **extra)
        background = Image.open(io.BytesIO(png))
    return layers_to_svg(layers, tuple(frame), crop_rect(tuple(frame), options), background).encode("utf-8")


def render_full_resolution_image(
    content: bytes, filename: str, options: dict, mask_source: Image.Image,
    local_signature: str | None, removal_signature: str | None, layers_signature: str | None,
    own_signatures: list[str] | None = None, layers_canvas: tuple[int, int] | None = None, painted: dict | None = None,
) -> Image.Image:
    """Refine-on-pause producer (P1, app.refine_preview): the same full-resolution render, decoded
    for the canvas, on a background thread in the backend's "refine" lane (any newer interactive
    request of this session supersedes it)."""
    local, removal, extra = _producer_masks(mask_source, local_signature, removal_signature, own_signatures, painted)
    return backend_client.process_via_backend(
        content, filename, options, local,
        _producer_layers(mask_source, layers_signature, layers_canvas, painted), removal,
        lane=backend_client.REFINE_LANE, timeout=backend_client.refine_timeout(options), **extra,
    )


def refine_status_view() -> None:
    """The canvas's refine-on-pause status line, run as a fragment that polls while a refine is
    pending: it starts the full-resolution job once the edit has paused, and when the job finishes
    reruns the whole page so the canvas, histogram and export switch to it (deferred to the end of
    the script in a full run; immediate in a fragment-only run, where nothing else is running)."""
    tracker = st.session_state.get("_refine_tracker")
    if tracker is None:
        return
    tracker.maybe_start()
    if tracker.needs_rerun():
        if in_fragment_rerun():
            st.rerun(scope="app")
        request_rerun()
    status = tracker.status()
    if status == "waiting":
        st.caption(t("⚡ Fast preview -- a full-resolution render follows when you pause."))
    elif status == "running":
        st.caption(t("⏳ Rendering full resolution…"))
    elif status == "ready":
        st.caption(t("✨ Full-resolution view"))
    elif status == "failed":
        st.caption(t("⚡ Fast preview (the full-resolution render didn't finish; it is still used for Save)."))


def build_options(ctx) -> dict:
    """The pipeline options for this run: every registry control's widget value through the one
    mapping (app.controls), plus the composite pieces the panels assembled."""
    values = ctx.values
    options = controls_to_options(values)
    options["hsl_bands"] = ctx.hsl_bands
    options["curve"] = merge_curve_points(curve_fixed_points(values), st.session_state.get("curve_extra_points", [])) if values.get("curve_enabled") else None
    options["curve_channels"] = curve_channels_from(values)
    options["local_adjustments"] = [dict(layer["adjustments"]) for layer in ctx.own_mask_layers] or None
    options["aspect"] = ctx.aspect_code
    options["crop_box"] = list(ctx.crop_box) if ctx.crop_box else None
    options["liquify"] = list(st.session_state.get("liquify_strokes", [])) or None
    from app.ui.plugin_filters import pipeline_steps

    options["plugin_filters"] = pipeline_steps()
    return options


def render_edit(ctx) -> None:
    """Run the edit through the backend (two cached stages, proxy-sized for a large photo unless a
    1:1 view or a finished refine needs real pixels) and fill ctx.edited_image/output_size."""
    source, digest = ctx.source, ctx.source_digest
    options = ctx.options = build_options(ctx)
    # The zoom selectbox renders later (canvas column); its last value is read from session_state.
    full_res_view = float(st.session_state.get("cv-zoom", 0.0)) >= 1.0
    use_proxy = ctx.use_proxy = max(source.size) > PROXY_TRIGGER_LONG_EDGE and not full_res_view
    mask_size = None
    if use_proxy:
        stage_input = proxy_source_png(digest, source, PROXY_LONG_EDGE)
        stage_input_name, stage_input_key = "proxy.png", f"{digest}|proxy{PROXY_LONG_EDGE}"
        proxy_size = Image.open(io.BytesIO(stage_input)).size
        # Scales the pipeline's pixel-unit radii and the full-resolution crop_box (app.pipeline).
        options["preview_scale"] = proxy_size[0] / source.width
        # Masks apply after the geometry step: built at the proxy's post-geometry size.
        mask_size = geometry_output_size(proxy_size, options)
    else:
        stage_input, stage_input_name, stage_input_key = ctx.upload_bytes, ctx.upload_name, digest

    mask_source, mask_source_key = ctx.mask_source, ctx.mask_source_key
    local_mask_bytes = local_mask_key = None
    if ctx.shared_mask_layers:
        local_mask_bytes = combined_mask_png(mask_source_key, mask_source, ctx.mask_signature, mask_size)
        local_mask_key = image_digest(f"{mask_source_key}|{ctx.mask_signature}|{mask_size}".encode("utf-8"))
    local_masks_bytes = local_masks_key = None
    if ctx.own_signatures:
        local_masks_bytes = local_masks_payload([combined_mask_png(mask_source_key, mask_source, signature, mask_size) for signature in ctx.own_signatures])
        local_masks_key = image_digest(f"{mask_source_key}|{ctx.own_signatures}|{mask_size}".encode("utf-8"))
    removal_mask_bytes = removal_mask_key = None
    if ctx.removal_active:
        removal_mask_bytes = combined_mask_png(mask_source_key, mask_source, ctx.removal_signature, mask_size)
        removal_mask_key = image_digest(f"{mask_source_key}|{ctx.removal_signature}|{mask_size}".encode("utf-8"))
    # Layers are composited after the geometry step, at that frame's size (the proxy's when
    # previewing through one, with pixel sizes scaled to match).
    full_canvas = ctx.full_canvas = geometry_output_size(source.size, options)
    layers_bytes = None
    if ctx.layers_signature:
        if use_proxy:
            layers_bytes = layers_payload(ctx.layers_signature, tuple(mask_size), mask_size[0] / max(1, full_canvas[0]), mask_source_key, mask_source)
        else:
            layers_bytes = layers_payload(ctx.layers_signature, tuple(full_canvas), 1.0, mask_source_key, mask_source)
    ctx.layers_key = image_digest(layers_bytes) if layers_bytes else None

    detail_options = {key: options[key] for key in (*DETAIL_STAGE_OPTION_KEYS, *BOTH_STAGES_OPTION_KEYS) if key in options}
    remaining_options = {key: value for key, value in options.items() if key not in DETAIL_STAGE_OPTION_KEYS}
    # Refine-on-pause (P1): while a proxy is in use, the edit's key plus a bound full-resolution
    # producer; once a background render of this exact edit finishes it replaces the proxy result.
    refine_tracker = ctx.refine_tracker = st.session_state.setdefault("_refine_tracker", RefineTracker())
    refine_key = refine_render = None
    if use_proxy:
        full_render_options = {key: value for key, value in options.items() if key != "preview_scale"}
        refine_inputs = [
            digest, full_render_options, mask_source_key, ctx.mask_signature if ctx.shared_mask_layers else None,
            ctx.own_signatures, ctx.removal_signature if ctx.removal_active else None, ctx.layers_signature,
        ]
        refine_key = image_digest(json.dumps(refine_inputs, sort_keys=True, default=str).encode("utf-8"))
        refine_render = functools.partial(
            render_full_resolution_image, ctx.upload_bytes, ctx.upload_name, full_render_options, mask_source,
            ctx.mask_signature if ctx.shared_mask_layers else None, ctx.removal_signature if ctx.removal_active else None, ctx.layers_signature,
            ctx.own_signatures, tuple(full_canvas), session_painted_masks(),
        )
    refine_tracker.observe(refine_key, refine_render)
    refined_image = ctx.refined_image = refine_tracker.result(refine_key)
    try:
        if refined_image is not None:
            edited_image = refined_image
        else:
            stage_a_png_bytes, stage_a_handle = cached_process_through_detail(stage_input_key, stage_input, stage_input_name, detail_options)
            stage_a_key = image_digest(f"{stage_input_key}|{json.dumps(detail_options, sort_keys=True)}".encode("utf-8"))
            edited_image = cached_process_from_repair_onward(
                stage_a_key, stage_a_png_bytes, remaining_options,
                local_mask_key, local_mask_bytes, ctx.layers_key, layers_bytes, removal_mask_key, removal_mask_bytes,
                stage_a_handle, local_masks_key, local_masks_bytes,
            )
    except backend_client.SupersededError:
        # A newer edit from this session replaced this render -- this run is already obsolete.
        st.stop()
    except RuntimeError as error:
        st.error(t("⚠️ {0}").format(t(error)))
        st.stop()
    ctx.edited_image = edited_image
    # What an export will measure -- on a proxy preview, the real (full-resolution) output size.
    ctx.output_size = expected_output_size(source.size, options) if use_proxy and refined_image is None else edited_image.size
