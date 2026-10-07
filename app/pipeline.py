"""The edit pipeline -- the single implementation of the step order every entry point shares.

app/backend.py's /process route, app/cli.py (single image and --input-dir), app/batch.py and
app/batch_release.py all call run_pipeline() instead of each applying the steps themselves. Before
this module existed the CLI and folder batch had their own hand-written copies, which had drifted
into a different order with fewer steps (contrast/saturation before exposure, texture before
temperature, denoise/sharpen after local masks, no ICC/lens/vibrance/HSL/grading/dehaze/effects),
so the same flags gave a different image than the GUI. The order, in one place:

    ICC -> sRGB, Geometry (rotate/flip, Lens, straighten, constrain crop), Tone (+ preset), Color,
    Detail | Repair, Layers, Masks, Portrait, Crop, Effects, AI Upscale

The "|" is the split the frontend's two-stage cache is built on (`stop_after="detail"` /
`resume_from_detail`, see run_pipeline()'s docstring).
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from io import BytesIO

from PIL import Image

from app.config import MAX_HEAVY_JOBS, MAX_UPSCALE_OUTPUT_PIXELS
from app.edit_schema import parse_aspect, parse_options
from app.geometry import geometry_output_size
from app.high_bit import attach, attached, decode_high_bit, to_image
from app.image_processor import PhotoProcessor
from app.layer_workspace import LayerWorkspace, WorkspaceLayer
from app.portrait_tools import smooth_skin
from app.presets import apply_preset
from app.professional_color import ICC_TRANSFORM_ERRORS, normalize_to_srgb, srgb_icc_bytes
from app.raw_import import is_raw_path
from app.reliability import check_pixel_count, validate_image_bytes

# Skin smoothing's blur radius at full resolution (see app.portrait_tools.smooth_skin).
_SKIN_BLUR_RADIUS = 2.0

# The Black & White mixer's option per colour band (spelled out: tests/test_parity_golden.py checks
# that every option name appears in this file).
_BW_MIX_OPTIONS = {
    "Red": "bw_red", "Orange": "bw_orange", "Yellow": "bw_yellow", "Green": "bw_green",
    "Aqua": "bw_aqua", "Blue": "bw_blue", "Purple": "bw_purple", "Magenta": "bw_magenta",
}

# Caps how many CPU-heavy steps (Non-Local Means denoise, dehaze) run at once across every request
# thread of the backend, so a burst of requests can't oversubscribe the CPU (B11).
HEAVY_STEP_SEMAPHORE = threading.BoundedSemaphore(MAX_HEAVY_JOBS)


class PipelineCancelled(Exception):
    """Raised between steps when `should_cancel()` says a newer request superseded this one. Not a
    ValueError on purpose: it isn't a bad request (app.backend answers 409)."""


def decode_source(content: bytes, filename: str, *, high_bit: bool = False) -> Image.Image:
    """Validate and decode an uploaded/source file (RAW included) into an RGB image whose .info
    still carries the file's EXIF (Orientation already applied and removed) and ICC profile.
    PIL (via validate_image_bytes) can't decode camera RAW -- PhotoProcessor.from_bytes() routes it
    through rawpy and raises its own RuntimeError on failure -- so RAW skips the PIL check and gets
    the exact pixel limit applied to its decoded size instead.

    `high_bit=True` (a 16-bit export, B4): a source that really holds more than 8 bits (16-bit
    PNG/TIFF, camera RAW) is decoded at full precision too; the returned 8-bit image is its
    quantization and carries the float pixels (app.high_bit.attached()) for run_pipeline()."""
    if not is_raw_path(filename):
        validate_image_bytes(content, filename)
    if high_bit:
        array = decode_high_bit(content, filename)
        if array is not None:
            check_pixel_count(array.shape[1], array.shape[0])
            info_source = None
            if not is_raw_path(filename):
                info_source = Image.open(BytesIO(content))  # EXIF (Orientation reset below) + ICC
            image = to_image(array, info_source)
            if info_source is not None and info_source.getexif().get(0x0112, 1) != 1:
                exif = info_source.getexif()
                exif[0x0112] = 1
                image.info["exif"] = exif.tobytes()
            return image
    image = PhotoProcessor.from_bytes(content, filename).image
    if is_raw_path(filename):
        check_pixel_count(*image.size)
    return image


def _exif_bytes(image: Image.Image) -> bytes | None:
    exif = image.getexif()
    return exif.tobytes() if len(exif) else None


def _fit(mask: Image.Image | None, size: tuple[int, int]) -> Image.Image | None:
    if mask is None:
        return None
    mask = mask.convert("L")
    return mask if mask.size == size else mask.resize(size, Image.Resampling.NEAREST)


def expected_output_size(source_size: tuple[int, int], options: dict) -> tuple[int, int]:
    """The pixel size run_pipeline() will produce for `options` on a `source_size` image --
    without running it. Geometry (a 90° rotation, constrain crop), Crop and AI Upscale change the
    size; the other Lens corrections warp in place.
    The frontend labels the canvas with this when it's showing a reduced-size preview, so the
    dimensions shown are the export's, not the preview's."""
    width, height = geometry_output_size(source_size, options)
    crop_box = options.get("crop_box")
    if crop_box:
        left, top, right, bottom = (int(value) for value in crop_box)
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        width, height = right - left, bottom - top
    else:
        ratio = parse_aspect(options.get("aspect"))
        if ratio:
            target = ratio[0] / ratio[1]
            if width / height > target:
                width = int(height * target)
            else:
                height = int(width / target)
    if options.get("ai_upscale"):
        factor = int(options.get("ai_upscale_factor", 2))
        width, height = width * factor, height * factor
    return width, height


def check_upscale_output(size: tuple[int, int], factor: int) -> None:
    """Refuse an AI Upscale whose output would exceed MAX_UPSCALE_OUTPUT_PIXELS *before* spending
    GPU time on it (a 4x of a 24 MP photo is ~384 MP -- several GB of RAM for one image)."""
    output_pixels = size[0] * factor * size[1] * factor
    if output_pixels > MAX_UPSCALE_OUTPUT_PIXELS:
        raise ValueError(
            f"AI Upscale {factor}x would produce a {output_pixels / 1e6:.0f} MP image, over the "
            f"{MAX_UPSCALE_OUTPUT_PIXELS / 1e6:.0f} MP limit -- crop first or use 2x."
        )


def run_pipeline(
    image: Image.Image,
    options: dict,
    *,
    removal_mask: Image.Image | None = None,
    local_mask: Image.Image | None = None,
    layers: list[WorkspaceLayer] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    local_masks: list[Image.Image] | None = None,
) -> Image.Image:
    """Apply every step `options` asks for, in the one canonical order (see the module docstring),
    to a copy of `image` (never modified). `options` is validated by
    app.edit_schema.parse_options() first -- an unknown key or out-of-domain value raises
    OptionsError (a ValueError) before any work happens.

    Metadata: `image.info`'s EXIF and ICC profile are carried onto the result's .info (ICC is the
    synthesized sRGB profile once the source was converted to it), unless options["strip_metadata"].

    Flags used by the frontend's two-stage cache:
    - `stop_after="detail"`: return right after Detail, skipping Repair onward.
    - `resume_from_detail`: `image` is a previous stop_after="detail" result (already ICC-
      normalized and through Lens..Detail); start at Repair.
    - `preview_scale` (0-1]: `image` is a reduced-size proxy of the real photo, `scale` times its
      size. Pixel-unit parameters (texture/sharpen/skin-blur/repair radii, the dehaze patch, and
      an explicit crop_box, which is always given in full-resolution pixels) are scaled by it so
      the proxy looks like a downscaled copy of the full-resolution export would. Non-Local Means
      denoise and film grain are deliberately not scaled: the frontend renders the 1:1 zoom views
      at full resolution for judging those.
    - `bit_depth=16` (B4): run in PhotoProcessor's high-precision mode -- a float32 working copy,
      seeded from the input's attached 16-bit pixels (decode_source(high_bit=True)) or from its
      8-bit values -- and attach the float result to the returned image for a 16-bit export
      (app.export.encode_export). Off by default; the 8-bit path is unchanged byte for byte.

    `should_cancel`, when given, is polled between steps; once it returns True the run stops with
    PipelineCancelled (the backend's "latest wins" job model -- a step already running finishes).
    """
    options = parse_options(options)

    def checkpoint() -> None:
        if should_cancel is not None and should_cancel():
            raise PipelineCancelled("superseded by a newer request")

    scale = options.get("preview_scale", 1.0)
    processor = PhotoProcessor.from_image(image)
    processor.pixel_scale = scale
    high_bit = options.get("bit_depth", 8) == 16
    if high_bit:
        processor.enable_high_precision(attached(image))
    strip = options.get("strip_metadata", False)
    source_exif = _exif_bytes(image)
    source_icc = image.info.get("icc_profile")

    if not options.get("resume_from_detail"):
        # ICC color management -- a real conversion (littlecms) of the source's embedded profile
        # into the sRGB working space every later step assumes. A no-op for sRGB sources; for a
        # wider gamut (Adobe RGB, ProPhoto, a camera profile) it's the difference between correct
        # and oversaturated color. The output is re-tagged as sRGB rather than re-attaching the
        # original (now stale) profile. Malformed profile bytes fall back to "keep whatever was
        # embedded, unconverted" rather than failing the whole edit.
        if source_icc:
            try:
                processor.image = normalize_to_srgb(processor.image, source_icc)
                source_icc = srgb_icc_bytes()
            except ICC_TRANSFORM_ERRORS:
                pass

        # Geometry -- rotate/flip, Lens corrections, straighten, constrain crop (app.geometry):
        # first, before anything else touches pixel positions, and before Crop so the user frames
        # against corrected geometry. The frontend's Crop/Masks/Repair base image is built by the
        # same apply_geometry(), so a mask or crop box lands on the same pixels here.
        processor.apply_geometry(options)

        checkpoint()
        # Tone (+ preset, right after the Tone group -- a preset applies its own order-sensitive
        # exposure/contrast/saturation/curve/sharpen adjustments)
        processor.apply_brightness(options.get("brightness", 1.0))
        processor.apply_exposure(options.get("exposure", 1.0))
        processor.apply_exposure_ev(options.get("exposure_ev", 0.0))
        processor.apply_contrast(options.get("contrast", 1.0))
        processor.apply_highlights_shadows(options.get("highlights", 0.0), options.get("shadows", 0.0))
        processor.apply_whites_blacks(options.get("whites", 0.0), options.get("blacks", 0.0))
        if options.get("preset") and options["preset"] != "None":
            apply_preset(processor, options["preset"])

        checkpoint()
        # Color
        processor.apply_temperature_tint(options.get("temperature", 0.0), options.get("tint", 0.0))
        processor.apply_saturation(options.get("saturation", 1.0))
        processor.apply_vibrance(options.get("vibrance", 0.0))
        if options.get("hsl_bands"):
            processor.apply_hsl_mixer(options["hsl_bands"])
        if options.get("black_white"):
            # After the colour adjustments that shape its input, before Color Grading (which can
            # then tone the gray -- a split-toned B&W) and the curve.
            processor.apply_black_white({band: options.get(key, 0.0) for band, key in _BW_MIX_OPTIONS.items()})
        processor.apply_color_grading(
            shadows_hue=options.get("cg_shadows_hue", 0.0), shadows_intensity=options.get("cg_shadows_intensity", 0.0),
            midtones_hue=options.get("cg_midtones_hue", 0.0), midtones_intensity=options.get("cg_midtones_intensity", 0.0),
            highlights_hue=options.get("cg_highlights_hue", 0.0), highlights_intensity=options.get("cg_highlights_intensity", 0.0),
            balance=options.get("cg_balance", 0.0),
        )
        if options.get("curve"):
            processor.apply_curves(options["curve"])

        checkpoint()
        # Detail
        processor.apply_texture_clarity(options.get("texture", 0.0), options.get("clarity", 0.0))
        if options.get("dehaze"):
            checkpoint()
            with HEAVY_STEP_SEMAPHORE:
                processor.apply_dehaze(options["dehaze"])
        if options.get("denoise"):
            checkpoint()
            with HEAVY_STEP_SEMAPHORE:
                processor.apply_denoise(options["denoise"], options.get("denoise_color", 0))
        if options.get("sharpen"):
            processor.apply_sharpen()

        if options.get("stop_after") == "detail":
            return _with_metadata(processor.image, source_exif, source_icc, strip, processor.hdr)

    checkpoint()
    # Repair
    removal = _fit(removal_mask, processor.image.size)
    if removal is not None:
        processor.remove_objects(removal, radius=max(1.0, options.get("removal_radius", 3.0) * scale))

    # Layers -- bottom-to-top, each composited onto the working image in turn.
    if layers:
        workspace = LayerWorkspace(processor.image)
        for layer in layers:
            workspace.add(layer)
        processor.image = workspace.composite().convert("RGB")

    checkpoint()
    # Masks
    local = _fit(local_mask, processor.image.size)
    if local is not None:
        processor.apply_masked_adjustments(
            local,
            exposure=options.get("mask_exposure", 1.0),
            contrast=options.get("mask_contrast", 1.0),
            saturation=options.get("mask_saturation", 1.0),
            exposure_ev=options.get("mask_exposure_ev", 0.0),
        )
    # Per-selection adjustment sets (B8): each of `local_masks` with its own entry of
    # options["local_adjustments"], in order -- e.g. brighten a face and darken the sky in one edit.
    local_adjustments = options.get("local_adjustments") or []
    if len(local_adjustments) != len(local_masks or []):
        raise ValueError(f"local_adjustments has {len(local_adjustments)} entries but {len(local_masks or [])} local masks were sent")
    for mask, adjustment in zip(local_masks or [], local_adjustments):
        processor.apply_masked_adjustments(
            _fit(mask, processor.image.size),
            exposure_ev=adjustment["exposure_ev"],
            contrast=1.0 + adjustment["contrast"] / 100.0,
            saturation=max(0.0, 1.0 + adjustment["saturation"] / 100.0),
            temperature=adjustment["temperature"],
            tint=adjustment["tint"],
            clarity=adjustment["clarity"],
        )

    # Portrait
    if options.get("portrait_smoothing", 0.0) > 0:
        processor.image = smooth_skin(processor.image, options["portrait_smoothing"], blur_radius=_SKIN_BLUR_RADIUS * scale)

    checkpoint()
    # Crop -- an explicit crop_box (full-resolution pixels) takes priority over a named aspect.
    crop_box = options.get("crop_box")
    if crop_box:
        left, top, right, bottom = (int(round(value * scale)) for value in crop_box)
        processor.crop(left, top, max(right, left + 1), max(bottom, top + 1))
    else:
        ratio = parse_aspect(options.get("aspect"))
        if ratio:
            processor.fit_aspect_ratio(*ratio)

    # Effects -- creative finishing on the final (cropped) frame.
    processor.apply_post_crop_vignette(options.get("effects_vignette_amount", 0.0), options.get("effects_vignette_midpoint", 50.0))
    if options.get("effects_grain_amount"):
        processor.apply_film_grain(options["effects_grain_amount"])

    # AI Upscale -- the one deliberate exception to the CPU-first, no-ML design (see "Known
    # Issues" in docs/PROJECT_STATUS.md). Last, so nothing re-processes the enlarged result at the
    # original resolution. ai_upscale_available() is checked here too so a stray option on a
    # machine without requirements-gpu.txt fails with a clear error, and the output size is
    # checked before any inference runs.
    if options.get("ai_upscale"):
        checkpoint()
        from app.ai_upscale import ai_upscale_available

        if not ai_upscale_available():
            raise RuntimeError(
                "AI Upscale requires the optional 'torch' and 'realesrgan' packages -- "
                "install requirements-gpu.txt to enable this feature."
            )
        factor = int(options.get("ai_upscale_factor", 2))
        check_upscale_output(processor.image.size, factor)
        processor.apply_ai_upscale(factor)

    return _with_metadata(processor.image, source_exif, source_icc, strip, processor.hdr)


def _with_metadata(image: Image.Image, exif: bytes | None, icc: bytes | None, strip: bool, high_bit=None) -> Image.Image:
    if high_bit is not None:
        attach(image, high_bit)
    image.info.pop("exif", None)
    image.info.pop("icc_profile", None)
    if not strip:
        if exif:
            image.info["exif"] = exif
        if icc:
            image.info["icc_profile"] = icc
    return image
