from __future__ import annotations

import json
import time
from io import BytesIO

from flask import Flask, g, jsonify, request, send_file
from PIL import Image

from app.config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, BACKEND_HOST, BACKEND_PORT
from app.image_processor import PhotoProcessor
from app.layer_workspace import LayerWorkspace, WorkspaceLayer
from app.observability import ProcessingAudit, configure_json_logger, log_audit
from app.portrait_tools import smooth_skin
from app.presets import PRESETS, apply_preset
from app.raw_import import is_raw_path
from app.reliability import new_request_id, validate_image_bytes

app = Flask(__name__)
backend_logger = configure_json_logger("photo-studio-backend")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@app.before_request
def assign_request_id():
    request.request_id = request.headers.get("X-Request-ID", new_request_id())
    g.processing_started = time.perf_counter()


@app.after_request
def attach_request_id(response):
    request_id = getattr(request, "request_id", new_request_id())
    response.headers["X-Request-ID"] = request_id
    if request.path == "/process":
        source_name = request.files.get("image").filename if request.files.get("image") else "unknown"
        audit = ProcessingAudit(request_id, source_name or "unknown", "process", "ok" if response.status_code < 400 else "error", (time.perf_counter() - getattr(g, "processing_started", time.perf_counter())) * 1000)
        log_audit(backend_logger, audit)
    return response


def _number(options: dict, key: str, default: float) -> float:
    return float(options.get(key, default))


def _image_mask(file_storage, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(file_storage.stream).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask


def process_image(image_file, options: dict, removal_mask=None, local_mask=None, layer_file=None) -> Image.Image:
    """Mirrors the exact step order of the unified UI's tabs (Tone -> Color -> Detail -> Layers ->
    Masks -> Portrait -> Crop) so that routing the frontend's edits through this endpoint produces
    the same image as applying them in-process. Keeping this order in sync with
    app/frontend_unified_app.py matters: a mismatched order is exactly the class of bug documented
    under "Recently Fixed" in docs/PROJECT_STATUS.md (layers/masks silently producing a different
    result than what the UI showed).
    """
    source_bytes = image_file.read()
    filename = image_file.filename or "image"
    # PIL (via validate_image_bytes) cannot decode camera RAW -- PhotoProcessor.from_bytes() routes
    # RAW through rawpy and raises its own (already-tested) RuntimeError on failure, so RAW uploads
    # skip this PIL-based check instead of always failing it.
    if not is_raw_path(filename):
        validate_image_bytes(source_bytes, filename)
    processor = PhotoProcessor.from_bytes(source_bytes, filename)
    source_exif = processor.image.getexif().tobytes()
    source_icc = processor.image.info.get("icc_profile")

    # Tone
    processor.apply_brightness(_number(options, "brightness", 1.0))
    processor.apply_exposure(_number(options, "exposure", 1.0))
    processor.apply_contrast(_number(options, "contrast", 1.0))
    processor.apply_highlights_shadows(_number(options, "highlights", 0), _number(options, "shadows", 0))
    if options.get("preset") and options["preset"] != "None":
        if options["preset"] not in PRESETS:
            raise ValueError(f"Unknown preset: {options['preset']}")
        apply_preset(processor, options["preset"])

    # Color
    processor.apply_temperature_tint(_number(options, "temperature", 0), _number(options, "tint", 0))
    processor.apply_saturation(_number(options, "saturation", 1.0))
    curve = options.get("curve")
    if curve:
        processor.apply_curves([(int(point[0]), int(point[1])) for point in curve])

    # Detail
    processor.apply_texture_clarity(_number(options, "texture", 0), _number(options, "clarity", 0))
    if options.get("denoise", 0):
        processor.apply_denoise(int(options["denoise"]))
    if options.get("sharpen", False):
        processor.apply_sharpen()

    if removal_mask:
        processor.remove_objects(_image_mask(removal_mask, processor.image.size))

    # Layers
    if layer_file:
        overlay = Image.open(layer_file.stream).convert("RGBA")
        overlay_layer = WorkspaceLayer("Overlay", overlay, _number(options, "layer_opacity", 1.0), options.get("layer_blend", "normal"))
        processor.image = LayerWorkspace(processor.image).add(overlay_layer).composite().convert("RGB")

    # Masks
    if local_mask:
        processor.apply_masked_adjustments(
            _image_mask(local_mask, processor.image.size),
            exposure=_number(options, "mask_exposure", 1.0),
            contrast=_number(options, "mask_contrast", 1.0),
            saturation=_number(options, "mask_saturation", 1.0),
        )

    # Portrait
    portrait_smoothing = _number(options, "portrait_smoothing", 0.0)
    if portrait_smoothing > 0:
        processor.image = smooth_skin(processor.image, portrait_smoothing)

    # Crop -- an explicit crop_box (free-form, from the unified UI's Crop tab) takes priority over
    # the named aspect-ratio fit; the frontend only ever sends one of the two at a time.
    crop_box = options.get("crop_box")
    if crop_box:
        left, top, right, bottom = (int(value) for value in crop_box)
        processor.crop(left, top, right, bottom)
    else:
        aspect = options.get("aspect", "Original")
        if aspect != "Original":
            ratios = {"1:1": (1, 1), "4:5": (4, 5), "5:4": (5, 4), "3:2": (3, 2), "16:9": (16, 9), "9:16": (9, 16)}
            processor.fit_aspect_ratio(*ratios[aspect])

    if source_exif and not options.get("strip_metadata", False):
        processor.image.info["exif"] = source_exif
    if source_icc and not options.get("strip_metadata", False):
        processor.image.info["icc_profile"] = source_icc
    return processor.image


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "photo-processing-backend"})


@app.post("/process")
def process():
    try:
        options = json.loads(request.form.get("options", "{}"))
        image_file = request.files["image"]
        result = process_image(
            image_file,
            options,
            request.files.get("removal_mask"),
            request.files.get("local_mask"),
            request.files.get("layer"),
        )
        output_format = options.get("output_format", "PNG")
        mime = "image/jpeg" if output_format == "JPEG" else "image/png"
        buffer = BytesIO()
        max_width = int(options.get("max_width", 0))
        max_height = int(options.get("max_height", 0))
        if max_width > 0 and max_height > 0:
            result.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        if output_format == "JPEG":
            save_options = {"format": "JPEG", "quality": int(options.get("quality", 92)), "optimize": True}
            if result.info.get("exif") and not options.get("strip_metadata", False):
                save_options["exif"] = result.info["exif"]
            if result.info.get("icc_profile") and not options.get("strip_metadata", False):
                save_options["icc_profile"] = result.info["icc_profile"]
            result.save(buffer, **save_options)
        else:
            save_options = {"format": "PNG", "optimize": True}
            if result.info.get("icc_profile") and not options.get("strip_metadata", False):
                save_options["icc_profile"] = result.info["icc_profile"]
            result.save(buffer, **save_options)
        buffer.seek(0)
        return send_file(buffer, mimetype=mime, download_name="edited.jpg" if output_format == "JPEG" else "edited.png")
    except (KeyError, ValueError, OSError, RuntimeError, json.JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 400
    except Image.DecompressionBombError as error:
        return jsonify({"error": f"Image exceeds the configured pixel limit: {error}"}), 413


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"error": f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"}), 413


if __name__ == "__main__":
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=False, use_reloader=False)