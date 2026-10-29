from __future__ import annotations

import base64
import binascii
import hmac
import json
import threading
import time
import uuid
from collections import OrderedDict
from io import BytesIO

import cv2
from flask import Flask, g, jsonify, request, send_file
from PIL import Image
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from app.config import (
    BACKEND_HOST, BACKEND_PORT, EDIT_GENERATION_HEADER, EDIT_LANE_HEADER, EDIT_SESSION_HEADER, INTERNAL_STAGE_HEADER,
    INTERNAL_STAGE_RESUME, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, STAGE_CACHE_ENTRIES, STAGE_CACHE_MAX_BYTES,
    STAGE_CACHE_MAX_PIXELS, STAGE_HANDLE_HEADER, TOKEN_HEADER, backend_token,
)
from app.edit_schema import parse_options
from app.export import FLOAT_FORMATS, encode_export
from app.image_processor import PhotoProcessor
from app.layer_workspace import layers_from_json
from app.observability import ProcessingAudit, configure_logging, get_logger, log_audit
from app.pipeline import PipelineCancelled, ai_tools_for, decode_source, run_pipeline
from app.reliability import check_pixel_count, new_request_id

app = Flask(__name__)
# No handler at import (R2): serve() -- the backend process's entry point -- configures the file log;
# an in-process caller (tests, the frontend's preset thumbnails) never opens one.
backend_logger = get_logger("backend")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# INTERNAL_STAGE_HEADER (app/config.py) is set by app/backend_client.py::process_via_backend() on a
# `resume_from_detail` request (the frontend's Stage 2, see process_image()'s docstring). That
# request's "image" is Stage 1's own
# lossless PNG output, not the user's upload -- for a large photo (e.g. 45 MP) it's ~75+ MB, over
# MAX_UPLOAD_BYTES even though the original JPEG the user opened was well under it, so every edit
# on such a photo used to fail with "Upload exceeds 50 MB". The byte cap exists to bound what a user
# can upload; this internal hop instead gets a cap derived from the pixel limit (a worst-case
# uncompressed RGB PNG plus full-resolution local/removal masks, plus the normal budget for the
# layers payload), and the pixel limit itself is still enforced on it (see process_image()).
INTERNAL_STAGE_MAX_BYTES = MAX_UPLOAD_BYTES + 6 * MAX_IMAGE_PIXELS

# Every "the request itself is bad" failure -> JSON 400. OptionsError (app.edit_schema) is a
# ValueError. cv2.error: Non-Local Means denoise can raise it for a degenerate input (e.g. an image
# smaller than its fixed 21px search window).
CLIENT_ERRORS = (KeyError, ValueError, OSError, RuntimeError, json.JSONDecodeError, cv2.error, binascii.Error)


@app.before_request
def assign_request_id():
    request.request_id = request.headers.get("X-Request-ID", new_request_id())
    g.processing_started = time.perf_counter()
    expected = backend_token()
    if expected and request.path != "/health":
        # Checked before anything reads the body: an unauthenticated caller can't even make the
        # backend parse a large upload (R1).
        if not hmac.compare_digest(request.headers.get(TOKEN_HEADER, ""), expected):
            return jsonify({"error": "Missing or invalid backend token"}), 401
    if request.path == "/process" and request.headers.get(INTERNAL_STAGE_HEADER) == INTERNAL_STAGE_RESUME:
        request.max_content_length = INTERNAL_STAGE_MAX_BYTES


# --- "Latest wins" (B11) ----------------------------------------------------------------------------
# The newest generation seen per (edit session, lane). A request whose generation has been passed
# by a newer one from the same session and lane stops at the next pipeline step (PipelineCancelled
# -> 409) instead of finishing a render nobody will look at.
_LATEST_GENERATION: "OrderedDict[tuple[str, str], int]" = OrderedDict()
_LATEST_LOCK = threading.Lock()
_LATEST_MAX_KEYS = 512
# The frontend's background refine-on-pause render (app.refine_preview) and the interactive lanes
# that supersede it.
_REFINE_LANE = "refine"
_INTERACTIVE_LANES = ("detail", "finish", "full")


def _cancellation_check():
    session = request.headers.get(EDIT_SESSION_HEADER)
    try:
        generation = int(request.headers.get(EDIT_GENERATION_HEADER, ""))
    except ValueError:
        return None
    if not session:
        return None
    lane = request.headers.get(EDIT_LANE_HEADER, "default")
    key = (session, lane)
    with _LATEST_LOCK:
        if generation >= _LATEST_GENERATION.get(key, -1):
            _LATEST_GENERATION[key] = generation
            _LATEST_GENERATION.move_to_end(key)
        if lane in _INTERACTIVE_LANES:
            # A new interactive edit makes a pending refine-on-pause render obsolete (P1): the
            # user moved on, so that full-resolution job stops at its next step.
            refine_key = (session, _REFINE_LANE)
            if generation > _LATEST_GENERATION.get(refine_key, -1):
                _LATEST_GENERATION[refine_key] = generation
                _LATEST_GENERATION.move_to_end(refine_key)
            while len(_LATEST_GENERATION) > _LATEST_MAX_KEYS:
                _LATEST_GENERATION.popitem(last=False)

    def superseded() -> bool:
        with _LATEST_LOCK:
            return _LATEST_GENERATION.get(key, generation) > generation

    return superseded


# --- Stage 1 hand-off by handle (P3) ----------------------------------------------------------------
# A `stop_after="detail"` result is kept here (a few entries, LRU) and its handle returned in
# STAGE_HANDLE_HEADER; the frontend's Stage 2 then sends just the handle instead of re-uploading the
# lossless PNG. A miss (evicted, or the backend restarted) is a 410 and the client re-uploads.
_STAGE_CACHE: "OrderedDict[str, tuple[Image.Image, int, int]]" = OrderedDict()
_STAGE_CACHE_LOCK = threading.Lock()

_STAGE_BYTES_PER_PIXEL = {
    "1": 0,
    "L": 1,
    "LA": 2,
    "P": 1,
    "RGB": 3,
    "RGBX": 4,
    "RGBA": 4,
    "CMYK": 4,
    "I;16": 2,
    "I;16L": 2,
    "I;16B": 2,
    "I": 4,
    "F": 4,
}


def _copy_with_info(image: Image.Image) -> Image.Image:
    copy = image.copy()
    copy.info.update(image.info)
    return copy


def _stage_image_cost(image: Image.Image) -> tuple[int, int] | None:
    bytes_per_pixel = _STAGE_BYTES_PER_PIXEL.get(image.mode)
    if bytes_per_pixel is None:
        return None
    pixels = image.width * image.height
    byte_count = ((image.width + 7) // 8) * image.height if image.mode == "1" else pixels * bytes_per_pixel
    return pixels, byte_count


def _remember_stage(image: Image.Image) -> str | None:
    cost = _stage_image_cost(image)
    if cost is None:
        return None
    pixels, byte_count = cost
    if (
        STAGE_CACHE_ENTRIES <= 0
        or pixels > STAGE_CACHE_MAX_PIXELS
        or byte_count > STAGE_CACHE_MAX_BYTES
    ):
        return None

    handle = uuid.uuid4().hex
    with _STAGE_CACHE_LOCK:
        while _STAGE_CACHE:
            cached_pixels = sum(entry[1] for entry in _STAGE_CACHE.values())
            cached_bytes = sum(entry[2] for entry in _STAGE_CACHE.values())
            if (
                len(_STAGE_CACHE) < STAGE_CACHE_ENTRIES
                and cached_pixels + pixels <= STAGE_CACHE_MAX_PIXELS
                and cached_bytes + byte_count <= STAGE_CACHE_MAX_BYTES
            ):
                break
            _STAGE_CACHE.popitem(last=False)
        _STAGE_CACHE[handle] = (_copy_with_info(image), pixels, byte_count)
    return handle


def _recall_stage(handle: str) -> Image.Image | None:
    with _STAGE_CACHE_LOCK:
        entry = _STAGE_CACHE.get(handle)
        if entry is None:
            return None
        _STAGE_CACHE.move_to_end(handle)
        return _copy_with_info(entry[0])


@app.after_request
def attach_request_id(response):
    request_id = getattr(request, "request_id", new_request_id())
    response.headers["X-Request-ID"] = request_id
    if request.path == "/process":
        try:
            # A rejected (401) request's body is never parsed, not even for this audit line.
            if response.status_code == 401:
                source_name = "unauthorized"
            else:
                source_name = request.files.get("image").filename if request.files.get("image") else "unknown"
        except RequestEntityTooLarge:
            # An over-cap upload: touching request.files re-raises the 413 from inside this hook,
            # which turned the intended clean JSON 413 (request_too_large() below) into an HTML 500.
            source_name = "unknown"
        audit = ProcessingAudit(request_id, source_name or "unknown", "process", "ok" if response.status_code < 400 else "error", (time.perf_counter() - getattr(g, "processing_started", time.perf_counter())) * 1000)
        log_audit(backend_logger, audit)
    return response


def _open_mask(file_storage) -> Image.Image | None:
    if not file_storage:
        return None
    return Image.open(BytesIO(file_storage.read())).convert("L")


def _open_local_masks(file_storage) -> list[Image.Image] | None:
    """The "local_masks" upload: JSON {"masks": [base64 PNG, ...]} (one per local_adjustments entry)."""
    if not file_storage:
        return None
    payload = json.loads(file_storage.read())
    masks = payload.get("masks") if isinstance(payload, dict) else None
    if not isinstance(masks, list):
        raise ValueError("local_masks must be a JSON object with a 'masks' list")
    return [Image.open(BytesIO(base64.b64decode(mask))).convert("L") for mask in masks]


def process_image(image_file, options: dict, removal_mask=None, local_mask=None, layers_file=None, *, should_cancel=None, stage_image: Image.Image | None = None, local_masks_file=None) -> Image.Image:
    """Decode an uploaded file and run it through app.pipeline.run_pipeline() -- the one
    implementation of the step order (see that module's docstring), shared with the CLI and batch
    processing. `image_file`/masks/`layers_file` are Flask FileStorage-like objects (`.filename`,
    `.read()`); `layers_file` carries app.layer_workspace.layers_to_json()'s payload.

    `options` is validated first (app.edit_schema.parse_options): unknown keys, wrong types and
    out-of-domain values raise OptionsError (a ValueError -> 400) before anything is decoded.

    With `resume_from_detail`, `image_file` is a previous `stop_after="detail"` result (the
    frontend's two-stage cache -- see run_pipeline()), already RAW-decoded and ICC-normalized; it
    carries its EXIF in the PNG's eXIf chunk, so metadata survives both hops. That hop may be
    exempt from the upload byte cap (INTERNAL_STAGE_MAX_BYTES), so the exact pixel limit is the
    bound that still applies to it. `stage_image` is that same Stage 1 result recalled from the
    backend's own cache by handle (P3) -- then `image_file` may be None.

    A 16-bit export (`options["export"]["bit_depth"] == 16`) implies `bit_depth=16`: the source is
    decoded at full precision and the pipeline runs in high-precision mode (B4, app.high_bit).
    """
    options = parse_options(options)
    export_request = options.get("export") or {}
    if export_request.get("bit_depth") == 16 or export_request.get("format") in FLOAT_FORMATS:
        options["bit_depth"] = 16
    # A Display P3 export renders in a Display P3 working space (B14), from the original file.
    if (options.get("export") or {}).get("color_space") == "Display P3":
        options["working_space"] = "Display P3"
    high_bit = options.get("bit_depth") == 16
    working_space = options.get("working_space", "sRGB")
    if stage_image is not None:
        image = stage_image
    else:
        source_bytes = image_file.read()
        filename = image_file.filename or "image"
        if options.get("resume_from_detail"):
            image = PhotoProcessor.from_bytes(source_bytes, filename).image
            check_pixel_count(*image.size)
        else:
            image = decode_source(source_bytes, filename, high_bit=high_bit, working_space=working_space)
    layers = layers_from_json(layers_file.read()) if layers_file else None
    return run_pipeline(
        image, options, removal_mask=_open_mask(removal_mask), local_mask=_open_mask(local_mask), layers=layers,
        should_cancel=should_cancel, local_masks=_open_local_masks(local_masks_file),
    )


def _encode_response(result: Image.Image, options: dict):
    """`options["export"]` (app.export settings) -> the final delivery file, encoded here so an
    AI-upscaled result (possibly over the frontend process's own decode limit) never has to be
    decoded by the caller. Otherwise the older transport formats: PNG (lossless internal hand-off)
    or JPEG, both carrying EXIF and ICC unless strip_metadata."""
    export_settings = options.get("export")
    if export_settings:
        tools = [*export_settings.get("ai_tools", ()), *ai_tools_for(options)]
        content, extension, mime = encode_export(result, {**export_settings, "ai_tools": tools})
        return BytesIO(content), mime, f"edited{extension}"
    output_format = options.get("output_format", "PNG")
    max_width, max_height = options.get("max_width", 0), options.get("max_height", 0)
    if max_width > 0 and max_height > 0:
        result.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    keep_metadata = not options.get("strip_metadata", False)
    if output_format == "PNG" and options.get("bit_depth") == 16 and not (max_width > 0 and max_height > 0):
        # A high-precision run asked for a plain PNG response (the API/CLI, not the UI): a 16-bit PNG.
        content, _extension, mime = encode_export(result, {"format": "PNG", "bit_depth": 16, "metadata": "all" if keep_metadata else "none"})
        return BytesIO(content), mime, "edited.png"
    buffer = BytesIO()
    if output_format == "JPEG":
        save_options = {"format": "JPEG", "quality": options.get("quality", 92), "optimize": True}
    else:
        # compress_level=1, not optimize=True: a PNG response is only ever an internal lossless
        # hand-off, often re-uploaded straight back for Stage 2 -- optimize=True spent seconds per
        # edit on a large photo squeezing out bytes nobody keeps.
        save_options = {"format": "PNG", "compress_level": 1}
    # EXIF now rides along on the PNG hand-off too (Pillow writes an eXIf chunk) -- it used to be
    # dropped at Stage 1, so no export from the UI ever carried camera/capture metadata.
    if keep_metadata and result.info.get("exif"):
        save_options["exif"] = result.info["exif"]
    if keep_metadata and result.info.get("icc_profile"):
        save_options["icc_profile"] = result.info["icc_profile"]
    result.save(buffer, **save_options)
    buffer.seek(0)
    return buffer, ("image/jpeg" if output_format == "JPEG" else "image/png"), ("edited.jpg" if output_format == "JPEG" else "edited.png")


@app.get("/health")
def health():
    # "auth" tells a client whether this backend expects a token (no secret is revealed).
    return jsonify({"status": "ok", "service": "photo-processing-backend", "auth": bool(backend_token())})


@app.post("/ai_mask")
def ai_mask():
    """AI masks (F16, app.ai_masks): the uploaded image's Subject/Person/Sky mask as a grayscale PNG.
    Inference runs here, not in the Streamlit process (torch stays out of it, as for AI Upscale)."""
    from app.ai_masks import AI_MASK_KINDS, MODEL_KINDS, ai_masks_installed, make_ai_mask

    kind = request.form.get("kind", "")
    if kind not in AI_MASK_KINDS:
        return jsonify({"error": f"kind must be one of {', '.join(AI_MASK_KINDS)}"}), 400
    if kind in MODEL_KINDS and not ai_masks_installed():
        return jsonify({"error": "Subject and Person masks need the GPU pack (PyTorch)."}), 400
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        image = Image.open(BytesIO(image_file.read()))
        check_pixel_count(*image.size)
        mask = make_ai_mask(image.convert("RGB"), kind)
    except CLIENT_ERRORS as error:
        return jsonify({"error": str(error)}), 400
    buffer = BytesIO()
    mask.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


@app.post("/plugin_filter/configure")
def plugin_filter_configure():
    """Photoshop filter plug-ins (PS20, app.filter_plugins): show a plug-in's own dialog on the
    uploaded image and return its chosen settings as {"step": {...}}, for the recipe's
    plugin_filters. Blocks until the dialog closes; 409 when it was cancelled."""
    from app.filter_plugins import FilterCancelled, FilterPluginError, configure

    try:
        step = json.loads(request.form.get("step", ""))
    except ValueError:
        return jsonify({"error": "step must be JSON"}), 400
    if not isinstance(step, dict) or not isinstance(step.get("file"), str) or not str(step["file"]).lower().endswith(".8bf") \
            or not isinstance(step.get("entry"), str) or not step["entry"]:
        return jsonify({"error": 'step must be {"file": "...8bf", "entry": "...", "title": "..."}'}), 400
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        image = Image.open(BytesIO(image_file.read()))
        check_pixel_count(*image.size)
        configured = configure(image.convert("RGB"), {"file": step["file"], "entry": step["entry"],
                                                      "title": str(step.get("title") or "")[:120]})
    except FilterCancelled as error:
        return jsonify({"error": str(error)}), 409
    except (FilterPluginError, *CLIENT_ERRORS) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"step": configured})


@app.post("/process")
def process():
    try:
        options = parse_options(json.loads(request.form.get("options", "{}")))
        image_file = request.files.get("image")
        stage_image = None
        stage_handle = request.form.get("stage_handle")
        if image_file is None and stage_handle and options.get("resume_from_detail"):
            stage_image = _recall_stage(stage_handle)
            if stage_image is None:
                return jsonify({"error": "Stage handle expired -- re-upload the image", "stage_handle_expired": True}), 410
        elif image_file is None:
            raise ValueError("Missing 'image' file in the request")
        result = process_image(
            image_file,
            options,
            request.files.get("removal_mask"),
            request.files.get("local_mask"),
            request.files.get("layers"),
            should_cancel=_cancellation_check(),
            stage_image=stage_image,
            local_masks_file=request.files.get("local_masks"),
        )
        handle = _remember_stage(result) if options.get("stop_after") == "detail" else None
        buffer, mime, download_name = _encode_response(result, options)
        response = send_file(buffer, mimetype=mime, download_name=download_name)
        if handle:
            response.headers[STAGE_HANDLE_HEADER] = handle
        return response
    except PipelineCancelled:
        return jsonify({"error": "Superseded by a newer edit", "superseded": True}), 409
    except Image.DecompressionBombError as error:
        return jsonify({"error": f"Image exceeds the configured pixel limit: {error}"}), 413
    except CLIENT_ERRORS as error:
        return jsonify({"error": str(error)}), 400
    except MemoryError:
        return jsonify({"error": "Not enough memory to process this image -- try a smaller image or fewer heavy adjustments", "request_id": request.request_id}), 507


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"error": f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"}), 413


@app.errorhandler(Exception)
def unexpected_error(error):
    """Anything not handled above is still a JSON body (never Flask's HTML 500 page), carrying the
    request id so it can be matched against the audit log."""
    if isinstance(error, HTTPException):
        return jsonify({"error": error.description}), error.code
    backend_logger.exception("Unhandled error", extra={"request_id": getattr(request, "request_id", None)})
    return jsonify({"error": f"Internal error: {type(error).__name__}", "request_id": getattr(request, "request_id", None)}), 500


def serve(host: str = BACKEND_HOST, port: int = BACKEND_PORT) -> None:
    """Run the backend: on waitress (a production WSGI server, pure Python, fine on Windows) when it
    is installed, else Flask's threaded development server as before."""
    configure_logging("backend")
    try:
        from waitress import serve as waitress_serve
    except ImportError:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
    else:
        waitress_serve(app, host=host, port=port, threads=8)


if __name__ == "__main__":
    serve()
