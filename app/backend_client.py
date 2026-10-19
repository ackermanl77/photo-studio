from __future__ import annotations

import contextvars
import itertools
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

from app.config import (
    BACKEND_HOST, BACKEND_PORT, EDIT_GENERATION_HEADER, EDIT_LANE_HEADER, EDIT_SESSION_HEADER, INTERNAL_STAGE_HEADER,
    INTERNAL_STAGE_RESUME, STAGE_HANDLE_HEADER, TOKEN_ENV, TOKEN_HEADER, backend_token, user_data_dir,
)

BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
_HEALTH_CHECK_TIMEOUT = 0.5
_STARTUP_TIMEOUT_SECONDS = 15


class SupersededError(RuntimeError):
    """The backend abandoned this request because a newer edit from the same session arrived
    ("latest wins", see app.backend). The Streamlit run that sent it is already obsolete."""


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _backend_is_healthy() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=_HEALTH_CHECK_TIMEOUT).ok
    except requests.RequestException:
        return False


def _backend_command() -> list[str]:
    """How to start the backend from this process: the packaged exe's own `--backend-child` mode in
    a frozen build (there is no `python -m` there), `python -m app.backend` otherwise."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--backend-child"]
    return [sys.executable, "-m", "app.backend"]


def backend_log_path() -> Path:
    folder = user_data_dir() / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "backend-standalone.log"


@st.cache_resource(show_spinner="Starting local processing backend...")
def ensure_backend_running() -> None:
    """Make sure app/backend.py is reachable at BACKEND_URL, starting it if needed.

    Cached with st.cache_resource so this runs at most once per Streamlit server process --
    every rerun after the first just returns immediately. _post_process() clears that cache and
    calls this again when the backend stops answering (crash, out-of-memory kill), so a dead backend
    is restarted once instead of every edit failing until Streamlit restarts. When launched through
    app/desktop.py the backend is already healthy, so the spawn below only fires for the standalone
    `streamlit run app/frontend_unified_app.py` workflow or after a crash.

    The child gets a per-launch token (app.config.TOKEN_ENV) unless this process already has one
    (the desktop launcher passes its own), and its output goes to logs/backend-standalone.log under
    the user data folder instead of being discarded -- the cause of a crash used to be lost.
    """
    if _backend_is_healthy():
        return

    if not backend_token():
        os.environ[TOKEN_ENV] = secrets.token_urlsafe(32)
    log_file = backend_log_path().open("a", encoding="utf-8")
    subprocess.Popen(
        _backend_command(),
        cwd=str(_project_root() if not getattr(sys, "frozen", False) else user_data_dir()),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=dict(os.environ),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _backend_is_healthy():
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"Could not start the local processing backend at {BACKEND_URL}. "
        f"See {backend_log_path()} for the underlying error."
    )


REQUEST_TIMEOUT_S = 30
AI_UPSCALE_REQUEST_TIMEOUT_S = 300
# A deliberate, user-initiated full-resolution export (with or without AI Upscale): a 45 MP photo
# with heavy Detail settings measured ~45 s for Stage 1 alone, over the interactive 30 s budget.
EXPORT_REQUEST_TIMEOUT_S = 900

# The edit session the current Streamlit script run belongs to (bind_edit_session(), called by the
# frontend at the top of every run). Requests sent from that thread carry it plus a generation from
# _GENERATIONS so the backend can drop superseded ones. Exports run on another thread (a
# download_button callable), never see a session, and so are never cancelled.
_EDIT_SESSION: contextvars.ContextVar[str | None] = contextvars.ContextVar("photo_edit_session", default=None)
_GENERATIONS = itertools.count(1)
_GENERATIONS_LOCK = threading.Lock()


def bind_edit_session(session_id: str | None) -> None:
    _EDIT_SESSION.set(session_id)


def _session_headers(lane: str) -> dict:
    session = _EDIT_SESSION.get()
    if not session:
        return {}
    with _GENERATIONS_LOCK:
        generation = next(_GENERATIONS)
    return {EDIT_SESSION_HEADER: session, EDIT_GENERATION_HEADER: str(generation), EDIT_LANE_HEADER: lane}


REFINE_LANE = "refine"
# A full-resolution refine-on-pause render (app.refine_preview) is a background job: generous timeout.
REFINE_REQUEST_TIMEOUT_S = 300


def _lane(request_options: dict) -> str:
    if request_options.get("export"):
        return "export"
    if request_options.get("stop_after") == "detail":
        return "detail"
    return "finish" if request_options.get("resume_from_detail") else "full"


def _send(files: dict, data: dict, headers: dict, timeout) -> requests.Response:
    return requests.post(f"{BACKEND_URL}/process", data=data, files=files, headers=headers, timeout=timeout)


def _post_process(
    image_bytes, filename, request_options, local_mask_bytes, layers_bytes, removal_mask_bytes, timeout,
    stage_handle: str | None = None, local_masks_bytes: bytes | None = None, lane: str | None = None,
) -> requests.Response:
    files = {}
    if local_masks_bytes:
        files["local_masks"] = ("local_masks.json", local_masks_bytes)
    if local_mask_bytes:
        files["local_mask"] = ("mask.png", local_mask_bytes)
    if layers_bytes:
        files["layers"] = ("layers.json", layers_bytes)
    if removal_mask_bytes:
        files["removal_mask"] = ("removal_mask.png", removal_mask_bytes)
    data = {"options": json.dumps(request_options)}
    # A resume_from_detail request re-uploads the backend's own Stage 1 PNG, not the user's file --
    # the header tells app/backend.py to apply its internal-stage size cap instead of the user
    # upload cap (see INTERNAL_STAGE_MAX_BYTES there for why).
    headers = _session_headers(lane or _lane(request_options))
    if backend_token():
        headers[TOKEN_HEADER] = backend_token()
    if request_options.get("resume_from_detail"):
        headers[INTERNAL_STAGE_HEADER] = INTERNAL_STAGE_RESUME

    def attempt(with_handle: bool) -> requests.Response:
        request_files = dict(files)
        request_data = dict(data)
        if with_handle:
            request_data["stage_handle"] = stage_handle
        else:
            request_files["image"] = (filename, image_bytes)
        return _send(request_files, request_data, headers, timeout)

    def attempt_with_restart(with_handle: bool) -> requests.Response:
        try:
            return attempt(with_handle)
        except requests.ConnectionError as error:
            # Nothing is listening any more (the backend crashed or was killed): restart it once and
            # retry, instead of failing every edit until Streamlit itself restarts.
            try:
                ensure_backend_running.clear()
                ensure_backend_running()
            except RuntimeError:
                raise RuntimeError(f"Could not reach the local processing backend: {error}") from error
            try:
                return attempt(False)
            except requests.RequestException as retry_error:
                raise RuntimeError(f"Could not reach the local processing backend: {retry_error}") from retry_error
        except requests.RequestException as error:
            raise RuntimeError(f"Could not reach the local processing backend: {error}") from error

    response = attempt_with_restart(bool(stage_handle))
    status = getattr(response, "status_code", 200)
    if stage_handle and status == 410:
        # The backend no longer holds that Stage 1 result (evicted, or restarted): upload it.
        response = attempt_with_restart(False)
        status = getattr(response, "status_code", 200)

    if status == 409:
        raise SupersededError("Superseded by a newer edit")
    if not response.ok:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        raise RuntimeError(f"Processing backend error: {message}")
    return response


def ai_mask_via_backend(image_bytes: bytes, kind: str, timeout: float = 600) -> bytes:
    """POST /ai_mask (F16): the Subject/Person/Sky mask of `image_bytes` as grayscale PNG bytes. The
    first Subject/Person mask may download the model's weights, hence the generous timeout."""
    headers = {TOKEN_HEADER: backend_token()} if backend_token() else {}
    try:
        response = requests.post(f"{BACKEND_URL}/ai_mask", data={"kind": kind}, files={"image": ("image.png", image_bytes)},
                                 headers=headers, timeout=timeout)
    except requests.RequestException as error:
        raise RuntimeError(f"Could not reach the local processing backend: {error}") from error
    if not response.ok:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        raise RuntimeError(message)
    return response.content


def export_via_backend(
    image_bytes: bytes,
    filename: str,
    options: dict,
    export_settings: dict,
    local_mask_bytes: bytes | None = None,
    layers_bytes: bytes | None = None,
    removal_mask_bytes: bytes | None = None,
    local_masks_bytes: bytes | None = None,
) -> bytes:
    """Like process_via_backend(), but the backend also encodes the final delivery file
    (app.export.encode_export with `export_settings`) and this returns those file bytes undecoded.
    Used for export-time work the interactive preview never does: the full-resolution render of a
    photo previewed through a proxy, and AI Upscale -- whose 2x/4x output can exceed the pixel
    limit this (Streamlit) process decodes images under, which used to crash the page with an
    uncaught DecompressionBombError. Generous timeout: this is a deliberate, user-initiated job."""
    request_options = {**options, "export": export_settings}
    return _post_process(image_bytes, filename, request_options, local_mask_bytes, layers_bytes, removal_mask_bytes, EXPORT_REQUEST_TIMEOUT_S, local_masks_bytes=local_masks_bytes).content


def process_via_backend(
    image_bytes: bytes,
    filename: str,
    options: dict,
    local_mask_bytes: bytes | None = None,
    layers_bytes: bytes | None = None,
    removal_mask_bytes: bytes | None = None,
    stage_handle: str | None = None,
    local_masks_bytes: bytes | None = None,
    *,
    lane: str | None = None,
    timeout: float | None = None,
) -> Image.Image:
    """POST an image plus the accumulated tab options to backend.py's /process endpoint and
    return the resulting decoded image.

    Always requests PNG (lossless) regardless of the user's chosen export format -- the caller
    applies that format/quality afterwards on the returned image, so this internal round trip
    must not add its own compression artifacts on top.

    Stage 1 (`stop_after="detail"`) results come back with the backend's handle for them in
    `image.info["stage_handle"]` and the undecoded PNG in `image.info["response_png"]` (so the
    frontend caches those bytes as-is instead of re-encoding). Passing that handle as
    `stage_handle` on the Stage 2 request makes it send the handle instead of the image; the image
    bytes are only uploaded if the backend no longer has it (P3 in the improvement report).

    `lane`/`timeout` override the defaults -- the refine-on-pause render uses REFINE_LANE, which
    any newer interactive request of the same session supersedes (app.backend), and a longer timeout.
    """
    request_options = dict(options)
    request_options["output_format"] = "PNG"

    # AI Upscale's first request loads torch and downloads its ~64 MB model before inferring --
    # measured at ~20 s on a fast connection in the packaged build, so the normal 30 s budget
    # would time out on a slower one. Every other request keeps the short timeout.
    if timeout is None:
        timeout = AI_UPSCALE_REQUEST_TIMEOUT_S if request_options.get("ai_upscale") else REQUEST_TIMEOUT_S
    response = _post_process(image_bytes, filename, request_options, local_mask_bytes, layers_bytes, removal_mask_bytes, timeout, stage_handle, local_masks_bytes, lane)
    content = response.content
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except Image.DecompressionBombError as error:
        # Not a RuntimeError, so the frontend's handler didn't catch it and the page showed a
        # traceback (e.g. a 4x AI Upscale of any photo over ~6 MP). Surface it as a clean error.
        raise RuntimeError(f"The processed image is too large to display here: {error}") from error
    handle = (getattr(response, "headers", None) or {}).get(STAGE_HANDLE_HEADER)
    if handle:
        image.info["stage_handle"] = handle
        image.info["response_png"] = content
    return image
