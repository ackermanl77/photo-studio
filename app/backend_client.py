from __future__ import annotations

import json
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

from app.config import BACKEND_HOST, BACKEND_PORT

BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
_HEALTH_CHECK_TIMEOUT = 0.5
_STARTUP_TIMEOUT_SECONDS = 15


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _backend_is_healthy() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=_HEALTH_CHECK_TIMEOUT).ok
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner="Starting local processing backend...")
def ensure_backend_running() -> None:
    """Make sure app/backend.py is reachable at BACKEND_URL, starting it if needed.

    Cached with st.cache_resource so this runs at most once per Streamlit server process --
    every rerun after the first just returns immediately. When launched through app/desktop.py,
    the backend is already started and healthy by the time this runs, so the spawn below is a
    no-op there; it only fires for the standalone `streamlit run app/frontend_unified_app.py`
    workflow (never a frozen build -- desktop.py always starts the backend first).
    """
    if _backend_is_healthy():
        return

    subprocess.Popen(
        [sys.executable, "-m", "app.backend"],
        cwd=str(_project_root()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _backend_is_healthy():
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"Could not start the local processing backend at {BACKEND_URL}. "
        "Try running `python -m app.backend` manually to see the underlying error."
    )


REQUEST_TIMEOUT_S = 30
AI_UPSCALE_REQUEST_TIMEOUT_S = 300


def process_via_backend(
    image_bytes: bytes,
    filename: str,
    options: dict,
    local_mask_bytes: bytes | None = None,
    layers_bytes: bytes | None = None,
    removal_mask_bytes: bytes | None = None,
) -> Image.Image:
    """POST an image plus the accumulated tab options to backend.py's /process endpoint and
    return the resulting decoded image.

    Always requests PNG (lossless) regardless of the user's chosen export format -- the caller
    applies that format/quality afterwards on the returned image, so this internal round trip
    must not add its own compression artifacts on top.
    """
    request_options = dict(options)
    request_options["output_format"] = "PNG"

    files = {"image": (filename, image_bytes)}
    if local_mask_bytes:
        files["local_mask"] = ("mask.png", local_mask_bytes)
    if layers_bytes:
        files["layers"] = ("layers.json", layers_bytes)
    if removal_mask_bytes:
        files["removal_mask"] = ("removal_mask.png", removal_mask_bytes)

    # AI Upscale's first request loads torch and downloads its ~64 MB model before inferring --
    # measured at ~20 s on a fast connection in the packaged build, so the normal 30 s budget
    # would time out on a slower one. Every other request keeps the short timeout.
    timeout = AI_UPSCALE_REQUEST_TIMEOUT_S if request_options.get("ai_upscale") else REQUEST_TIMEOUT_S
    try:
        response = requests.post(
            f"{BACKEND_URL}/process",
            data={"options": json.dumps(request_options)},
            files=files,
            timeout=timeout,
        )
    except requests.RequestException as error:
        raise RuntimeError(f"Could not reach the local processing backend: {error}") from error

    if not response.ok:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        raise RuntimeError(f"Processing backend error: {message}")

    return Image.open(BytesIO(response.content))
