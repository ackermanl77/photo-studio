from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

MAX_UPLOAD_BYTES = int(os.getenv("PHOTO_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("PHOTO_MAX_IMAGE_PIXELS", "50000000"))
# Upper bound on an AI Upscale result, checked before inference runs (app.pipeline). The pipeline
# input is still capped at MAX_IMAGE_PIXELS; this is the separate cap on what 2x/4x may produce --
# 200 MP is a 4x of a ~12.5 MP photo, or a 2x of a 50 MP one.
MAX_UPSCALE_OUTPUT_PIXELS = int(os.getenv("PHOTO_MAX_UPSCALE_OUTPUT_PIXELS", "200000000"))
BACKEND_HOST = os.getenv("PHOTO_BACKEND_HOST", "127.0.0.1")
# Fixed defaults for a plain `python -m app.backend` / `streamlit run` session. The desktop launcher
# picks free ports per launch instead (app.desktop) and hands them to both children through these
# same variables, so a second program holding 8502/8504 no longer stops the app from starting.
BACKEND_PORT = int(os.getenv("PHOTO_BACKEND_PORT", "8502"))
FRONTEND_PORT = int(os.getenv("PHOTO_FRONTEND_PORT", "8504"))

# Marks a /process request as the frontend's internal Stage 2 hop (resume_from_detail) rather than
# a user upload -- set by app/backend_client.py, read by app/backend.py (see INTERNAL_STAGE_MAX_BYTES).
INTERNAL_STAGE_HEADER = "X-Photo-Internal-Stage"
INTERNAL_STAGE_RESUME = "resume-from-detail"

# Per-launch shared secret between the frontend and the backend (R1 in the improvement report).
# When PHOTO_BACKEND_TOKEN is set in the backend's environment, /process rejects any request that
# doesn't carry it in TOKEN_HEADER -- so a web page open in a browser on the same machine can no
# longer make the backend burn CPU/GPU (a multipart POST is a CORS "simple request"; a custom header
# also forces a preflight the backend never answers). The desktop launcher and the frontend's
# on-demand backend start both generate one; a manually started backend without it stays open, as
# before, for scripting.
TOKEN_ENV = "PHOTO_BACKEND_TOKEN"
TOKEN_HEADER = "X-Photo-Token"

# "Latest wins" (B11): the frontend tags each interactive request with its Streamlit session and a
# monotonically increasing generation; the backend abandons an older generation of the same
# session/lane between pipeline steps once a newer one arrives (a fast slider drag used to queue
# every intermediate position's full render).
EDIT_SESSION_HEADER = "X-Edit-Session"
EDIT_GENERATION_HEADER = "X-Edit-Generation"
EDIT_LANE_HEADER = "X-Edit-Lane"
# Stage 1 results the backend keeps for Stage 2 to resume from by handle (P3) instead of the client
# re-uploading the whole lossless PNG (~76 MB at 24 MP).
STAGE_HANDLE_HEADER = "X-Photo-Stage-Handle"
STAGE_CACHE_ENTRIES = int(os.getenv("PHOTO_STAGE_CACHE_ENTRIES", "4"))
# Stage results are bounded independently by pixel count and decoded pixel storage. Defaults derive
# from the upload pixel limit (at most one 50 MP RGBA image at the default settings).
_STAGE_CACHE_PIXEL_LIMIT = max(0, MAX_IMAGE_PIXELS)
STAGE_CACHE_MAX_PIXELS = min(
    _STAGE_CACHE_PIXEL_LIMIT,
    max(0, int(os.getenv("PHOTO_STAGE_CACHE_MAX_PIXELS", str(_STAGE_CACHE_PIXEL_LIMIT)))),
)
_STAGE_CACHE_BYTE_LIMIT = _STAGE_CACHE_PIXEL_LIMIT * 4
STAGE_CACHE_MAX_BYTES = min(
    _STAGE_CACHE_BYTE_LIMIT,
    max(0, int(os.getenv("PHOTO_STAGE_CACHE_MAX_BYTES", str(_STAGE_CACHE_BYTE_LIMIT)))),
)
# How many CPU-heavy steps (Non-Local Means denoise, dehaze) may run at once across all requests.
MAX_HEAVY_JOBS = max(1, int(os.getenv("PHOTO_MAX_HEAVY_JOBS", "2")))


def backend_token() -> str:
    """The token this process expects (backend) or sends (frontend); empty = no auth. Read on every
    call, not at import, so the frontend can set it after spawning its own backend."""
    return os.getenv(TOKEN_ENV, "")


def user_data_dir() -> Path:
    """Per-user writable folder for files the app creates at runtime (desktop child-process logs,
    downloaded AI Upscale model weights). Never the install folder: the packaged desktop build is
    installed under Program Files, which a normal user can't write to. A "PhotoStudio" folder
    under %LOCALAPPDATA% on Windows; elsewhere ~/.photo-studio. PHOTO_STUDIO_DATA_DIR overrides
    both. Created on first use. Every per-user file lives here -- logs, AI Upscale weights, custom
    presets, the library catalog, batch temp files, diagnostics' write probe; see
    user_data_file() for files an older build left in ~/.photo-studio."""
    override = os.getenv("PHOTO_STUDIO_DATA_DIR")
    if override:
        folder = Path(override)
    elif sys.platform == "win32" and os.getenv("LOCALAPPDATA"):
        folder = Path(os.environ["LOCALAPPDATA"]) / "PhotoStudio"
    else:
        folder = Path.home() / ".photo-studio"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def legacy_data_dir() -> Path:
    """Where builds before the single-data-folder change kept custom presets and the library
    catalog on every platform (~/.photo-studio), even on Windows where logs and weights already
    went to %LOCALAPPDATA%\\PhotoStudio."""
    return Path.home() / ".photo-studio"


def user_data_file(name: str) -> Path:
    """Path of a per-user file inside user_data_dir(). One-time migration: if the file doesn't
    exist there yet but an older build left one under legacy_data_dir(), it is copied over first
    (copied, not moved -- the original stays as a backup), so upgrading doesn't silently "lose" a
    user's saved presets or catalog ratings."""
    target = user_data_dir() / name
    if not target.exists():
        legacy = legacy_data_dir() / name
        try:
            if legacy.is_file() and legacy.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy, target)
        except OSError:
            pass  # a failed migration must not stop the app -- it just starts with an empty file
    return target


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write `text` to `path` via a temporary sibling file + os.replace(), so a crash or a full
    disk mid-write leaves the previous file intact instead of a truncated one (a truncated
    presets.json used to make PresetStore.load() raise on every rerun of the Presets tab)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
            temp_file.write(text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
