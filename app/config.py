from __future__ import annotations

import os
import sys
from pathlib import Path

MAX_UPLOAD_BYTES = int(os.getenv("PHOTO_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("PHOTO_MAX_IMAGE_PIXELS", "50000000"))
BACKEND_HOST = os.getenv("PHOTO_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("PHOTO_BACKEND_PORT", "8502"))

# Marks a /process request as the frontend's internal Stage 2 hop (resume_from_detail) rather than
# a user upload -- set by app/backend_client.py, read by app/backend.py (see INTERNAL_STAGE_MAX_BYTES).
INTERNAL_STAGE_HEADER = "X-Photo-Internal-Stage"
INTERNAL_STAGE_RESUME = "resume-from-detail"


def user_data_dir() -> Path:
    """Per-user writable folder for files the app creates at runtime (desktop child-process logs,
    downloaded AI Upscale model weights). Never the install folder: the packaged desktop build is
    installed under Program Files, which a normal user can't write to. A "PhotoStudio" folder
    under %LOCALAPPDATA% on Windows; elsewhere the same ~/.photo-studio folder the library catalog and presets already
    use. PHOTO_STUDIO_DATA_DIR overrides both. Created on first use."""
    override = os.getenv("PHOTO_STUDIO_DATA_DIR")
    if override:
        folder = Path(override)
    elif sys.platform == "win32" and os.getenv("LOCALAPPDATA"):
        folder = Path(os.environ["LOCALAPPDATA"]) / "PhotoStudio"
    else:
        folder = Path.home() / ".photo-studio"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
