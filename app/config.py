from __future__ import annotations

import os

MAX_UPLOAD_BYTES = int(os.getenv("PHOTO_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("PHOTO_MAX_IMAGE_PIXELS", "50000000"))
BACKEND_HOST = os.getenv("PHOTO_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("PHOTO_BACKEND_PORT", "8502"))
