from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from app.performance import PreviewCache, timed_report


def cached_preview(
    source_bytes: bytes,
    source_name: str,
    options: dict,
    processor,
    cache_directory: str | Path,
) -> tuple[Image.Image, object]:
    cache = PreviewCache(cache_directory)
    key = cache.key(source_bytes, options)
    started = time.perf_counter()
    cached = cache.get(key)
    if cached is not None:
        return cached, timed_report(source_name, key, started, cached, True)
    image = processor()
    cache.put(key, image)
    preview = cache.get(key) or image
    return preview, timed_report(source_name, key, started, preview, False)
