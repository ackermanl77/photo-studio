from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

def preview_image(image, max_size=(1600, 1600)):
    """Kept for existing callers; the one implementation is app.professional_tools.preview_image."""
    from app.professional_tools import preview_image as _preview_image

    return _preview_image(image, max_size)


@dataclass
class ProcessingReport:
    source_name: str
    cache_key: str
    cache_hit: bool
    elapsed_ms: float
    width: int
    height: int


def recipe_hash(options: dict) -> str:
    payload = json.dumps(options, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


class PreviewCache:
    def __init__(self, directory: str | Path, max_items: int = 100):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_items = max(1, int(max_items))

    def key(self, source_bytes: bytes, options: dict) -> str:
        source_hash = hashlib.sha256(source_bytes).hexdigest()[:20]
        return f"{source_hash}-{recipe_hash(options)}"

    def get(self, key: str) -> Image.Image | None:
        path = self.directory / f"{key}.jpg"
        if not path.exists():
            return None
        try:
            with Image.open(path) as image:
                return image.convert("RGB").copy()
        except OSError:
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, image: Image.Image, max_size: tuple[int, int] = (1600, 1600)) -> None:
        path = self.directory / f"{key}.jpg"
        preview_image(image, max_size).save(path, format="JPEG", quality=86, optimize=True)
        self._trim()

    def clear(self) -> None:
        for path in self.directory.glob("*.jpg"):
            path.unlink(missing_ok=True)

    def _trim(self) -> None:
        files = sorted(self.directory.glob("*.jpg"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in files[self.max_items :]:
            path.unlink(missing_ok=True)


def timed_report(source_name: str, key: str, started: float, image: Image.Image, cache_hit: bool) -> ProcessingReport:
    return ProcessingReport(source_name, key, cache_hit, (time.perf_counter() - started) * 1000, image.width, image.height)
