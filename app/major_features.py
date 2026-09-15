from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class Layer:
    name: str
    image: Image.Image
    opacity: float = 1.0
    blend_mode: str = "normal"
    mask: Image.Image | None = None
    visible: bool = True


class LayerStack:
    """Float-based compositing stack for non-destructive layer edits."""

    MODES = {"normal", "multiply", "screen"}

    def __init__(self, base: Image.Image):
        self.layers = [Layer("Background", base.convert("RGBA"))]

    def add(self, layer: Layer) -> "LayerStack":
        if layer.blend_mode not in self.MODES:
            raise ValueError(f"Unsupported blend mode: {layer.blend_mode}")
        if layer.image.size != self.layers[0].image.size:
            raise ValueError("Layer dimensions must match the background")
        self.layers.append(layer)
        return self

    def composite(self) -> Image.Image:
        result = np.asarray(self.layers[0].image, dtype=np.float32) / 255.0
        for layer in self.layers[1:]:
            if not layer.visible:
                continue
            source = np.asarray(layer.image.convert("RGBA"), dtype=np.float32) / 255.0
            alpha = source[:, :, 3] * max(0.0, min(1.0, layer.opacity))
            if layer.mask is not None:
                alpha *= np.asarray(layer.mask.convert("L"), dtype=np.float32) / 255.0
            base_rgb = result[:, :, :3]
            source_rgb = source[:, :, :3]
            if layer.blend_mode == "multiply":
                blended = base_rgb * source_rgb
            elif layer.blend_mode == "screen":
                blended = 1.0 - (1.0 - base_rgb) * (1.0 - source_rgb)
            else:
                blended = source_rgb
            result[:, :, :3] = base_rgb * (1.0 - alpha[:, :, None]) + blended * alpha[:, :, None]
        return Image.fromarray(np.clip(result * 255.0, 0, 255).astype(np.uint8), mode="RGBA")


def heal_patch(image: Image.Image, source_box: tuple[int, int, int, int], target_box: tuple[int, int, int, int]) -> Image.Image:
    """Copy a clean source patch over a target region with soft OpenCV blending."""
    source = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    sx1, sy1, sx2, sy2 = source_box
    tx1, ty1, tx2, ty2 = target_box
    width, height = tx2 - tx1, ty2 - ty1
    if width <= 0 or height <= 0 or sx2 - sx1 != width or sy2 - sy1 != height:
        raise ValueError("Source and target boxes must have equal positive dimensions")
    patch = source[sy1:sy2, sx1:sx2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    center = (tx1 + width // 2, ty1 + height // 2)
    repaired = cv2.seamlessClone(patch, source, mask, center, cv2.NORMAL_CLONE)
    return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB), mode="RGB")


def preview_image(image: Image.Image, max_size: tuple[int, int] = (1600, 1600)) -> Image.Image:
    preview = image.copy()
    preview.thumbnail(max_size, Image.Resampling.LANCZOS)
    return preview


class PhotoCatalog:
    """Small local SQLite catalog for ratings, tags, and searchable image paths."""

    def __init__(self, database: str | Path):
        self.database = str(database)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS photos (path TEXT PRIMARY KEY, rating INTEGER DEFAULT 0, tags TEXT DEFAULT '')")

    def _connect(self):
        return sqlite3.connect(self.database)

    def upsert(self, path: str | Path, rating: int = 0, tags: list[str] | None = None) -> None:
        rating = max(0, min(5, int(rating)))
        tag_text = ",".join(sorted({tag.strip() for tag in (tags or []) if tag.strip()}))
        with self._connect() as connection:
            connection.execute("INSERT INTO photos(path, rating, tags) VALUES (?, ?, ?) ON CONFLICT(path) DO UPDATE SET rating=excluded.rating, tags=excluded.tags", (str(path), rating, tag_text))

    def get(self, path: str | Path) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute("SELECT rating, tags FROM photos WHERE path = ?", (str(path),)).fetchone()
        if row is None:
            return {"path": str(path), "rating": 0, "tags": []}
        rating, tags = row
        return {"path": str(path), "rating": rating, "tags": tags.split(",") if tags else []}

    def search(self, text: str = "", minimum_rating: int = 0) -> list[dict[str, object]]:
        pattern = f"%{text}%"
        with self._connect() as connection:
            rows = connection.execute("SELECT path, rating, tags FROM photos WHERE rating >= ? AND (path LIKE ? OR tags LIKE ?) ORDER BY rating DESC, path", (minimum_rating, pattern, pattern)).fetchall()
        return [{"path": path, "rating": rating, "tags": tags.split(",") if tags else []} for path, rating, tags in rows]
