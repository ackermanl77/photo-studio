from __future__ import annotations

import sqlite3
from contextlib import contextmanager
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


# Every function below takes two float arrays in [0, 1] -- base (the composite so far) and source
# (the layer being blended in) -- and returns the blended result, unclipped; composite() clips
# once at the end. Formulas match the W3C Compositing and Blending spec, which is also what every
# modern browser's CSS mix-blend-mode uses and what Photoshop's own separable/non-separable blend
# modes are defined by, so these match Photoshop's own rendering rather than an approximation of it.
def _multiply(b, s): return b * s
def _screen(b, s): return 1.0 - (1.0 - b) * (1.0 - s)
def _darken(b, s): return np.minimum(b, s)
def _lighten(b, s): return np.maximum(b, s)
def _linear_dodge(b, s): return np.minimum(1.0, b + s)
def _linear_burn(b, s): return np.maximum(0.0, b + s - 1.0)
def _difference(b, s): return np.abs(b - s)
def _exclusion(b, s): return b + s - 2.0 * b * s
def _subtract(b, s): return np.maximum(0.0, b - s)


def _color_dodge(b, s):
    return np.where(s >= 1.0, 1.0, np.minimum(1.0, b / np.maximum(1.0 - s, 1e-6)))


def _color_burn(b, s):
    return np.where(s <= 0.0, 0.0, 1.0 - np.minimum(1.0, (1.0 - b) / np.maximum(s, 1e-6)))


def _divide(b, s):
    return np.where(s <= 0.0, 1.0, np.minimum(1.0, b / np.maximum(s, 1e-6)))


def _overlay(b, s):
    return np.where(b <= 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))


def _hard_light(b, s):
    return _overlay(s, b)


def _soft_light(b, s):
    d = np.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, np.sqrt(b))
    return np.where(s <= 0.5, b - (1.0 - 2.0 * s) * b * (1.0 - b), b + (2.0 * s - 1.0) * (d - b))


def _vivid_light(b, s):
    return np.where(s <= 0.5, _color_burn(b, 2.0 * s), _color_dodge(b, 2.0 * (s - 0.5)))


def _linear_light(b, s):
    return np.where(s <= 0.5, _linear_burn(b, 2.0 * s), _linear_dodge(b, 2.0 * (s - 0.5)))


def _pin_light(b, s):
    return np.where(s <= 0.5, _darken(b, 2.0 * s), _lighten(b, 2.0 * (s - 0.5)))


def _hard_mix(b, s):
    return np.where(_vivid_light(b, s) < 0.5, 0.0, 1.0)


def _darker_color(b, s):
    pick_base = (_lum(b) <= _lum(s))[..., None]
    return np.where(pick_base, b, s)


def _lighter_color(b, s):
    pick_base = (_lum(b) >= _lum(s))[..., None]
    return np.where(pick_base, b, s)


# Non-separable modes (Hue/Saturation/Color/Luminosity) can't blend each channel independently --
# they need a per-pixel luminosity/saturation computed across all three channels first. Same
# algorithm the W3C Compositing spec defines (SetLum/SetSat/ClipColor), simplified to plain numpy
# array ops since it turns out to be linear per-channel once you don't insist on the spec's
# min/mid/max case split.
def _lum(c):
    return 0.3 * c[..., 0] + 0.59 * c[..., 1] + 0.11 * c[..., 2]


def _clip_color(c):
    lum = _lum(c)[..., None]
    minimum = c.min(axis=-1, keepdims=True)
    maximum = c.max(axis=-1, keepdims=True)
    c = np.where(minimum < 0.0, lum + (c - lum) * lum / np.maximum(lum - minimum, 1e-6), c)
    c = np.where(maximum > 1.0, lum + (c - lum) * (1.0 - lum) / np.maximum(maximum - lum, 1e-6), c)
    return c


def _set_lum(c, target_lum):
    return _clip_color(c + (target_lum - _lum(c))[..., None])


def _sat(c):
    return c.max(axis=-1) - c.min(axis=-1)


def _set_sat(c, target_sat):
    minimum = c.min(axis=-1, keepdims=True)
    maximum = c.max(axis=-1, keepdims=True)
    spread = maximum - minimum
    scale = np.where(spread > 1e-6, target_sat[..., None] / spread, 0.0)
    return (c - minimum) * scale


def _hue_mode(b, s): return _set_lum(_set_sat(s, _sat(b)), _lum(b))
def _saturation_mode(b, s): return _set_lum(_set_sat(b, _sat(s)), _lum(b))
def _color_mode(b, s): return _set_lum(s, _lum(b))
def _luminosity_mode(b, s): return _set_lum(b, _lum(s))


# Per-channel (separable) blend modes: each output channel only depends on that same channel of
# base and source. "normal" is handled separately in composite() since it just returns source.
_SEPARABLE_BLEND_MODES = {
    "darken": _darken, "multiply": _multiply, "color_burn": _color_burn, "linear_burn": _linear_burn,
    "darker_color": _darker_color,
    "lighten": _lighten, "screen": _screen, "color_dodge": _color_dodge, "linear_dodge": _linear_dodge,
    "lighter_color": _lighter_color,
    "overlay": _overlay, "soft_light": _soft_light, "hard_light": _hard_light,
    "vivid_light": _vivid_light, "linear_light": _linear_light, "pin_light": _pin_light, "hard_mix": _hard_mix,
    "difference": _difference, "exclusion": _exclusion, "subtract": _subtract, "divide": _divide,
}

# Non-separable (whole-pixel) blend modes: each output channel depends on all three of the
# pixel's channels (via a shared luminosity/saturation), not just its own.
_NON_SEPARABLE_BLEND_MODES = {
    "hue": _hue_mode, "saturation": _saturation_mode, "color": _color_mode, "luminosity": _luminosity_mode,
}

# Human-readable labels for the modes above, in the same grouped order Photoshop's own Blend Mode
# dropdown uses -- shared with the frontend so the UI never hand-maintains its own copy of "which
# modes exist" separately from what composite() actually implements.
BLEND_MODE_LABELS = {
    "normal": "Normal",
    "darken": "Darken", "multiply": "Multiply", "color_burn": "Color Burn", "linear_burn": "Linear Burn",
    "darker_color": "Darker Color",
    "lighten": "Lighten", "screen": "Screen", "color_dodge": "Color Dodge", "linear_dodge": "Linear Dodge (Add)",
    "lighter_color": "Lighter Color",
    "overlay": "Overlay", "soft_light": "Soft Light", "hard_light": "Hard Light",
    "vivid_light": "Vivid Light", "linear_light": "Linear Light", "pin_light": "Pin Light", "hard_mix": "Hard Mix",
    "difference": "Difference", "exclusion": "Exclusion", "subtract": "Subtract", "divide": "Divide",
    "hue": "Hue", "saturation": "Saturation", "color": "Color", "luminosity": "Luminosity",
}


class LayerStack:
    """Float-based compositing stack for non-destructive layer edits."""

    MODES = {"normal", *_SEPARABLE_BLEND_MODES, *_NON_SEPARABLE_BLEND_MODES}

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
            if layer.blend_mode == "normal":
                blended = source_rgb
            elif layer.blend_mode in _SEPARABLE_BLEND_MODES:
                blended = _SEPARABLE_BLEND_MODES[layer.blend_mode](base_rgb, source_rgb)
            else:
                blended = _NON_SEPARABLE_BLEND_MODES[layer.blend_mode](base_rgb, source_rgb)
            blended = np.clip(blended, 0.0, 1.0)
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

    @contextmanager
    def _connect(self):
        # sqlite3.Connection's own context manager (what every call site's `with self._connect()
        # as connection:` relies on) only commits or rolls back the current transaction on exit --
        # it does NOT close the connection. Returning a bare Connection here meant every call
        # opened a fresh one that was never closed: index_folder() (app/library.py) calls
        # upsert() once per scanned photo, so indexing a large library leaked one open file handle
        # per photo. Wrapping the same open/commit/rollback behavior in an explicit try/finally
        # that closes the connection fixes that without changing any call site.
        connection = sqlite3.connect(self.database)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

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
