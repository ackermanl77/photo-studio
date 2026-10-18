from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

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


# Lightroom's / Bridge's own xmp:Label values, in their order (keys 6-9 in the Library, F21).
COLOR_LABELS = ("Red", "Yellow", "Green", "Blue", "Purple")
# Library sort orders (F20): unscored photos (focus -1) go last when sorting by focus.
SEARCH_SORTS = {
    "rating": "rating DESC, path",
    "sharpest": "focus < 0, focus DESC, path",
    "blurriest": "focus < 0, focus ASC, path",
    "name": "path",
}


class PhotoCatalog:
    """Small local SQLite catalog for ratings, tags, and searchable image paths."""

    def __init__(self, database: str | Path):
        self.database = str(database)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS photos (path TEXT PRIMARY KEY, rating INTEGER DEFAULT 0, tags TEXT DEFAULT '', xmp_mtime REAL DEFAULT 0)")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(photos)")}
            if "xmp_mtime" not in columns:
                # A catalog from before sidecar tracking: its photos' sidecars are "unknown" (-1), so
                # the next scan only records their current time instead of re-importing them over
                # ratings given in the app since. New rows are inserted with 0 explicitly.
                connection.execute("ALTER TABLE photos ADD COLUMN xmp_mtime REAL DEFAULT -1")
            # Colour label (xmp:Label, F21) and culling scores (app.culling, F20; -1 = not scored),
            # added to older catalogs in place.
            for column, declaration in (("label", "TEXT DEFAULT ''"), ("focus", "REAL DEFAULT -1"),
                                        ("highlights", "REAL DEFAULT -1"), ("shadows", "REAL DEFAULT -1"),
                                        ("scored_mtime", "REAL DEFAULT 0")):
                if column not in columns:
                    connection.execute(f"ALTER TABLE photos ADD COLUMN {column} {declaration}")

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
            connection.execute("INSERT INTO photos(path, rating, tags, xmp_mtime) VALUES (?, ?, ?, 0) ON CONFLICT(path) DO UPDATE SET rating=excluded.rating, tags=excluded.tags", (str(path), rating, tag_text))

    def ensure(self, path: str | Path) -> bool:
        """Add `path` unrated if it isn't catalogued yet -- a rescan must not reset the ratings and
        tags already given (index_folder() used upsert(), which overwrote them with 0 / none).
        True when it was added just now."""
        with self._connect() as connection:
            cursor = connection.execute("INSERT OR IGNORE INTO photos(path, rating, tags, xmp_mtime) VALUES (?, 0, '', 0)", (str(path),))
            return cursor.rowcount > 0

    def xmp_mtime(self, path: str | Path) -> float | None:
        """Modification time of the photo's XMP sidecar when it was last read or written (0: never,
        -1: unknown, a catalog from before this was tracked); None when the photo isn't catalogued."""
        with self._connect() as connection:
            row = connection.execute("SELECT xmp_mtime FROM photos WHERE path = ?", (str(path),)).fetchone()
        return None if row is None else float(row[0])

    def set_xmp_mtime(self, path: str | Path, mtime: float) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE photos SET xmp_mtime = ? WHERE path = ?", (float(mtime), str(path)))

    _COLUMNS = "path, rating, tags, label, focus, highlights, shadows"

    @staticmethod
    def _entry(row) -> dict[str, object]:
        path, rating, tags, label, focus, highlights, shadows = row
        return {
            "path": path, "rating": rating, "tags": tags.split(",") if tags else [], "label": label or "",
            "focus": None if focus is None or focus < 0 else focus,
            "highlights": None if highlights is None or highlights < 0 else highlights,
            "shadows": None if shadows is None or shadows < 0 else shadows,
        }

    def get(self, path: str | Path) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(f"SELECT {self._COLUMNS} FROM photos WHERE path = ?", (str(path),)).fetchone()
        if row is None:
            return {"path": str(path), "rating": 0, "tags": [], "label": "", "focus": None, "highlights": None, "shadows": None}
        return self._entry(row)

    def set_label(self, path: str | Path, label: str) -> None:
        """The colour label (one of COLOR_LABELS, or "" for none)."""
        if label not in ("", *COLOR_LABELS):
            raise ValueError(f"Unknown colour label: {label!r}")
        with self._connect() as connection:
            connection.execute("UPDATE photos SET label = ? WHERE path = ?", (label, str(path)))

    def scored_mtime(self, path: str | Path) -> float | None:
        with self._connect() as connection:
            row = connection.execute("SELECT scored_mtime FROM photos WHERE path = ?", (str(path),)).fetchone()
        return None if row is None else float(row[0])

    def set_scores(self, path: str | Path, scores: dict, mtime: float) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE photos SET focus = ?, highlights = ?, shadows = ?, scored_mtime = ? WHERE path = ?",
                (float(scores["focus"]), float(scores["highlights"]), float(scores["shadows"]), float(mtime), str(path)),
            )

    def search(self, text: str = "", minimum_rating: int = 0, label: str | None = None, sort: str = "rating") -> list[dict[str, object]]:
        """Photos whose path or tags contain `text`, rated at least `minimum_rating`. `label`: None =
        any, "" = no label, or one colour. `sort`: one of SEARCH_SORTS."""
        pattern = f"%{text}%"
        where, params = "rating >= ? AND (path LIKE ? OR tags LIKE ?)", [minimum_rating, pattern, pattern]
        if label is not None:
            where += " AND label = ?"
            params.append(label)
        order = SEARCH_SORTS.get(sort, SEARCH_SORTS["rating"])
        with self._connect() as connection:
            rows = connection.execute(f"SELECT {self._COLUMNS} FROM photos WHERE {where} ORDER BY {order}", params).fetchall()
        return [self._entry(row) for row in rows]
