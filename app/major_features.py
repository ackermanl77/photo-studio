from __future__ import annotations

import json
import sqlite3
import time
import uuid
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
            if "tags_json" not in columns:
                # Preserve legacy comma-delimited tags as-is; only new writes have an unambiguous
                # structured representation because commas may be part of a tag.
                connection.execute("ALTER TABLE photos ADD COLUMN tags_json TEXT")
            # Colour label (xmp:Label, F21) and culling scores (app.culling, F20; -1 = not scored),
            # added to older catalogs in place.
            for column, declaration in (("label", "TEXT DEFAULT ''"), ("focus", "REAL DEFAULT -1"),
                                        ("highlights", "REAL DEFAULT -1"), ("shadows", "REAL DEFAULT -1"),
                                        ("scored_mtime", "REAL DEFAULT 0"),
                                        # Perceptual hash for the duplicate finder (F29), hex; "" = not hashed.
                                        ("phash", "TEXT DEFAULT ''"), ("hash_mtime", "REAL DEFAULT 0")):
                if column not in columns:
                    connection.execute(f"ALTER TABLE photos ADD COLUMN {column} {declaration}")
            # Library organisation (F21): collections, saved searches and virtual copies.
            connection.execute("CREATE TABLE IF NOT EXISTS collections (name TEXT NOT NULL, path TEXT NOT NULL, "
                               "PRIMARY KEY (name, path))")
            connection.execute("CREATE TABLE IF NOT EXISTS saved_searches (name TEXT PRIMARY KEY, params TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS virtual_copies (id TEXT PRIMARY KEY, path TEXT NOT NULL, "
                               "name TEXT NOT NULL, recipe TEXT NOT NULL, created REAL NOT NULL)")

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
        normalized_tags = sorted({tag.strip() for tag in (tags or []) if tag.strip()})
        tag_text = ",".join(normalized_tags)
        tags_json = json.dumps(normalized_tags, ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO photos(path, rating, tags, tags_json, xmp_mtime) VALUES (?, ?, ?, ?, 0) "
                "ON CONFLICT(path) DO UPDATE SET rating=excluded.rating, tags=excluded.tags, tags_json=excluded.tags_json",
                (str(path), rating, tag_text, tags_json),
            )

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

    _COLUMNS = "path, rating, tags, tags_json, label, focus, highlights, shadows"

    @staticmethod
    def _entry(row) -> dict[str, object]:
        path, rating, tags, tags_json, label, focus, highlights, shadows = row
        try:
            structured_tags = json.loads(tags_json) if tags_json is not None else None
        except (TypeError, json.JSONDecodeError):
            structured_tags = None
        tag_list = (
            [tag for tag in structured_tags if isinstance(tag, str)]
            if isinstance(structured_tags, list)
            else tags.split(",") if tags else []
        )
        return {
            "path": path, "rating": rating, "tags": tag_list, "label": label or "",
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

    def get_hash(self, path: str | Path) -> tuple[int | None, float]:
        """(perceptual hash, the file's mtime when it was hashed) -- (None, 0) when not hashed (F29)."""
        with self._connect() as connection:
            row = connection.execute("SELECT phash, hash_mtime FROM photos WHERE path = ?", (str(path),)).fetchone()
        if row is None or not row[0]:
            return None, 0.0
        return int(row[0], 16), float(row[1])

    def set_hash(self, path: str | Path, value: int, mtime: float) -> None:
        self.ensure(path)
        with self._connect() as connection:
            connection.execute("UPDATE photos SET phash = ?, hash_mtime = ? WHERE path = ?", (f"{value:016x}", float(mtime), str(path)))

    def search(self, text: str = "", minimum_rating: int = 0, label: str | None = None, sort: str = "rating",
               collection: str | None = None) -> list[dict[str, object]]:
        """Photos whose path or tags contain `text`, rated at least `minimum_rating`. `label`: None =
        any, "" = no label, or one colour. `sort`: one of SEARCH_SORTS. `collection`: only its photos."""
        escaped_text = text.replace("!", "!!").replace("%", "!%").replace("_", "!_")
        pattern = f"%{escaped_text}%"
        where, params = (
            "rating >= ? AND (path LIKE ? ESCAPE '!' OR COALESCE(tags_json, tags) LIKE ? ESCAPE '!')",
            [minimum_rating, pattern, pattern],
        )
        if label is not None:
            where += " AND label = ?"
            params.append(label)
        if collection:
            where += " AND path IN (SELECT path FROM collections WHERE name = ?)"
            params.append(collection)
        order = SEARCH_SORTS.get(sort, SEARCH_SORTS["rating"])
        with self._connect() as connection:
            rows = connection.execute(f"SELECT {self._COLUMNS} FROM photos WHERE {where} ORDER BY {order}", params).fetchall()
        return [self._entry(row) for row in rows]

    # ---- collections (F21) ----
    def add_to_collection(self, name: str, path: str | Path) -> None:
        name = name.strip()
        if not name:
            raise ValueError("A collection needs a name.")
        with self._connect() as connection:
            connection.execute("INSERT OR IGNORE INTO collections(name, path) VALUES (?, ?)", (name, str(path)))

    def remove_from_collection(self, name: str, path: str | Path) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM collections WHERE name = ? AND path = ?", (name, str(path)))

    def collections(self) -> dict[str, int]:
        """Collection name -> number of photos, by name."""
        with self._connect() as connection:
            rows = connection.execute("SELECT name, COUNT(*) FROM collections GROUP BY name ORDER BY name COLLATE NOCASE").fetchall()
        return {name: count for name, count in rows}

    def collections_of(self, path: str | Path) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT name FROM collections WHERE path = ? ORDER BY name COLLATE NOCASE", (str(path),)).fetchall()
        return [row[0] for row in rows]

    def delete_collection(self, name: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM collections WHERE name = ?", (name,))

    # ---- saved searches (F21) ----
    SEARCH_KEYS = ("text", "minimum_rating", "label", "sort", "collection")

    def save_search(self, name: str, params: dict) -> None:
        """Store a search under `name`: the keyword arguments of search() (unknown keys dropped)."""
        name = name.strip()
        if not name:
            raise ValueError("A saved search needs a name.")
        clean = {key: params[key] for key in self.SEARCH_KEYS if key in params}
        with self._connect() as connection:
            connection.execute("INSERT INTO saved_searches(name, params) VALUES (?, ?) ON CONFLICT(name) DO UPDATE "
                               "SET params = excluded.params", (name, json.dumps(clean)))

    def saved_searches(self) -> dict[str, dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT name, params FROM saved_searches ORDER BY name COLLATE NOCASE").fetchall()
        out = {}
        for name, params in rows:
            try:
                data = json.loads(params)
            except ValueError:
                continue
            if isinstance(data, dict):
                out[name] = {key: data[key] for key in self.SEARCH_KEYS if key in data}
        return out

    def delete_search(self, name: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM saved_searches WHERE name = ?", (name,))

    # ---- virtual copies (F21): another recipe for the same file ----
    def add_virtual_copy(self, path: str | Path, name: str, recipe: dict) -> str:
        copy_id = uuid.uuid4().hex[:12]
        with self._connect() as connection:
            connection.execute("INSERT INTO virtual_copies(id, path, name, recipe, created) VALUES (?, ?, ?, ?, ?)",
                               (copy_id, str(path), name.strip() or "Virtual copy", json.dumps(recipe), time.time()))
        return copy_id

    def update_virtual_copy(self, copy_id: str, recipe: dict | None = None, name: str | None = None) -> None:
        with self._connect() as connection:
            if recipe is not None:
                connection.execute("UPDATE virtual_copies SET recipe = ? WHERE id = ?", (json.dumps(recipe), copy_id))
            if name is not None and name.strip():
                connection.execute("UPDATE virtual_copies SET name = ? WHERE id = ?", (name.strip(), copy_id))

    def delete_virtual_copy(self, copy_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM virtual_copies WHERE id = ?", (copy_id,))

    def virtual_copies(self, path: str | Path | None = None) -> list[dict[str, object]]:
        """Virtual copies (of `path`, or all), oldest first: {"id", "path", "name", "recipe"}."""
        query = "SELECT id, path, name, recipe FROM virtual_copies"
        params: tuple = ()
        if path is not None:
            query += " WHERE path = ?"
            params = (str(path),)
        with self._connect() as connection:
            rows = connection.execute(query + " ORDER BY created", params).fetchall()
        out = []
        for copy_id, copy_path, name, recipe in rows:
            try:
                data = json.loads(recipe)
            except ValueError:
                data = {}
            out.append({"id": copy_id, "path": copy_path, "name": name, "recipe": data if isinstance(data, dict) else {}})
        return out
