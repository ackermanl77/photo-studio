"""Recent files (Photoshop's Home screen, "Recent"): the photos and projects opened lately, newest
first, each with a thumbnail -- offered on the start screen and in the open bar.

Kept in user_data_dir()/recent/: index.json, a JPEG thumbnail per entry, and -- for a file that came
in through the browser's uploader, which hands over bytes but no path -- a copy of the file. A file
opened from a path (the Library, the desktop launcher) is remembered by that path instead. The same
file opened again (same bytes) moves to the top instead of being listed twice. At most MAX_ENTRIES
are kept, copies over MAX_COPY_BYTES aren't (the file just isn't listed), and copies past
MAX_TOTAL_COPY_BYTES drop the oldest entries. Streamlit-free; failures to save are silent (the
list is a convenience), failures to open raise OSError for the caller to report.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from PIL import Image

MAX_ENTRIES = 8
MAX_COPY_BYTES = 120 * 1024 * 1024
MAX_TOTAL_COPY_BYTES = 600 * 1024 * 1024
THUMBNAIL_SIZE = (360, 240)
KINDS = ("photo", "project")


def recent_dir() -> Path:
    from app.config import user_data_dir

    folder = user_data_dir() / "recent"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _index_path() -> Path:
    return recent_dir() / "index.json"


def _valid(entry) -> bool:
    return (isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry.get("kind") in KINDS
            and isinstance(entry.get("name"), str) and bool(entry.get("copy") or entry.get("path")))


def load() -> list[dict]:
    """The recent files, newest first (entries whose file is gone are left out)."""
    try:
        entries = json.loads(_index_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if _valid(entry) and _source_path(entry).is_file()]


def _save(entries: list[dict]) -> None:
    from app.config import atomic_write_text

    atomic_write_text(_index_path(), json.dumps(entries, indent=2, ensure_ascii=False))


def _source_path(entry: dict) -> Path:
    return recent_dir() / entry["copy"] if entry.get("copy") else Path(entry["path"])


def thumbnail_path(entry: dict) -> Path | None:
    path = recent_dir() / entry["thumb"] if entry.get("thumb") else None
    return path if path is not None and path.is_file() else None


def read_bytes(entry: dict) -> bytes:
    """The remembered file's bytes (OSError when it was moved or deleted)."""
    return _source_path(entry).read_bytes()


def _write_thumbnail(entry_id: str, image: Image.Image | None) -> str | None:
    if image is None:
        return None
    name = f"{entry_id}-thumb.jpg"  # never the copy's own name (an uploaded .jpg)
    try:
        thumb = image.convert("RGB")
        thumb.thumbnail(THUMBNAIL_SIZE)
        thumb.save(recent_dir() / name, format="JPEG", quality=85)
    except (OSError, ValueError):
        return None
    return name


def _delete_files(entry: dict) -> None:
    for name in (entry.get("copy"), entry.get("thumb")):
        if name:
            (recent_dir() / name).unlink(missing_ok=True)


def remember(name: str, data: bytes, kind: str = "photo", thumbnail: Image.Image | None = None, path: str | None = None) -> str | None:
    """List a file just opened at the top; returns its entry id (None when it isn't kept)."""
    if kind not in KINDS or (path is None and len(data) > MAX_COPY_BYTES):
        return None
    digest = hashlib.sha256(data).hexdigest()
    try:
        entries = load()
        previous = next((entry for entry in entries if entry.get("digest") == digest and entry["kind"] == kind), None)
        if previous is not None:
            entries.remove(previous)
            entry = {**previous, "name": name, "opened": time.time()}
            if path is not None and not previous.get("path"):
                _delete_files({"copy": previous.get("copy")})
                entry.update(path=path, copy=None)
            if thumbnail is not None:
                entry["thumb"] = _write_thumbnail(entry["id"], thumbnail) or entry.get("thumb")
        else:
            entry_id = uuid.uuid4().hex
            copy = None
            if path is None:
                copy = f"{entry_id}{Path(name).suffix.lower() or '.bin'}"
                (recent_dir() / copy).write_bytes(data)
            entry = {"id": entry_id, "name": name, "kind": kind, "digest": digest, "opened": time.time(), "size": len(data),
                     "path": path, "copy": copy, "thumb": _write_thumbnail(entry_id, thumbnail)}
        entries.insert(0, entry)
        kept, copied = [], 0
        for item in entries:
            copied += item.get("size", 0) if item.get("copy") else 0
            if len(kept) < MAX_ENTRIES and (copied <= MAX_TOTAL_COPY_BYTES or item is entry):
                kept.append(item)
            else:
                _delete_files(item)
        _save(kept)
        return entry["id"]
    except OSError:
        return None


def set_thumbnail(entry_id: str, image: Image.Image) -> None:
    """Give an entry its thumbnail later (a project's, once its photo is decoded)."""
    try:
        entries = load()
        for entry in entries:
            if entry["id"] == entry_id:
                entry["thumb"] = _write_thumbnail(entry_id, image) or entry.get("thumb")
        _save(entries)
    except OSError:
        pass


def forget(entry_id: str) -> None:
    try:
        entries = load()
        for entry in [entry for entry in entries if entry["id"] == entry_id]:
            entries.remove(entry)
            _delete_files(entry)
        _save(entries)
    except OSError:
        pass


def clear() -> None:
    try:
        for entry in load():
            _delete_files(entry)
        _save([])
    except OSError:
        pass
