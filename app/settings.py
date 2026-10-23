"""The user's remembered preferences: user_data_dir()/settings.json, one JSON object -- the interface
language (app.i18n) and the workspace layout (the last open tool tabs, app.ui.layout_memory).

Best effort both ways: an unreadable file reads as no settings, and a read-only data folder makes
saving a no-op, never an error on the page.
"""
from __future__ import annotations

import json
from pathlib import Path


def settings_path() -> Path:
    from app.config import user_data_dir

    return user_data_dir() / "settings.json"


def read_settings() -> dict:
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_setting(key: str, value) -> None:
    """Remember `value` under `key`, keeping the other settings."""
    from app.config import atomic_write_text

    settings = read_settings()
    if settings.get(key) == value:
        return
    settings[key] = value
    try:
        atomic_write_text(settings_path(), json.dumps(settings, indent=2, ensure_ascii=False))
    except OSError:
        pass
