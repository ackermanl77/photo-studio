from __future__ import annotations

import json
from pathlib import Path

from app.custom_presets import PresetStore


def export_presets(store: PresetStore, names: list[str] | None = None) -> bytes:
    presets = store.load()
    selected = presets if names is None else {name: presets[name] for name in names if name in presets}
    return json.dumps({"version": 1, "presets": selected}, indent=2, sort_keys=True).encode("utf-8")


def import_presets(store: PresetStore, content: bytes | str, overwrite: bool = False) -> list[str]:
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Preset file is not valid JSON") from error
    if payload.get("version") != 1 or not isinstance(payload.get("presets"), dict):
        raise ValueError("Unsupported preset file")
    existing = store.load()
    imported: list[str] = []
    for name, recipe in payload["presets"].items():
        if not isinstance(name, str) or not isinstance(recipe, dict):
            raise ValueError("Invalid preset entry")
        if name in existing and not overwrite:
            continue
        existing[name] = recipe
        imported.append(name)
    store.path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    return imported
