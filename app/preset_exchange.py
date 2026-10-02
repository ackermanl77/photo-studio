from __future__ import annotations

import json

from app.config import atomic_write_text
from app.custom_presets import PresetStore
from app.edit_schema import RECIPE_VERSION, migrate_recipe, sanitize_recipe


def export_presets(store: PresetStore, names: list[str] | None = None) -> bytes:
    presets = store.load()
    selected = presets if names is None else {name: presets[name] for name in names if name in presets}
    return json.dumps({"version": 1, "recipe_version": RECIPE_VERSION, "presets": selected}, indent=2, sort_keys=True).encode("utf-8")


def import_presets(store: PresetStore, content: bytes | str, overwrite: bool = False) -> list[str]:
    """Imported recipes are migrated and sanitized (app.edit_schema.sanitize_recipe) before they
    are stored: a hand-edited or foreign file with an out-of-range value, a wrong type, or keys
    this build doesn't know used to be stored verbatim and could later take the page down when
    applied. A file whose recipe_version is newer than this build is refused outright."""
    try:
        payload = json.loads(content)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Preset file is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("presets"), dict):
        raise ValueError("Unsupported preset file")
    recipe_version = payload.get("recipe_version")
    existing = store.load()
    imported: list[str] = []
    for name, recipe in payload["presets"].items():
        if not isinstance(name, str) or not name.strip() or not isinstance(recipe, dict):
            raise ValueError("Invalid preset entry")
        if name in existing and not overwrite:
            continue
        existing[name] = sanitize_recipe(migrate_recipe(recipe, recipe_version))
        imported.append(name)
    atomic_write_text(store.path, json.dumps(existing, indent=2, sort_keys=True))
    return imported
