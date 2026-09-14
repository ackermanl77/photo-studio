from __future__ import annotations

import json
from pathlib import Path


class PresetStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or (Path.home() / ".photo-studio" / "presets.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Preset file is invalid") from error

    def save(self, name: str, recipe: dict) -> None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Preset name cannot be empty")
        presets = self.load()
        presets[clean_name] = recipe
        self.path.write_text(json.dumps(presets, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, name: str) -> dict:
        try:
            return self.load()[name]
        except KeyError as error:
            raise KeyError(f"Preset not found: {name}") from error

    def delete(self, name: str) -> None:
        presets = self.load()
        presets.pop(name, None)
        self.path.write_text(json.dumps(presets, indent=2, sort_keys=True), encoding="utf-8")

    def names(self) -> list[str]:
        return sorted(self.load())
