from __future__ import annotations

import json
from pathlib import Path

from app.config import atomic_write_text, user_data_file


class PresetStore:
    def __init__(self, path: str | Path | None = None):
        # Default: presets.json in the one per-user data folder (app.config.user_data_dir()),
        # migrated from the old ~/.photo-studio location on first use -- see user_data_file().
        self.path = Path(path) if path else user_data_file("presets.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            presets = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Preset file is invalid: {self.path}") from error
        if not isinstance(presets, dict):
            raise ValueError(f"Preset file is invalid: {self.path}")
        return presets

    def _write(self, presets: dict) -> None:
        # Atomic (temp file + os.replace): a crash mid-write used to be able to leave a truncated
        # presets.json behind -- exactly the corruption that makes load() raise.
        atomic_write_text(self.path, json.dumps(presets, indent=2, sort_keys=True))

    def save(self, name: str, recipe: dict) -> None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Preset name cannot be empty")
        presets = self.load()
        presets[clean_name] = recipe
        self._write(presets)

    def get(self, name: str) -> dict:
        try:
            return self.load()[name]
        except KeyError as error:
            raise KeyError(f"Preset not found: {name}") from error

    def delete(self, name: str) -> None:
        presets = self.load()
        presets.pop(name, None)
        self._write(presets)

    def names(self) -> list[str]:
        return sorted(self.load())

    def reset(self) -> Path | None:
        """Recovery for an unreadable presets file (the frontend's "Reset presets" button): moves
        it aside as presets.json.corrupt-N rather than deleting it, so nothing is lost for good, and
        returns where it went (None if there was no file)."""
        if not self.path.exists():
            return None
        counter = 1
        while (backup := self.path.with_name(f"{self.path.name}.corrupt-{counter}")).exists():
            counter += 1
        self.path.replace(backup)
        return backup
