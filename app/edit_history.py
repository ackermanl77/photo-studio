from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EditHistory:
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    cursor: int = -1

    def record(self, options: dict[str, Any]) -> None:
        snapshot = json.loads(json.dumps(options))
        if self.current == snapshot:
            return
        if self.cursor < len(self.snapshots) - 1:
            self.snapshots = self.snapshots[: self.cursor + 1]
        self.snapshots.append(snapshot)
        self.cursor = len(self.snapshots) - 1

    @property
    def current(self) -> dict[str, Any] | None:
        if self.cursor < 0:
            return None
        return json.loads(json.dumps(self.snapshots[self.cursor]))

    @property
    def can_undo(self) -> bool:
        return self.cursor > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self.cursor < len(self.snapshots) - 1

    def undo(self) -> dict[str, Any] | None:
        if not self.can_undo:
            return self.current
        self.cursor -= 1
        return self.current

    def redo(self) -> dict[str, Any] | None:
        if not self.can_redo:
            return self.current
        self.cursor += 1
        return self.current

    def to_json(self) -> str:
        return json.dumps({"version": 1, "options": self.current or {}}, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, content: str | bytes) -> "EditHistory":
        payload = json.loads(content)
        options = payload.get("options", payload)
        history = cls()
        history.record(options)
        return history
