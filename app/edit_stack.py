from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from app.image_processor import PhotoProcessor
from app.presets import apply_preset


@dataclass
class Operation:
    name: str
    kind: str
    parameters: dict[str, Any]
    enabled: bool = True


class EditStack:
    """Serializable, reorderable non-destructive edit recipe."""

    def __init__(self, operations: list[Operation] | None = None):
        self.operations = operations or []

    def add(self, name: str, kind: str, parameters: dict[str, Any]) -> "EditStack":
        self.operations.append(Operation(name, kind, parameters))
        return self

    def set_enabled(self, name: str, enabled: bool) -> "EditStack":
        self._find(name).enabled = enabled
        return self

    def move(self, name: str, offset: int) -> "EditStack":
        index = next(index for index, operation in enumerate(self.operations) if operation.name == name)
        target = max(0, min(len(self.operations) - 1, index + int(offset)))
        operation = self.operations.pop(index)
        self.operations.insert(target, operation)
        return self

    def remove(self, name: str) -> "EditStack":
        self.operations = [operation for operation in self.operations if operation.name != name]
        return self

    def apply(self, processor: PhotoProcessor) -> PhotoProcessor:
        for operation in self.operations:
            if not operation.enabled:
                continue
            parameters = operation.parameters
            if operation.kind == "preset":
                apply_preset(processor, parameters["name"])
            else:
                method = getattr(processor, operation.kind)
                method(**parameters)
        return processor

    def to_json(self) -> str:
        return json.dumps({"version": 1, "operations": [asdict(operation) for operation in self.operations]}, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, content: str | bytes) -> "EditStack":
        payload = json.loads(content)
        if payload.get("version") != 1:
            raise ValueError("Unsupported edit stack version")
        return cls([Operation(**operation) for operation in payload.get("operations", [])])

    def _find(self, name: str) -> Operation:
        for operation in self.operations:
            if operation.name == name:
                return operation
        raise KeyError(f"Unknown operation: {name}")
