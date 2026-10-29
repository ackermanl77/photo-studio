from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# Undo steps kept (P5): each snapshot is a full recipe -- painted masks included -- so an unbounded
# list grew for the whole session.
MAX_HISTORY_STEPS = 200

# History list labels (F13) for the recipe's list-valued entries, which have no registry control.
_LIST_LABELS = {"mask_layers": "Masks", "removal_mask_layers": "Repair", "curve_extra_points": "Curve points", "layers": "Layers",
                "liquify_strokes": "Liquify", "plugin_filters": "Plug-in filters"}


def describe_change(before: dict[str, Any] | None, after: dict[str, Any]) -> str:
    """A short label for one Undo step: what changed from `before` to `after` -- "Exposure (EV)
    0.3" for one control, "Contrast, Clarity" for two, "Tone curve +3 more" for many."""
    from app.controls import CONTROLS_BY_KEY

    if before is None:
        return "Opened"
    changed = [key for key in set(before) | set(after) if before.get(key) != after.get(key)]
    changed.sort(key=lambda key: (list(CONTROLS_BY_KEY).index(key) if key in CONTROLS_BY_KEY else len(CONTROLS_BY_KEY), key))
    if not changed:
        return "No change"

    def label(key: str) -> str:
        control = CONTROLS_BY_KEY.get(key)
        if control is None:
            return _LIST_LABELS.get(key, key.replace("_", " ").capitalize())
        if key.startswith(("hsl_", "bw_")):
            return f"{'HSL' if key.startswith('hsl_') else 'B&W'} {key.split('_')[1].title()} {control.label}"
        if key.startswith("curve_"):
            return "Tone curve"
        return control.label

    labels = list(dict.fromkeys(label(key) for key in changed))
    if len(changed) == 1 and changed[0] in CONTROLS_BY_KEY:
        value = after.get(changed[0])
        if isinstance(value, bool):
            return f"{labels[0]} {'on' if value else 'off'}"
        if isinstance(value, float):
            value = round(value, 2)
        return f"{labels[0]} {value}"
    if len(labels) <= 2:
        return ", ".join(labels)
    return f"{labels[0]} +{len(labels) - 1} more"


@dataclass
class EditHistory:
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    cursor: int = -1
    # Named bookmarks, independent of the linear Undo/Redo step list above. Keyed by name (not a
    # list index) so saving/restoring/deleting one never disturbs another's identity the way a
    # positional index would after a reorder -- the same reason mask/layer lists in the frontend
    # use stable ids instead of list position (see CLAUDE.md).
    named_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, options: dict[str, Any]) -> None:
        snapshot = json.loads(json.dumps(options))
        if self.current == snapshot:
            return
        if self.cursor < len(self.snapshots) - 1:
            self.snapshots = self.snapshots[: self.cursor + 1]
        self.snapshots.append(snapshot)
        if len(self.snapshots) > MAX_HISTORY_STEPS:
            del self.snapshots[: len(self.snapshots) - MAX_HISTORY_STEPS]
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

    def steps(self) -> list[str]:
        """One label per Undo step, oldest first (describe_change); `cursor` is the current one."""
        return [describe_change(self.snapshots[index - 1] if index else None, snapshot) for index, snapshot in enumerate(self.snapshots)]

    def jump(self, index: int) -> dict[str, Any] | None:
        """Move to Undo step `index` (as if Undo/Redo were pressed until there) and return it."""
        if 0 <= index < len(self.snapshots):
            self.cursor = index
        return self.current

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

    def save_snapshot(self, name: str, options: dict[str, Any]) -> None:
        """Bookmark the current edit state under `name` for one-click recall later. Deliberately
        does not touch `self.snapshots`/`self.cursor` -- saving a snapshot is not an edit, so it
        must not create an Undo step or truncate a redo branch the way `record()` does.
        Overwrites any existing snapshot with the same name."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Snapshot name must not be empty")
        self.named_snapshots[clean_name] = json.loads(json.dumps(options))

    def restore_snapshot(self, name: str) -> dict[str, Any] | None:
        """Return a deep copy of a saved snapshot's options, or None if no snapshot has that
        name. Restoring never removes the snapshot -- it's a bookmark you can jump back to any
        number of times, not a one-shot consumed on first use, and it never mutates the linear
        Undo/Redo list either (the caller is expected to route the result through the same
        `_pending_restore` + rerun mechanism as Undo/Redo/Reset, at which point the frontend's
        own `history.record()` call folds it into Undo/Redo history exactly like any other new
        edit would -- restoring a snapshot is not itself special-cased there)."""
        snapshot = self.named_snapshots.get(name)
        if snapshot is None:
            return None
        return json.loads(json.dumps(snapshot))

    def delete_snapshot(self, name: str) -> None:
        """No-op if `name` isn't a saved snapshot."""
        self.named_snapshots.pop(name, None)

    @property
    def snapshot_names(self) -> list[str]:
        return list(self.named_snapshots)

    def to_json(self) -> str:
        return json.dumps({"version": 1, "options": self.current or {}}, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, content: str | bytes) -> "EditHistory":
        payload = json.loads(content)
        options = payload.get("options", payload)
        history = cls()
        history.record(options)
        return history
