from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.layer_workspace import LayerWorkspace
from app.reliability import validate_project_archive


@dataclass
class ProjectBundle:
    source_name: str
    source_bytes: bytes
    recipe: dict
    layer_workspace: str | None = None
    masks: dict[str, bytes] | None = None

    def save(self, destination: str | Path | io.BytesIO) -> str | Path | io.BytesIO:
        manifest = {
            "version": 2,
            "source_name": self.source_name,
            "recipe": self.recipe,
            "files": {
                "source": "source" + Path(self.source_name).suffix.lower(),
                "layers": "layers.json" if self.layer_workspace else None,
                "masks": sorted((self.masks or {}).keys()),
            },
        }
        target = destination if hasattr(destination, "write") else Path(destination)
        if isinstance(target, Path):
            target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.json", json.dumps(manifest, indent=2, sort_keys=True))
            archive.writestr(manifest["files"]["source"], self.source_bytes)
            if self.layer_workspace:
                archive.writestr("layers.json", self.layer_workspace)
            for name, content in (self.masks or {}).items():
                archive.writestr(f"masks/{name}", content)
        return destination

    @classmethod
    def load(cls, content: bytes | str | Path) -> "ProjectBundle":
        archive_bytes = content if isinstance(content, bytes) else Path(content).read_bytes()
        manifest = validate_project_archive(archive_bytes)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            files = manifest["files"]
            layers = archive.read(files["layers"]).decode("utf-8") if files.get("layers") else None
            masks = {}
            for name in files.get("masks", []):
                safe_name = str(name)
                masks[safe_name] = archive.read(f"masks/{safe_name}")
            return cls(manifest["source_name"], archive.read(files["source"]), manifest.get("recipe", {}), layers, masks)

    def layer_workspace_object(self) -> LayerWorkspace | None:
        return LayerWorkspace.from_json(self.layer_workspace) if self.layer_workspace else None
