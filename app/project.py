from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PhotoProject:
    source_name: str
    source_bytes: bytes
    recipe: dict
    removal_mask: bytes | None = None
    local_mask: bytes | None = None

    def save(self, destination: str | Path | io.BytesIO) -> Path | io.BytesIO:
        if hasattr(destination, "write"):
            archive_target = destination
            path = None
        else:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            archive_target = path
        manifest = {
            "version": 1,
            "source_name": self.source_name,
            "recipe": self.recipe,
            "files": {
                "source": "source" + Path(self.source_name).suffix.lower(),
                "removal_mask": "masks/removal.png" if self.removal_mask else None,
                "local_mask": "masks/local.png" if self.local_mask else None,
            },
        }
        with zipfile.ZipFile(archive_target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.json", json.dumps(manifest, indent=2, sort_keys=True))
            archive.writestr(manifest["files"]["source"], self.source_bytes)
            if self.removal_mask:
                archive.writestr("masks/removal.png", self.removal_mask)
            if self.local_mask:
                archive.writestr("masks/local.png", self.local_mask)
        return destination

    @classmethod
    def load(cls, project_file: str | Path | bytes) -> "PhotoProject":
        if isinstance(project_file, (str, Path)):
            archive = zipfile.ZipFile(project_file)
        else:
            archive = zipfile.ZipFile(io.BytesIO(project_file))
        with archive:
            manifest = json.loads(archive.read("project.json"))
            files = manifest["files"]
            return cls(
                source_name=manifest["source_name"],
                source_bytes=archive.read(files["source"]),
                recipe=manifest.get("recipe", {}),
                removal_mask=archive.read(files["removal_mask"]) if files.get("removal_mask") else None,
                local_mask=archive.read(files["local_mask"]) if files.get("local_mask") else None,
            )
