"""The original (version 1) .photoproject writer/loader, kept only as a test fixture: the app no
longer writes this format (app.project_bundle.ProjectBundle does), but ProjectBundle.load() must
keep opening files saved by it. Moved out of app/ (B12) since nothing in the app used it."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.reliability import validate_project_archive


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
        # validate_project_archive() (app/reliability.py) enforces the same entry-count/size/path/
        # manifest-reference checks ProjectBundle.load() already relies on -- this used to open the
        # ZIP directly with none of them, so a hand-crafted archive could decompress far beyond its
        # declared size, or a corrupt/hand-edited manifest could raise a bare KeyError instead of a
        # clear error.
        archive_bytes = project_file if isinstance(project_file, bytes) else Path(project_file).read_bytes()
        manifest = validate_project_archive(archive_bytes)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            files = manifest["files"]
            return cls(
                source_name=manifest["source_name"],
                source_bytes=archive.read(files["source"]),
                recipe=manifest.get("recipe", {}),
                removal_mask=archive.read(files["removal_mask"]) if files.get("removal_mask") else None,
                local_mask=archive.read(files["local_mask"]) if files.get("local_mask") else None,
            )
