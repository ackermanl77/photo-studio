from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app import __version__
from app.edit_schema import RECIPE_VERSION, migrate_recipe
from app.layer_workspace import LayerWorkspace
from app.reliability import validate_project_archive


# Layer stack document version inside a project (files.layer_stack). 1 was never written under that
# key -- older projects stored flattened pixels as LayerWorkspace JSON in files.layers instead.
LAYER_STACK_VERSION = 2
_IMAGE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}\.png$")


@dataclass
class ProjectBundle:
    """A .photoproject archive. Layers are stored one of two ways:

    - `layer_stack` (current): the Layers tab's own list of layer dicts, parametric -- a Text,
      Shape or Path layer keeps its text/font/geometry/styles and reopens editable -- with each
      Image layer's pixels as its own PNG entry under layers/ (`layer_images`, keyed by file name;
      the dict references it as "image_file"), not base64 inside JSON (B10 in the improvement
      report: one photo-sized layer used to cost ~122 MB of JSON).
    - `layer_workspace` (older projects, still loadable): LayerWorkspace JSON of flattened RGBA
      layers, which reopen as plain Image layers.
    """

    source_name: str
    source_bytes: bytes
    recipe: dict
    layer_workspace: str | None = None
    masks: dict[str, bytes] | None = None
    layer_stack: list[dict] | None = None
    layer_images: dict[str, bytes] | None = None

    def _manifest(self) -> dict:
        # app_version/recipe_version: which build wrote this file and which recipe shape it
        # stores, so load() can migrate an older recipe (app.edit_schema.migrate_recipe) and
        # refuse one from a newer build instead of misreading it.
        return {
            "version": 2,
            "app_version": __version__,
            "recipe_version": RECIPE_VERSION,
            "source_name": self.source_name,
            "recipe": self.recipe,
            "files": {
                "source": "source" + Path(self.source_name).suffix.lower(),
                "layers": "layers.json" if self.layer_workspace else None,
                "layer_stack": "layers/stack.json" if self.layer_stack is not None else None,
                "layer_images": sorted((self.layer_images or {}).keys()),
                "masks": sorted((self.masks or {}).keys()),
            },
        }

    def save(self, destination: str | Path | io.BytesIO) -> str | Path | io.BytesIO:
        manifest = self._manifest()
        target = destination if hasattr(destination, "write") else Path(destination)
        if isinstance(target, Path):
            target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.json", json.dumps(manifest, indent=2, sort_keys=True))
            archive.writestr(manifest["files"]["source"], self.source_bytes)
            if self.layer_workspace:
                archive.writestr("layers.json", self.layer_workspace)
            if self.layer_stack is not None:
                archive.writestr("layers/stack.json", json.dumps({"version": LAYER_STACK_VERSION, "layers": self.layer_stack}, indent=2, sort_keys=True))
            for name, content in (self.layer_images or {}).items():
                if not _IMAGE_NAME.match(name):
                    raise ValueError(f"Invalid layer image name: {name!r}")
                archive.writestr(f"layers/{name}", content)
            for name, content in (self.masks or {}).items():
                archive.writestr(f"masks/{name}", content)
        return destination

    def to_bytes(self) -> bytes:
        """The archive as bytes, checked against the same limits load() enforces
        (app.reliability.validate_project_archive) -- a project over the decompressed-size or
        entry-count limit used to save fine and then refuse to reopen. Raises ValueError instead,
        so the caller can tell the user *before* they rely on the file."""
        buffer = io.BytesIO()
        self.save(buffer)
        content = buffer.getvalue()
        try:
            validate_project_archive(content)
        except ValueError as error:
            raise ValueError(
                f"This project couldn't be reopened once saved ({error}) -- usually because of very "
                "large image layers. Remove or shrink some layers and save again."
            ) from error
        return content

    @classmethod
    def load(cls, content: bytes | str | Path) -> "ProjectBundle":
        archive_bytes = content if isinstance(content, bytes) else Path(content).read_bytes()
        manifest = validate_project_archive(archive_bytes)
        recipe = manifest.get("recipe", {})
        if not isinstance(recipe, dict):
            raise ValueError("Project recipe is not an object")
        recipe = migrate_recipe(recipe, manifest.get("recipe_version"))
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            files = manifest["files"]
            layers = archive.read(files["layers"]).decode("utf-8") if files.get("layers") else None
            masks = {}
            for name in files.get("masks", []):
                safe_name = str(name)
                masks[safe_name] = archive.read(f"masks/{safe_name}")
            layer_stack = layer_images = None
            if files.get("layer_stack"):
                document = json.loads(archive.read(files["layer_stack"]))
                if not isinstance(document, dict) or document.get("version") != LAYER_STACK_VERSION or not isinstance(document.get("layers"), list):
                    raise ValueError("Unsupported layer stack in project")
                layer_stack = [layer for layer in document["layers"] if isinstance(layer, dict)]
                layer_images = {str(name): archive.read(f"layers/{name}") for name in files.get("layer_images") or []}
            return cls(manifest["source_name"], archive.read(files["source"]), recipe, layers, masks, layer_stack, layer_images)

    def layer_workspace_object(self) -> LayerWorkspace | None:
        return LayerWorkspace.from_json(self.layer_workspace) if self.layer_workspace else None
