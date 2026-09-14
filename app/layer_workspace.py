from __future__ import annotations

import base64
import io
import json
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from app.major_features import Layer, LayerStack


@dataclass
class WorkspaceLayer:
    name: str
    image: Image.Image
    opacity: float = 1.0
    blend_mode: str = "normal"
    visible: bool = True
    mask: Image.Image | None = None


class LayerWorkspace:
    def __init__(self, base: Image.Image):
        self.size = base.size
        self.layers = [WorkspaceLayer("Background", base.convert("RGBA"))]

    def add(self, layer: WorkspaceLayer) -> "LayerWorkspace":
        if layer.image.size != self.size:
            layer.image = layer.image.resize(self.size, Image.Resampling.LANCZOS)
        if layer.blend_mode not in LayerStack.MODES:
            raise ValueError(f"Unsupported blend mode: {layer.blend_mode}")
        self.layers.append(layer)
        return self

    def remove(self, name: str) -> "LayerWorkspace":
        if name == "Background":
            raise ValueError("The background layer cannot be removed")
        self.layers = [layer for layer in self.layers if layer.name != name]
        return self

    def duplicate(self, name: str) -> "LayerWorkspace":
        original = self._find(name)
        self.layers.append(WorkspaceLayer(f"{name} copy", original.image.copy(), original.opacity, original.blend_mode, original.visible, original.mask.copy() if original.mask else None))
        return self

    def move(self, name: str, offset: int) -> "LayerWorkspace":
        index = next(index for index, layer in enumerate(self.layers) if layer.name == name)
        target = max(1, min(len(self.layers) - 1, index + int(offset)))
        self.layers.insert(target, self.layers.pop(index))
        return self

    def composite(self) -> Image.Image:
        stack = LayerStack(self.layers[0].image)
        for layer in self.layers[1:]:
            stack.add(Layer(layer.name, layer.image, layer.opacity, layer.blend_mode, layer.mask, layer.visible))
        return stack.composite()

    def to_json(self) -> str:
        payload = {"version": 1, "size": self.size, "layers": []}
        for layer in self.layers:
            image_buffer = io.BytesIO()
            layer.image.save(image_buffer, format="PNG")
            mask = None
            if layer.mask:
                mask_buffer = io.BytesIO()
                layer.mask.save(mask_buffer, format="PNG")
                mask = base64.b64encode(mask_buffer.getvalue()).decode("ascii")
            payload["layers"].append({"name": layer.name, "image": base64.b64encode(image_buffer.getvalue()).decode("ascii"), "opacity": layer.opacity, "blend_mode": layer.blend_mode, "visible": layer.visible, "mask": mask})
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, content: str | bytes) -> "LayerWorkspace":
        payload = json.loads(content)
        if payload.get("version") != 1 or not payload.get("layers"):
            raise ValueError("Invalid layer workspace")
        layers = []
        for data in payload["layers"]:
            image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGBA")
            mask = Image.open(io.BytesIO(base64.b64decode(data["mask"]))).convert("L") if data.get("mask") else None
            layers.append(WorkspaceLayer(data["name"], image, data.get("opacity", 1.0), data.get("blend_mode", "normal"), data.get("visible", True), mask))
        workspace = cls(layers[0].image)
        workspace.layers = layers
        workspace.size = tuple(payload.get("size", workspace.size))
        return workspace

    def _find(self, name: str) -> WorkspaceLayer:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(name)
