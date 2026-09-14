import io

import numpy as np
from PIL import Image

from app.layer_workspace import LayerWorkspace, WorkspaceLayer


def test_layer_workspace_manages_layers_and_composites():
    base = Image.new("RGB", (20, 12), (80, 80, 80))
    overlay = Image.new("RGBA", (20, 12), (220, 20, 20, 255))
    workspace = LayerWorkspace(base).add(WorkspaceLayer("Subject", overlay, opacity=0.5, blend_mode="normal"))

    result = workspace.composite()

    assert result.size == base.size
    assert 130 <= int(np.asarray(result)[4, 4, 0]) <= 160
    workspace.duplicate("Subject").move("Subject copy", -1).remove("Subject")
    assert [layer.name for layer in workspace.layers] == ["Background", "Subject copy"]


def test_layer_workspace_round_trips_mask_and_visibility():
    base = Image.new("RGB", (8, 8), (10, 20, 30))
    overlay = Image.new("RGBA", (8, 8), (200, 100, 50, 255))
    mask = Image.new("L", (8, 8), 255)
    workspace = LayerWorkspace(base).add(WorkspaceLayer("Masked", overlay, visible=False, mask=mask))

    restored = LayerWorkspace.from_json(workspace.to_json())

    assert restored.layers[1].name == "Masked"
    assert restored.layers[1].visible is False
    assert restored.layers[1].mask.size == (8, 8)
