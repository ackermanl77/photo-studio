import io

from PIL import Image

from app.layer_workspace import LayerWorkspace, WorkspaceLayer
from app.project_bundle import ProjectBundle


def test_project_bundle_round_trip_restores_source_layers_masks_and_recipe(tmp_path):
    source = io.BytesIO()
    Image.new("RGB", (10, 10), (30, 40, 50)).save(source, format="PNG")
    overlay = Image.new("RGBA", (10, 10), (200, 20, 30, 255))
    workspace = LayerWorkspace(Image.open(io.BytesIO(source.getvalue()))).add(WorkspaceLayer("Overlay", overlay, opacity=0.5))
    destination = tmp_path / "session.photoproject"

    bundle = ProjectBundle("source.png", source.getvalue(), {"exposure": 1.1}, workspace.to_json(), {"local.png": b"mask"})
    bundle.save(destination)
    restored = ProjectBundle.load(destination)

    assert restored.source_name == "source.png"
    assert restored.recipe == {"exposure": 1.1}
    assert restored.layer_workspace_object().layers[1].name == "Overlay"
    assert restored.masks["local.png"] == b"mask"
