from pathlib import Path

from PIL import Image

from app.layer_workspace import LayerWorkspace
from app.project_bundle import ProjectBundle


def test_project_frontend_contract_can_restore_and_resave(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (12, 8), (40, 50, 60)).save(source)
    project_path = tmp_path / "session.photoproject"
    ProjectBundle(source.name, source.read_bytes(), {"preset": "portrait-natural"}, LayerWorkspace(Image.open(source)).to_json()).save(project_path)

    restored = ProjectBundle.load(project_path)
    output_path = tmp_path / "resaved.photoproject"
    ProjectBundle(restored.source_name, restored.source_bytes, restored.recipe, restored.layer_workspace, restored.masks).save(output_path)

    assert output_path.exists()
    assert restored.layer_workspace_object().layers[0].name == "Background"
