import io
from unittest import mock

import numpy as np
from PIL import Image

from app.layer_workspace import LayerWorkspace, WorkspaceLayer, layers_from_json, layers_to_json, render_text_layer


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


def test_layers_to_json_and_from_json_round_trip_a_plain_layer_list_with_no_background():
    """layers_to_json/layers_from_json are the wire format the frontend's now-multi-layer Layers
    tab sends to app/backend.py's /process endpoint -- unlike LayerWorkspace.to_json(), there is no
    mandatory first "Background" entry, since the backend already has its own in-progress
    processor.image to composite these on top of."""
    first = WorkspaceLayer("Sky", Image.new("RGBA", (6, 4), (10, 20, 200, 255)), opacity=0.8, blend_mode="multiply")
    second = WorkspaceLayer("Glow", Image.new("RGBA", (6, 4), (255, 240, 200, 128)), opacity=1.0, blend_mode="soft_light", visible=False)

    restored = layers_from_json(layers_to_json([first, second]))

    assert [layer.name for layer in restored] == ["Sky", "Glow"]
    assert restored[0].opacity == 0.8
    assert restored[0].blend_mode == "multiply"
    assert restored[1].visible is False
    assert restored[1].blend_mode == "soft_light"


def test_layers_from_json_returns_an_empty_list_for_no_layers():
    assert layers_from_json(layers_to_json([])) == []


def test_render_text_layer_produces_transparent_image_with_visible_text():
    image = render_text_layer((200, 100), "Hi", font_size=40, color=(255, 0, 0, 255))

    assert image.size == (200, 100)
    assert image.mode == "RGBA"
    alpha = np.asarray(image)[:, :, 3]
    assert alpha.max() == 255  # some fully-opaque text pixels
    assert alpha.min() == 0  # and plenty of fully-transparent background


def test_render_text_layer_with_empty_text_is_fully_transparent():
    image = render_text_layer((50, 50), "")

    assert np.asarray(image)[:, :, 3].max() == 0


def test_render_text_layer_x_and_y_pct_move_where_the_text_is_drawn():
    """x_pct/y_pct place the text block's own CENTER at that fraction of the canvas -- text
    anchored near the left edge should leave the right half of the canvas empty, and vice versa."""
    left_aligned = render_text_layer((300, 100), "AB", font_size=30, x_pct=10.0, y_pct=50.0)
    right_aligned = render_text_layer((300, 100), "AB", font_size=30, x_pct=90.0, y_pct=50.0)

    left_alpha = np.asarray(left_aligned)[:, :, 3]
    right_alpha = np.asarray(right_aligned)[:, :, 3]
    assert left_alpha[:, :150].sum() > left_alpha[:, 150:].sum()
    assert right_alpha[:, 150:].sum() > right_alpha[:, :150].sum()


def test_render_text_layer_falls_back_to_pillows_default_font_when_no_system_font_is_found():
    """On a machine (or CI runner) with none of the candidate TrueType fonts installed, this must
    still render legible text rather than raising -- confirmed here by forcing that path instead of
    depending on which fonts happen to be installed on whatever machine runs this test."""
    from app import fonts

    with mock.patch.object(fonts, "_DEFAULT_CANDIDATES", ()):
        image = render_text_layer((200, 100), "Hi", font_size=40)

    assert np.asarray(image)[:, :, 3].max() > 0
