"""Adjustment layers (F25): an adjustment set applied to everything below it in the Layers stack,
through its mask and opacity; the wire format, the payload builder and the Layers tab."""
import json

import numpy as np
import pytest
from PIL import Image

from app.edit_schema import clean_adjustment_set, parse_options
from app.layer_workspace import WorkspaceLayer, layers_from_json, layers_to_json
from app.pipeline import run_pipeline


def _photo():
    return Image.new("RGB", (80, 60), (100, 100, 100))


def _square_layer():
    image = Image.new("RGBA", (80, 60), (0, 0, 0, 0))
    image.paste((200, 50, 50, 255), (10, 10, 30, 30))
    return WorkspaceLayer("Square", image)


def _adjustment(ev=1.0, mask=None, opacity=1.0, visible=True, **extra_adjustments):
    return WorkspaceLayer("Brighten", Image.new("RGBA", (1, 1)), opacity, "normal", visible, mask,
                          clean_adjustment_set({"exposure_ev": ev, **extra_adjustments}))


def test_it_adjusts_the_photo_and_the_layers_below_but_not_above():
    below, above = _square_layer(), WorkspaceLayer("Top", Image.new("RGBA", (80, 60), (0, 0, 0, 0)))
    above.image.paste((50, 50, 200, 255), (50, 10, 70, 30))
    plain = np.asarray(run_pipeline(_photo(), parse_options({}), layers=[below, above])).astype(int)
    adjusted = np.asarray(run_pipeline(_photo(), parse_options({}), layers=[below, _adjustment(1.0), above])).astype(int)
    assert adjusted[40, 40, 0] > plain[40, 40, 0] + 30  # the photo is brighter
    assert adjusted[20, 20, 1] > plain[20, 20, 1] + 15  # so is the square under it
    np.testing.assert_array_equal(adjusted[20, 60], plain[20, 60])  # the layer above isn't


def test_mask_opacity_and_visibility():
    mask = Image.new("L", (80, 60), 0)
    mask.paste(255, (40, 0, 80, 60))  # right half only
    plain = np.asarray(run_pipeline(_photo(), parse_options({}), layers=[_square_layer()])).astype(int)
    masked = np.asarray(run_pipeline(_photo(), parse_options({}), layers=[_adjustment(1.0, mask)])).astype(int)
    assert masked[30, 60, 0] > plain[30, 60, 0] + 30 and masked[30, 50 - 40, 0] == plain[30, 10, 0]
    half = np.asarray(run_pipeline(_photo(), parse_options({}), layers=[_adjustment(1.0, opacity=0.5)])).astype(int)
    full = np.asarray(run_pipeline(_photo(), parse_options({}), layers=[_adjustment(1.0)])).astype(int)
    assert plain[30, 60, 0] < half[30, 60, 0] < full[30, 60, 0]
    hidden = np.asarray(run_pipeline(_photo(), parse_options({}), layers=[_adjustment(1.0, visible=False)])).astype(int)
    np.testing.assert_array_equal(hidden, np.asarray(_photo()))


def test_wire_format_round_trip_and_cleaning():
    layer = _adjustment(0.5)
    restored = layers_from_json(layers_to_json([layer, _square_layer()]))
    assert restored[0].is_adjustment and restored[0].adjustments["exposure_ev"] == 0.5
    assert not restored[1].is_adjustment
    assert clean_adjustment_set({"exposure_ev": 99, "contrast": "x", "bogus": 1}) == {
        "exposure_ev": 5.0, "contrast": 0.0, "saturation": 0.0, "temperature": 0.0, "tint": 0.0,
        "clarity": 0.0, "exposure_offset": 0.0, "exposure_gamma": 1.0,
        "posterize_levels": 0.0, "threshold_level": 0.0,
    }


def test_ps1_adjustments_are_available_through_adjustment_layers_and_masks():
    mask = Image.new("L", (80, 60), 0)
    mask.paste(255, (40, 0, 80, 60))
    plain = np.asarray(run_pipeline(_photo(), parse_options({}), layers=[_square_layer()])).astype(int)
    threshold = np.asarray(run_pipeline(
        _photo(), parse_options({}), layers=[_adjustment(ev=0.0, mask=mask, threshold_level=80)]
    )).astype(int)
    assert threshold[30, 60, 0] == 255 and threshold[30, 20, 0] == plain[30, 20, 0]

    posterized = np.asarray(run_pipeline(
        _photo(), parse_options({}), layers=[_adjustment(ev=0.0, posterize_levels=2)]
    ))
    assert posterized[30, 60, 0] == 0

    offset = np.asarray(run_pipeline(
        _photo(), parse_options({}), layers=[_adjustment(ev=0.0, exposure_offset=0.25)]
    ))
    assert offset[30, 60, 0] > plain[30, 60, 0]


def test_payload_builder_and_psd_export_skip_pixels():
    from app.ui.layers import build_layers_payload, layer_dict_to_image

    stack = [{"kind": "adjustment", "name": "Warm", "adjustments": {"temperature": 40}, "opacity": 1.0,
              "blend_mode": "normal", "visible": True}]
    layers = layers_from_json(build_layers_payload(json.dumps(stack), (80, 60)))
    assert layers[0].is_adjustment and layers[0].adjustments["temperature"] == 40
    assert layer_dict_to_image(stack[0], (80, 60)).getbbox() is None  # transparent: nothing to draw


@pytest.mark.slow
def test_the_layers_tab_adds_an_adjustment_layer(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import backend_client
    from tests.test_frontend_unified_app import APP_PATH, _fake_process_via_backend, image_bytes, open_tab

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("photo.png", image_bytes(size=(40, 30)), "image/png"))
    at.run(timeout=30)
    open_tab(at, "◈ Layers")
    at.radio(key="l-new-kind").set_value("Adjustment")
    at.run(timeout=30)
    at.button(key="l-add-adjustment-btn").click()
    at.run(timeout=30)
    at.run(timeout=30)
    assert not at.exception
    layer = at.session_state["image_layers"][0]
    assert layer["kind"] == "adjustment"
    at.slider(key=f"l-adj-exposure_ev-{layer['id']}").set_value(1.0)
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["image_layers"][0]["adjustments"]["exposure_ev"] == 1.0
