"""AI masks (F16): the classical Sky mask, Subject/Person from (stand-in) class probabilities, the
Painted-selection format, the backend route and the Masks tab. No model is downloaded here."""
import io
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.ai_masks import PERSON_CLASS, make_ai_mask, mask_to_painted_b64, sky_mask, weights_cached
from app.ui.selections import decode_painted_mask


def _landscape(size=(320, 240)):
    image = Image.new("RGB", size, (90, 150, 220))  # sky
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, size[1] * 0.55, size[0], size[1]), fill=(60, 110, 40))  # field
    rng = np.random.default_rng(0)
    array = np.asarray(image).astype(np.int16)
    array[int(size[1] * 0.55):] += rng.integers(-40, 40, array[int(size[1] * 0.55):].shape, dtype=np.int16)  # textured grass
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def test_sky_mask_covers_the_sky_and_not_the_field():
    mask = np.asarray(sky_mask(_landscape()))
    assert mask.shape == (240, 320)
    assert mask[:100].mean() > 200 and mask[150:].mean() < 30


def test_sky_must_touch_the_top_edge():
    image = Image.new("RGB", (200, 150), (60, 60, 60))
    ImageDraw.Draw(image).rectangle((60, 60, 140, 120), fill=(90, 150, 220))  # a blue box in the middle
    assert np.asarray(sky_mask(image)).max() < 40


def _fake_probabilities(image):
    """Background everywhere except a person in the middle and a dog on the right."""
    h, w = 60, 80
    probs = np.zeros((21, h, w), np.float32)
    probs[0] = 1.0
    probs[0, 10:50, 30:50], probs[PERSON_CLASS, 10:50, 30:50] = 0.0, 1.0
    probs[0, 30:55, 60:75], probs[12, 30:55, 60:75] = 0.0, 1.0  # VOC 12 = dog
    return probs


def test_person_and_subject_from_class_probabilities():
    image = Image.new("RGB", (160, 120))
    person = np.asarray(make_ai_mask(image, "Person", _fake_probabilities))
    subject = np.asarray(make_ai_mask(image, "Subject", _fake_probabilities))
    assert person.shape == (120, 160)
    assert person[60, 80] > 200 and person[85, 135] < 30  # the dog isn't a person
    assert subject[60, 80] > 200 and subject[85, 135] > 200 and subject[5, 5] < 30
    with pytest.raises(ValueError):
        make_ai_mask(image, "Car")


def test_painted_format_round_trips():
    mask = Image.new("L", (2000, 1000), 0)
    ImageDraw.Draw(mask).rectangle((1000, 0, 2000, 1000), fill=255)
    decoded = decode_painted_mask(mask_to_painted_b64(mask))
    assert decoded.size == (1024, 512)
    assert decoded.getpixel((900, 256)) > 250 and decoded.getpixel((100, 256)) < 5


def test_weights_check_does_not_import_torch(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert weights_cached() is False
    assert "torch" not in sys.modules
    (tmp_path / "hub" / "checkpoints").mkdir(parents=True)
    (tmp_path / "hub" / "checkpoints" / "deeplabv3_mobilenet_v3_large-fc3c493d.pth").write_bytes(b"x")
    assert weights_cached() is True


def test_backend_route(monkeypatch):
    from app import ai_masks
    from app.backend import app as flask_app

    buffer = io.BytesIO()
    _landscape().save(buffer, format="PNG")
    client = flask_app.test_client()
    response = client.post("/ai_mask", data={"kind": "Sky", "image": (io.BytesIO(buffer.getvalue()), "a.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.data)).mode == "L"
    assert client.post("/ai_mask", data={"kind": "Car", "image": (io.BytesIO(buffer.getvalue()), "a.png")},
                       content_type="multipart/form-data").status_code == 400
    monkeypatch.setattr(ai_masks, "ai_masks_installed", lambda: True)
    monkeypatch.setattr(ai_masks, "class_probabilities", _fake_probabilities)
    monkeypatch.setattr(ai_masks, "make_ai_mask", lambda image, kind: ai_masks.model_mask(image, kind, _fake_probabilities))
    response = client.post("/ai_mask", data={"kind": "Person", "image": (io.BytesIO(buffer.getvalue()), "a.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 200
    assert np.asarray(Image.open(io.BytesIO(response.data)))[120, 160] > 200


@pytest.mark.slow
def test_the_masks_tab_adds_an_ai_mask_as_a_painted_selection(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import backend_client
    from tests.test_frontend_unified_app import APP_PATH, _fake_process_via_backend, image_bytes, open_tab

    def fake_ai_mask(image_bytes_, kind, timeout=600):
        mask = sky_mask(Image.open(io.BytesIO(image_bytes_)))
        out = io.BytesIO()
        mask.save(out, format="PNG")
        return out.getvalue()

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    monkeypatch.setattr(backend_client, "ai_mask_via_backend", fake_ai_mask)
    landscape = io.BytesIO()
    _landscape((160, 120)).save(landscape, format="PNG")
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("landscape.png", landscape.getvalue(), "image/png"))
    at.run(timeout=30)
    open_tab(at, "◉ Local", "◉ Masks")
    selectbox = next(s for s in at.selectbox if s.key and s.key.endswith("-new-kind") and "AI Mask" in s.options)
    selectbox.set_value("AI Mask")
    at.run(timeout=30)
    radio = next(r for r in at.radio if r.key and r.key.endswith("-ai-kind"))
    radio.set_value("Sky")
    at.run(timeout=30)
    next(b for b in at.button if b.key and b.key.endswith("-ai-btn")).click()
    at.run(timeout=30)
    at.run(timeout=30)
    assert not at.exception
    selections = [layer for layer in at.session_state["mask_layers"] if layer.get("name", "").startswith("Sky")]
    assert selections and selections[0]["kind"] == "Painted" and "mask_ref" in selections[0]["params"]
