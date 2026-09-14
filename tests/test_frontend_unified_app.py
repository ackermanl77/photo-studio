"""Regression tests for app/frontend_unified_app.py's tab composition.

Every tab in the unified workspace runs on every Streamlit rerun and feeds the same
accumulated `options` dict to app/backend.py's /process pipeline (see the module docstring /
"Recently Fixed" notes in docs/PROJECT_STATUS.md). Nothing exercised the composed script
directly before this file, which is exactly how Layers and Masks were once silently dropped
from the final image without any test catching it. These tests drive the real app through
Streamlit's AppTest harness instead of the individual processing functions in isolation.

The network call to the processing backend is stubbed (see `_stub_backend` below) so these
stay fast, offline unit tests; the stub still runs the real app.backend.process_image()
pipeline in-process, just without the HTTP/subprocess round trip, so a regression in how the
frontend assembles the request would still be caught.
"""

import io
import re
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from app import backend_client
from app.backend import process_image

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "frontend_unified_app.py"


def image_bytes(color=(90, 120, 160), size=(40, 30), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


class _FakeFileStorage:
    """Minimal stand-in for Flask's FileStorage -- just enough of its interface
    (`.filename`, `.read()`, `.stream`) for app.backend.process_image() to accept it."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.stream = io.BytesIO(content)

    def read(self) -> bytes:
        self.stream.seek(0)
        return self.stream.read()


def _fake_process_via_backend(image_bytes_, filename, options, local_mask_bytes=None, layer_bytes=None):
    request_options = dict(options)
    request_options["output_format"] = "PNG"
    image_file = _FakeFileStorage(filename, image_bytes_)
    local_mask = _FakeFileStorage("mask.png", local_mask_bytes) if local_mask_bytes else None
    layer_file = _FakeFileStorage("layer.png", layer_bytes) if layer_bytes else None
    return process_image(image_file, request_options, removal_mask=None, local_mask=local_mask, layer_file=layer_file)


@pytest.fixture(autouse=True)
def _stub_backend(monkeypatch):
    """Skip the real HTTP call / on-demand subprocess: app.frontend_unified_app imports
    `ensure_backend_running`/`process_via_backend` fresh from app.backend_client on every
    Streamlit rerun (`from ... import ...`), so patching them on the shared app.backend_client
    module object is picked up by every rerun without needing to reach into the app module."""
    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)


def run_with_uploaded_image(**upload_kwargs):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("photo.jpg", image_bytes(**upload_kwargs), "image/jpeg"))
    at.run(timeout=30)
    return at


def canvas_hash(at):
    """The media URL Streamlit assigns the main preview image -- distinct content gets a
    distinct URL, so comparing it before/after an edit is a cheap way to assert the edit
    actually reached the composed image without decoding pixels."""
    matches = [img for img in at.image if img.captions and img.captions[0] == "Current Edited Image"]
    assert matches, "expected the main canvas image to be rendered"
    return matches[0].value[0]


def canvas_dims(at):
    markdown = next(m.value for m in at.markdown if "Live View" in m.value)
    match = re.search(r"\((\d+)×(\d+)\)", markdown)
    assert match, f"could not find dimensions in: {markdown!r}"
    return int(match.group(1)), int(match.group(2))


def test_no_image_shows_upload_prompt_and_stops():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    assert any("Upload an image" in info.value for info in at.info)
    assert len(at.tabs) == 0


def test_uploading_image_renders_every_tab_without_errors():
    at = run_with_uploaded_image()

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "☼ Tone", "\U0001f3a8 Color", "\U0001f52c Detail", "◈ Layers",
        "◉ Masks", "◌ Portrait", "⛶ Crop", "✦ Presets", "☰ More",
    ]
    assert canvas_dims(at) == (40, 30)


def test_layer_composite_is_reflected_in_final_image():
    """Regression test for the bug where the Layers tab composited into a local variable
    that every later tab (and the final export) ignored -- see docs/PROJECT_STATUS.md's
    "Recently Fixed" entry. A layer that changes the composite must change what's shown."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.get_by_key("l-file").set_value(("overlay.png", image_bytes((255, 0, 0), (40, 30)), "image/png"))
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_mask_stack_target_adjustment_is_reflected_in_final_image():
    """Regression test for the bug where only one mask type could be active at a time --
    exercises the same MaskStack path (add a layer, then adjust Target Exposure) that the
    unified UI now uses."""
    at = run_with_uploaded_image(color=(80, 80, 80))
    baseline = canvas_hash(at)

    at.selectbox(key="m-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.button(key="m-add-btn").click()
    at.run(timeout=30)
    assert at.session_state["mask_layers"], "expected the mask layer to be added"

    at.slider(key="m-texp").set_value(1.5)
    at.run(timeout=30)

    assert not at.exception
    assert canvas_hash(at) != baseline


def test_crop_tab_changes_output_dimensions():
    at = run_with_uploaded_image(size=(40, 30))
    assert canvas_dims(at) == (40, 30)

    at.selectbox(key="crop-aspect").set_value("1:1 Square")
    at.run(timeout=30)

    assert not at.exception
    assert canvas_dims(at) == (30, 30)


def test_undo_restores_previous_control_value():
    at = run_with_uploaded_image()

    at.slider(key="t-exp").set_value(1.2)
    at.run(timeout=30)
    assert at.slider(key="t-exp").value == 1.2

    at.button(key="hist-undo-btn").click()
    at.run(timeout=30)

    assert not at.exception
    assert at.slider(key="t-exp").value == 1.0
