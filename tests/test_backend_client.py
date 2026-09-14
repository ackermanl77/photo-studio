from __future__ import annotations

import io
import socket
import sys
import threading
import time

import pytest
from PIL import Image

from app import backend_client


def _png_bytes(color=(90, 120, 160), size=(16, 16)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _reset_backend_running_cache():
    """ensure_backend_running is cached with st.cache_resource so it only ever runs once per
    process -- clear that cache around each test so every test actually exercises the function
    body instead of a stale cached result from a previous test."""
    backend_client.ensure_backend_running.clear()
    yield
    backend_client.ensure_backend_running.clear()


def test_ensure_backend_running_does_not_spawn_when_already_healthy(monkeypatch):
    monkeypatch.setattr(backend_client, "_backend_is_healthy", lambda: True)
    spawned = []
    monkeypatch.setattr(backend_client.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))

    backend_client.ensure_backend_running()

    assert spawned == []


def test_ensure_backend_running_spawns_and_waits_for_health(monkeypatch):
    calls = {"count": 0}

    def fake_healthy():
        calls["count"] += 1
        return calls["count"] > 2  # unhealthy for the first couple checks, then healthy

    spawned = []
    monkeypatch.setattr(backend_client, "_backend_is_healthy", fake_healthy)
    monkeypatch.setattr(backend_client.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))
    monkeypatch.setattr(backend_client.time, "sleep", lambda seconds: None)

    backend_client.ensure_backend_running()

    assert len(spawned) == 1
    assert spawned[0][0][0] == [sys.executable, "-m", "app.backend"]


def test_ensure_backend_running_raises_if_it_never_comes_up(monkeypatch):
    monkeypatch.setattr(backend_client, "_backend_is_healthy", lambda: False)
    monkeypatch.setattr(backend_client.subprocess, "Popen", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_client, "_STARTUP_TIMEOUT_SECONDS", 0)

    with pytest.raises(RuntimeError):
        backend_client.ensure_backend_running()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def live_backend(monkeypatch):
    """Runs the real Flask app (app.backend.app) on a background thread against an ephemeral
    port, so process_via_backend() is exercised against an actual HTTP server instead of Flask's
    in-process test client."""
    from app.backend import app as flask_app

    port = _free_port()
    monkeypatch.setattr(backend_client, "BACKEND_URL", f"http://127.0.0.1:{port}")

    thread = threading.Thread(
        target=lambda: flask_app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if backend_client._backend_is_healthy():
            break
        time.sleep(0.1)
    else:
        pytest.fail("test backend did not become healthy in time")

    yield port


def test_process_via_backend_returns_processed_image(live_backend):
    result = backend_client.process_via_backend(_png_bytes(size=(16, 16)), "input.png", {"exposure": 1.0})

    assert result.size == (16, 16)


def test_process_via_backend_sends_mask_and_layer_files(live_backend):
    base = _png_bytes((40, 40, 40), (12, 12))
    layer = _png_bytes((220, 30, 30), (12, 12))
    mask_buffer = io.BytesIO()
    Image.new("L", (12, 12), 255).save(mask_buffer, format="PNG")

    baseline = backend_client.process_via_backend(base, "input.png", {})
    with_extras = backend_client.process_via_backend(
        base, "input.png", {"layer_opacity": 1.0, "layer_blend": "normal", "mask_exposure": 0.5},
        local_mask_bytes=mask_buffer.getvalue(), layer_bytes=layer,
    )

    assert with_extras.size == (12, 12)
    assert list(baseline.getdata()) != list(with_extras.getdata())


def test_process_via_backend_raises_runtime_error_on_backend_error(live_backend):
    with pytest.raises(RuntimeError, match="Unknown preset"):
        backend_client.process_via_backend(_png_bytes(), "input.png", {"preset": "does-not-exist"})
