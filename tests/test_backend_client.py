from __future__ import annotations

import io
import socket
import sys
import threading
import time

import pytest
from PIL import Image

from app import backend_client
from app.layer_workspace import WorkspaceLayer, layers_to_json


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
    monkeypatch.setattr(backend_client, "_backend_accepts_our_token", lambda: True)
    spawned = []
    monkeypatch.setattr(backend_client.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))

    backend_client.ensure_backend_running()

    assert spawned == []


def test_a_backend_that_rejects_our_token_gets_a_sibling_on_a_free_port(monkeypatch):
    """Another session's backend (or one an earlier run left behind) answers /health but 401s our
    token: start our own on a free port instead of failing every edit with 'invalid backend token'."""
    monkeypatch.setattr(backend_client, "BACKEND_URL", "http://127.0.0.1:8502")
    monkeypatch.setenv("PHOTO_BACKEND_PORT", "8502")
    monkeypatch.setattr(backend_client, "_backend_is_healthy", lambda: True)
    monkeypatch.setattr(backend_client, "_backend_accepts_our_token", lambda: False)
    monkeypatch.setattr(backend_client, "_free_port", lambda: 54321)
    spawned = []
    monkeypatch.setattr(backend_client.subprocess, "Popen", lambda *args, **kwargs: spawned.append(kwargs["env"]))

    backend_client.ensure_backend_running()

    assert len(spawned) == 1
    assert spawned[0]["PHOTO_BACKEND_PORT"] == "54321"
    assert backend_client.BACKEND_URL == "http://127.0.0.1:54321"


def test_the_token_probe_reads_401_as_someone_elses_backend(monkeypatch):
    class Reply:
        def __init__(self, status):
            self.status_code = status

    monkeypatch.setattr(backend_client.requests, "post", lambda *a, **k: Reply(401))
    assert backend_client._backend_accepts_our_token() is False
    monkeypatch.setattr(backend_client.requests, "post", lambda *a, **k: Reply(400))
    assert backend_client._backend_accepts_our_token() is True


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
    layers_payload = layers_to_json([WorkspaceLayer("Overlay", Image.new("RGBA", (12, 12), (220, 30, 30, 255)))]).encode("utf-8")
    mask_buffer = io.BytesIO()
    Image.new("L", (12, 12), 255).save(mask_buffer, format="PNG")

    baseline = backend_client.process_via_backend(base, "input.png", {})
    with_extras = backend_client.process_via_backend(
        base, "input.png", {"mask_exposure": 0.5},
        local_mask_bytes=mask_buffer.getvalue(), layers_bytes=layers_payload,
    )

    assert with_extras.size == (12, 12)
    assert baseline.tobytes() != with_extras.tobytes()


def test_process_via_backend_raises_runtime_error_on_backend_error(live_backend):
    with pytest.raises(RuntimeError, match="Unknown preset"):
        backend_client.process_via_backend(_png_bytes(), "input.png", {"preset": "does-not-exist"})


class _FakeResponse:
    ok = True

    def __init__(self, content: bytes):
        self.content = content


@pytest.mark.parametrize(
    "options, expected_timeout",
    [
        ({"exposure": 1.1}, backend_client.REQUEST_TIMEOUT_S),
        ({"ai_upscale": False}, backend_client.REQUEST_TIMEOUT_S),
        ({"ai_upscale": True, "ai_upscale_factor": 2}, backend_client.AI_UPSCALE_REQUEST_TIMEOUT_S),
    ],
)
def test_process_via_backend_gives_ai_upscale_a_longer_timeout(monkeypatch, options, expected_timeout):
    """AI Upscale's first request loads torch and downloads its model before inferring (~20 s on a
    fast connection in the packaged build), so it must not share the short default timeout."""
    seen = {}

    def fake_post(url, data=None, files=None, headers=None, timeout=None):
        seen["timeout"] = timeout
        return _FakeResponse(_png_bytes())

    monkeypatch.setattr(backend_client.requests, "post", fake_post)

    backend_client.process_via_backend(_png_bytes(), "in.png", options)

    assert seen["timeout"] == expected_timeout
    assert backend_client.AI_UPSCALE_REQUEST_TIMEOUT_S > backend_client.REQUEST_TIMEOUT_S
