"""Keeps the suite from touching the real per-user data folder.

Every per-user file (presets, the library catalog, logs, weights) lives under
app.config.user_data_dir() -- %LOCALAPPDATA%\\PhotoStudio on Windows. The session-wide override is
set at import time, before any test module imports app.backend (whose audit logger opens its file
at import); the per-test one points each test at `<tmp_path>/.photo-studio`, which is also where
the older tests that redirect Path.home() to tmp_path already look for presets.json/catalog.sqlite.
"""
import os
import tempfile

import pytest

os.environ.setdefault("PHOTO_STUDIO_DATA_DIR", tempfile.mkdtemp(prefix="photo-studio-tests-"))


os.environ.pop("PHOTO_BACKEND_TOKEN", None)


@pytest.fixture(autouse=True)
def _isolated_user_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOTO_STUDIO_DATA_DIR", str(tmp_path / ".photo-studio"))
    # No backend token unless a test sets one -- and one set during a test (the frontend generates
    # one when it spawns a backend) is removed again afterwards instead of leaking into later tests.
    monkeypatch.setenv("PHOTO_BACKEND_TOKEN", "")


@pytest.fixture(autouse=True)
def _no_refine_on_pause(monkeypatch):
    """The proxy preview's refine-on-pause (app.refine_preview) starts a background full-resolution
    render after a real-time pause -- nondeterministic under AppTest. Off unless a test enables it."""
    from app import refine_preview

    monkeypatch.setattr(refine_preview, "PAUSE_SECONDS", float("inf"))
