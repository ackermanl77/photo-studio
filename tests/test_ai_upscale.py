import numpy as np
import pytest
from PIL import Image

from app import ai_upscale


def test_ai_upscale_available_returns_a_bool_without_raising():
    """Real call, no mocking -- this test environment only installs requirements.txt (never
    requirements-gpu.txt, per CLAUDE.md's Commands section and CI's own install list), so
    ai_upscale_available() is expected to return False here, but the real contract this checks
    is that it NEVER raises regardless of whether torch/realesrgan are importable -- that's what
    lets the frontend and app.backend.process_image() both treat AI Upscale as a plain optional
    feature instead of something that can crash the rest of the app."""
    result = ai_upscale.ai_upscale_available()
    assert isinstance(result, bool)
    # Calling it again exercises the lru_cache path -- still must not raise.
    assert ai_upscale.ai_upscale_available() == result


def test_upscale_raises_a_clear_runtime_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(ai_upscale, "ai_upscale_available", lambda: False)
    image = Image.new("RGB", (8, 8), (100, 120, 140))

    with pytest.raises(RuntimeError, match="requirements-gpu.txt"):
        ai_upscale.upscale(image, 2)


def test_upscale_rejects_an_unsupported_factor(monkeypatch):
    """factor validation happens after the availability check but before any model loading --
    monkeypatches availability to True so this test isolates the factor check specifically."""
    monkeypatch.setattr(ai_upscale, "ai_upscale_available", lambda: True)
    image = Image.new("RGB", (8, 8), (100, 120, 140))

    with pytest.raises(ValueError, match="2 or 4"):
        ai_upscale.upscale(image, 3)


class _FakeUpsampler:
    """Stands in for RealESRGANer -- just enough of its interface (`.enhance(bgr, outscale=...)`
    returning `(array, None)`) for upscale() to run against, without downloading a real model or
    needing a GPU. Resizes via plain nearest-neighbor repeat so output size/content is
    deterministic and easy to assert on."""

    def enhance(self, bgr, outscale):
        resized = np.repeat(np.repeat(bgr, outscale, axis=0), outscale, axis=1)
        return resized, None


def test_upscale_delegates_to_the_model_and_resizes_by_the_requested_factor(monkeypatch):
    """Mocks the actual Real-ESRGAN inference (_load_upsampler) -- confirms upscale()'s own
    plumbing (BGR<->RGB channel handling, output size) is correct independent of whether the real
    optional deps are installed in whatever environment runs this suite."""
    monkeypatch.setattr(ai_upscale, "ai_upscale_available", lambda: True)
    monkeypatch.setattr(ai_upscale, "_load_upsampler", lambda: _FakeUpsampler())
    image = Image.new("RGB", (6, 4), (10, 20, 30))

    result = ai_upscale.upscale(image, 2)

    assert result.size == (12, 8)
    assert result.getpixel((0, 0)) == (10, 20, 30)


def test_upscale_4x_also_resizes_correctly(monkeypatch):
    monkeypatch.setattr(ai_upscale, "ai_upscale_available", lambda: True)
    monkeypatch.setattr(ai_upscale, "_load_upsampler", lambda: _FakeUpsampler())
    image = Image.new("RGB", (5, 3), (200, 50, 25))

    result = ai_upscale.upscale(image, 4)

    assert result.size == (20, 12)
    assert result.getpixel((0, 0)) == (200, 50, 25)
