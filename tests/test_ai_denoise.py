"""AI Denoise (app.ai_denoise): the SCUNet network, the per-input cache, the pipeline step with its
classical fallback, provenance and the CLI flag. The model's weights are only used by the opt-in
real-model test (PHOTO_AI_MODEL_TESTS=1; PHOTO_AI_DENOISE_WEIGHTS may point at a local copy)."""
from __future__ import annotations

import importlib.util
import os
import shutil

import numpy as np
import pytest
from PIL import Image

from app import ai_denoise
from app.pipeline import ai_tools_for, run_pipeline
from tests.synthetic_assets import photo_like

needs_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch (the GPU pack)")


@pytest.fixture(autouse=True)
def _empty_cache():
    ai_denoise._RESULTS.clear()
    yield
    ai_denoise._RESULTS.clear()


def _fake_infer(calls):
    def infer(rgb, between_tiles=None):
        calls.append(rgb.shape)
        return np.full_like(rgb, 0.5)
    return infer


@needs_torch
def test_the_network_matches_the_released_weights_layout():
    """The parameter names and shapes of KAIR's scunet_color_real_psnr.pth (540 tensors)."""
    state = ai_denoise._build_model().state_dict()
    assert len(state) == 540
    assert tuple(state["m_head.0.weight"].shape) == (64, 3, 3, 3)
    assert tuple(state["m_down1.0.trans_block.msa.relative_position_params"].shape) == (1, 15, 15)
    assert tuple(state["m_body.3.trans_block.msa.relative_position_params"].shape) == (8, 15, 15)
    assert tuple(state["m_up1.4.conv_block.2.weight"].shape) == (32, 32, 3, 3)
    assert tuple(state["m_tail.0.weight"].shape) == (3, 64, 3, 3)


@needs_torch
def test_the_network_keeps_any_size():
    import torch

    model = ai_denoise._build_model().eval()
    with torch.inference_mode():
        out = model(torch.rand(1, 3, 70, 93))
    assert tuple(out.shape) == (1, 3, 70, 93)


def test_tiles_cover_the_image_and_meet_seamlessly(monkeypatch):
    """_infer keeps each tile's core: an identity model must give back the input exactly."""
    torch = pytest.importorskip("torch")

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    monkeypatch.setattr(ai_denoise, "_load_model", lambda: (Identity(), "cpu"))
    monkeypatch.setattr(ai_denoise, "_TILE", 48)
    monkeypatch.setattr(ai_denoise, "_TILE_PAD", 8)
    rgb = np.random.default_rng(1).random((101, 130, 3), dtype=np.float32)
    ticks = []
    assert np.array_equal(ai_denoise._infer(rgb, lambda: ticks.append(1)), rgb)
    assert len(ticks) == len(ai_denoise._windows(101 + 16)) * len(ai_denoise._windows(130 + 16))


@pytest.mark.parametrize("length", [5, 48, 49, 100, 117, 1067])
def test_tile_windows_are_full_size_and_keep_every_pixel_once(monkeypatch, length):
    monkeypatch.setattr(ai_denoise, "_TILE", 48)
    monkeypatch.setattr(ai_denoise, "_TILE_PAD", 8)
    windows = ai_denoise._windows(length)
    assert windows[0][2] == 0 and windows[-1][3] == length
    for (start, end, keep_from, keep_to), following in zip(windows, windows[1:] + [None]):
        assert end - start == min(48, length) and start <= keep_from < keep_to <= end
        if following is not None:
            assert following[2] == keep_to


def test_one_inference_per_input(monkeypatch):
    calls = []
    monkeypatch.setattr(ai_denoise, "_infer", _fake_infer(calls))
    rgb = np.random.default_rng(2).random((20, 30, 3), dtype=np.float32)
    first = ai_denoise.denoise(rgb)
    assert ai_denoise.denoise(rgb.copy()) is first and len(calls) == 1
    ai_denoise.denoise(rgb[:10])
    assert len(calls) == 2


def test_blend_amount():
    original, denoised = np.zeros((2, 2, 3), np.float32), np.ones((2, 2, 3), np.float32)
    assert np.allclose(ai_denoise.blend(original, denoised, 100), 1.0)
    assert np.allclose(ai_denoise.blend(original, denoised, 25), 0.25)
    assert np.allclose(ai_denoise.blend(original, denoised, 0), 0.0)


def test_pipeline_applies_it_first_and_blends(monkeypatch):
    calls = []
    monkeypatch.setattr(ai_denoise, "ai_denoise_available", lambda: True)
    monkeypatch.setattr(ai_denoise, "_infer", _fake_infer(calls))
    image = photo_like((64, 48))
    full = np.asarray(run_pipeline(image, {"ai_denoise": True}), np.int16)
    assert np.all(np.abs(full - 128) <= 1)
    half = np.asarray(run_pipeline(image, {"ai_denoise": True, "ai_denoise_amount": 50}), np.int16)
    expected = (np.asarray(image.convert("RGB"), np.float32) + 127.5) / 2
    assert np.all(np.abs(half - expected) <= 1)
    assert len(calls) == 1  # the second run reused the cached result
    # It sees the photo before Geometry: a flipped edit denoises the same, unflipped input.
    run_pipeline(image, {"ai_denoise": True, "flip_horizontal": True})
    assert len(calls) == 1
    # Off, or at 0, it does nothing.
    assert np.array_equal(np.asarray(run_pipeline(image, {"ai_denoise": True, "ai_denoise_amount": 0})), np.asarray(run_pipeline(image, {})))


def test_pipeline_keeps_16_bit_precision(monkeypatch):
    from app.high_bit import attached

    monkeypatch.setattr(ai_denoise, "ai_denoise_available", lambda: True)
    monkeypatch.setattr(ai_denoise, "_infer", lambda rgb, between_tiles=None: np.clip(rgb * 0.5 + 0.1234, 0, 1))
    result = attached(run_pipeline(photo_like((32, 24)), {"ai_denoise": True, "bit_depth": 16}))
    assert result is not None and result.dtype == np.float32
    assert len(np.unique(np.round(result * 65535))) > len(np.unique(np.round(result * 255)))


def test_without_torch_the_classical_denoise_stands_in(monkeypatch):
    monkeypatch.setattr(ai_denoise, "ai_denoise_available", lambda: False)
    monkeypatch.setattr(ai_denoise, "_infer", lambda *args: pytest.fail("the model must not run"))
    image = photo_like((64, 48))
    fallback = run_pipeline(image, {"ai_denoise": True})
    classical = run_pipeline(image, {"denoise": ai_denoise.FALLBACK_DENOISE})
    assert np.array_equal(np.asarray(fallback), np.asarray(classical))


def test_provenance_names_it_only_when_it_can_run(monkeypatch):
    monkeypatch.setattr(ai_denoise, "ai_denoise_installed", lambda: True)
    assert ai_tools_for({"ai_denoise": True}) == ["AI Denoise"]
    assert ai_tools_for({"ai_denoise": True, "ai_denoise_amount": 0}) == []
    monkeypatch.setattr(ai_denoise, "ai_denoise_installed", lambda: False)
    assert ai_tools_for({"ai_denoise": True}) == []


def test_cli_flag():
    from app.cli import build_parser, options_from_args

    options = options_from_args(build_parser().parse_args(["--input", "a.jpg", "--output", "b.jpg", "--ai-denoise", "70"]))
    assert options["ai_denoise"] is True and options["ai_denoise_amount"] == 70


def test_weights_check_does_not_download(tmp_path):
    assert not ai_denoise.weights_present()
    assert ai_denoise.weights_path().parent.name == "weights"


@needs_torch
@pytest.mark.skipif(os.environ.get("PHOTO_AI_MODEL_TESTS") != "1", reason="set PHOTO_AI_MODEL_TESTS=1 (downloads ~72 MB)")
def test_the_real_model_removes_camera_like_noise():
    local = os.environ.get("PHOTO_AI_DENOISE_WEIGHTS")
    if local:
        ai_denoise.weights_path().parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, ai_denoise.weights_path())
    ai_denoise._load_model.cache_clear()
    clean = np.asarray(photo_like((480, 320)), np.float32) / 255
    rng = np.random.default_rng(0)
    noisy = np.clip(rng.poisson(clean * 60) / 60 + rng.normal(0, 3 / 255, clean.shape), 0, 1).astype(np.float32)

    def psnr(a):
        return 10 * np.log10(1 / np.mean((a - clean) ** 2))

    assert psnr(ai_denoise.denoise(noisy)) > psnr(noisy) + 8
    image = Image.fromarray((noisy * 255).round().astype(np.uint8))
    classical = np.asarray(run_pipeline(image, {"denoise": 5}), np.float32) / 255
    assert psnr(np.asarray(run_pipeline(image, {"ai_denoise": True}), np.float32) / 255) > psnr(classical)
