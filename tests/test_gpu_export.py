from __future__ import annotations

import io
import logging

import numpy as np
import pytest
from PIL import Image

from app import export
from app.cli import build_parser
from app.high_bit import attach


def _gradient(width=80, height=48):
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    red, green = np.meshgrid(x, y)
    pixels = np.stack((red, green, (red + green) / 2), axis=-1)
    return Image.fromarray(np.rint(pixels * 255).astype(np.uint8))


def test_gpu_export_is_opt_in_and_validated():
    assert export.validate_export_settings({})["gpu_export"] is False
    assert export.validate_export_settings({"gpu_export": True})["gpu_export"] is True
    with pytest.raises(ValueError, match="gpu_export must be true or false"):
        export.validate_export_settings({"gpu_export": 1})


def test_cli_exposes_an_opt_in_gpu_export_flag():
    args = build_parser().parse_args(["--input", "in.jpg", "--output", "out.jpg", "--gpu-export"])
    assert args.gpu_export is True
    args = build_parser().parse_args(["--input", "in.jpg", "--output", "out.jpg"])
    assert args.gpu_export is False


def test_eight_bit_export_uses_gpu_transform_when_selected(monkeypatch):
    image = _gradient()
    calls = []

    def fake_gpu(pixels, output_size, sharpening):
        calls.append((pixels.shape, output_size, sharpening))
        resized = Image.fromarray(pixels).resize(output_size, Image.Resampling.BICUBIC)
        return np.asarray(resized)

    monkeypatch.setattr(export, "gpu_export_transform", fake_gpu)
    content, extension, mime = export.encode_export(
        image, {"format": "PNG", "long_edge": 40, "sharpen": "Screen", "gpu_export": True}
    )

    assert calls == [((48, 80, 3), (40, 24), export.OUTPUT_SHARPENING["Screen"])]
    assert Image.open(io.BytesIO(content)).size == (40, 24)
    assert (extension, mime) == (".png", "image/png")


def test_sixteen_bit_export_uses_gpu_transform_and_keeps_16_bit_output(monkeypatch):
    image = _gradient()
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    attach(image, pixels)
    calls = []

    def fake_gpu(array, output_size, sharpening):
        calls.append((array.dtype, output_size, sharpening))
        return np.full((output_size[1], output_size[0], 3), 0.5, dtype=np.float32)

    monkeypatch.setattr(export, "gpu_export_transform", fake_gpu)
    content, _extension, _mime = export.encode_export(
        image,
        {"format": "PNG", "bit_depth": 16, "long_edge": 40, "sharpen": "Matte paper", "gpu_export": True},
    )

    assert calls == [(np.dtype(np.float32), (40, 24), export.OUTPUT_SHARPENING["Matte paper"])]
    import cv2

    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (24, 40, 3) and decoded.dtype == np.uint16


def test_unavailable_cuda_falls_back_to_the_cpu_export(caplog, monkeypatch):
    from app import gpu_export

    image = _gradient()
    cpu_content = export.encode_export(image, {"format": "PNG", "long_edge": 40, "sharpen": "Screen"})[0]
    monkeypatch.setattr(gpu_export, "resize_and_sharpen", lambda *_args: None)

    with caplog.at_level(logging.WARNING, logger="app.export"):
        gpu_requested_content = export.encode_export(
            image, {"format": "PNG", "long_edge": 40, "sharpen": "Screen", "gpu_export": True}
        )[0]

    assert gpu_requested_content == cpu_content
    assert "using the CPU export path" in caplog.text


def test_cuda_operation_error_falls_back_to_the_cpu_export(caplog, monkeypatch):
    from app import gpu_export

    image = _gradient()
    cpu_content = export.encode_export(image, {"format": "PNG", "long_edge": 40})[0]

    def fail_gpu(*_args):
        raise RuntimeError("CUDA allocation failed")

    monkeypatch.setattr(gpu_export, "resize_and_sharpen", fail_gpu)
    with caplog.at_level(logging.WARNING, logger="app.export"):
        gpu_requested_content = export.encode_export(
            image, {"format": "PNG", "long_edge": 40, "gpu_export": True}
        )[0]

    assert gpu_requested_content == cpu_content
    assert "CUDA export failed; using the CPU export path" in caplog.text


def test_cuda_transform_returns_none_when_torch_or_cuda_is_unavailable(monkeypatch):
    import sys
    import types

    from app.gpu_export import resize_and_sharpen

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", types.ModuleType("torch.nn"))
    monkeypatch.setitem(sys.modules, "torch.nn.functional", types.ModuleType("torch.nn.functional"))

    assert resize_and_sharpen(np.zeros((8, 8, 3), dtype=np.uint8), (4, 4), None) is None
