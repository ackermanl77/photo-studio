import sys

import numpy as np
from PIL import Image

from app import cli
from app.image_processor import PhotoProcessor
from app.presets import apply_preset


def test_cli_applies_preset_in_the_same_relative_position_as_the_backend(tmp_path, monkeypatch):
    """Regression test: main() used to apply --preset last (after --crop/--aspect), while
    app/backend.py::process_image() applies a preset right after the Tone group (brightness/
    exposure/contrast/highlights-shadows) and before Color/Detail/Crop. A preset's own exposure/
    contrast/saturation/curve adjustments (app/presets.py) are order-sensitive relative to
    --contrast/--temperature here, so the two orders produced different pixel output for the same
    flags on the same source image via the CLI vs. the GUI/backend. This reproduces that
    combination and checks the CLI's actual output matches the "preset right after Tone" order,
    not the old "preset last" order."""
    input_path = tmp_path / "in.png"
    output_path = tmp_path / "out.png"
    Image.new("RGB", (16, 12), (120, 110, 130)).save(input_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--contrast", "1.4",
            "--temperature", "30",
            "--preset", "cinematic-soft",
        ],
    )
    cli.main()
    cli_result = np.array(Image.open(output_path).convert("RGB"))

    correct_order = PhotoProcessor(input_path)
    correct_order.apply_brightness(1.0).apply_contrast(1.4).apply_saturation(1.0).apply_exposure(1.0)
    correct_order.apply_highlights_shadows(0.0, 0.0)
    apply_preset(correct_order, "cinematic-soft")
    correct_order.apply_texture_clarity(0.0, 0.0)
    correct_order.apply_temperature_tint(30.0, 0.0)
    correct_order_result = np.array(correct_order.image.convert("RGB"))

    wrong_order = PhotoProcessor(input_path)
    wrong_order.apply_brightness(1.0).apply_contrast(1.4).apply_saturation(1.0).apply_exposure(1.0)
    wrong_order.apply_highlights_shadows(0.0, 0.0)
    wrong_order.apply_texture_clarity(0.0, 0.0)
    wrong_order.apply_temperature_tint(30.0, 0.0)
    apply_preset(wrong_order, "cinematic-soft")
    wrong_order_result = np.array(wrong_order.image.convert("RGB"))

    assert np.array_equal(cli_result, correct_order_result)
    assert not np.array_equal(cli_result, wrong_order_result)


def test_cli_applies_ai_upscale_last_after_crop(tmp_path, monkeypatch):
    """--ai-upscale should apply after every other adjustment including --crop, matching
    app/backend.py::process_image()'s pipeline position (AI Upscale runs last, after Effects).
    Mocks PhotoProcessor.apply_ai_upscale (no real GPU/model needed, and requirements-gpu.txt
    isn't installed in this test environment) and confirms it receives the already-cropped
    image's dimensions, not the original upload's."""
    input_path = tmp_path / "in.png"
    output_path = tmp_path / "out.png"
    Image.new("RGB", (20, 20), (100, 100, 100)).save(input_path)

    calls = []

    def fake_apply_ai_upscale(self, factor=2):
        calls.append((factor, self.image.size))
        self.image = self.image.resize((self.image.width * factor, self.image.height * factor))
        return self

    monkeypatch.setattr(PhotoProcessor, "apply_ai_upscale", fake_apply_ai_upscale)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--crop", "0", "0", "10", "10",
            "--ai-upscale", "4",
        ],
    )

    cli.main()

    assert calls == [(4, (10, 10))]  # ran after the crop had already shrunk the image
    assert Image.open(output_path).size == (40, 40)


def test_cli_ai_upscale_defaults_to_off():
    """--ai-upscale defaults to 0 (disabled) -- an expensive, optional operation opted into
    explicitly, matching the frontend control's own off-by-default behavior."""
    parser = cli.build_parser()
    args = parser.parse_args(["--input", "in.png", "--output", "out.png"])
    assert args.ai_upscale == 0
