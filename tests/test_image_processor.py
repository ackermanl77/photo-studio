from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.batch import process_folder
from app.image_processor import PhotoProcessor
from app.presets import PRESETS, apply_preset
from app.export import save_for_delivery
from app.edit_history import EditHistory


def test_default_image_pipeline(tmp_path):
    source = tmp_path / "input.png"
    destination = tmp_path / "output.png"

    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(source)

    result = (
        PhotoProcessor(source)
        .apply_brightness(1.2)
        .apply_contrast(1.3)
        .apply_saturation(1.1)
    )

    output = result.save(destination)

    assert output.exists()
    assert output.suffix.lower() == ".png"
    img = Image.open(output)
    assert img.size == (10, 10)


def test_exposure_and_white_balance_controls(tmp_path):
    source = tmp_path / "portrait.png"
    destination = tmp_path / "enhanced.png"

    Image.new("RGB", (20, 20), color=(120, 110, 100)).save(source)

    result = (
        PhotoProcessor(source)
        .apply_exposure(1.2)
        .apply_white_balance(1.1, 0.0)
        .apply_denoise(8)
    )

    output = result.save(destination)

    assert output.exists()
    enhanced = np.array(Image.open(output))
    assert enhanced.shape[:2] == (20, 20)
    assert enhanced.dtype == np.uint8


def test_apply_hsl_mixer_delegates_to_professional_color(tmp_path):
    source = tmp_path / "red.png"
    Image.new("RGB", (12, 12), color=(220, 40, 40)).save(source)
    processor = PhotoProcessor(source)

    result = processor.apply_hsl_mixer({"Red": {"hue": 0, "sat": -100, "lum": 0}})

    assert result is processor
    desaturated = np.array(processor.image, dtype=np.float32)
    assert abs(int(desaturated[0, 0, 1]) - int(desaturated[0, 0, 2])) < 5  # closer to gray: G ~= B


def test_apply_color_grading_delegates_to_professional_color(tmp_path):
    source = tmp_path / "dark.png"
    Image.new("RGB", (12, 12), color=(20, 20, 20)).save(source)
    processor = PhotoProcessor(source)

    result = processor.apply_color_grading(shadows_hue=220, shadows_intensity=100)

    assert result is processor
    pixel = np.array(processor.image, dtype=np.int32)[0, 0]
    assert pixel[2] > pixel[0]  # tinted toward blue


def test_apply_dehaze_delegates_to_dehaze_module(tmp_path):
    source = tmp_path / "hazy.png"
    hazy = np.full((30, 30, 3), 190, dtype=np.uint8)
    hazy[8:22, 8:22] = 90
    Image.fromarray(hazy, "RGB").save(source)
    processor = PhotoProcessor(source)
    before = processor.as_numpy().copy()

    result = processor.apply_dehaze(90)

    assert result is processor
    assert not np.array_equal(before, processor.as_numpy())


def test_apply_denoise_actually_reduces_noise():
    """apply_denoise() switched from a flat median filter to OpenCV's Non-Local Means -- this
    confirms the new algorithm still does its actual job (reduces pixel-to-pixel variance on a
    noisy image) rather than just asserting it runs without error."""
    rng = np.random.default_rng(0)
    base = np.full((60, 60, 3), 128, dtype=np.float32)
    noisy = np.clip(base + rng.normal(0, 25, base.shape), 0, 255).astype(np.uint8)
    processor = PhotoProcessor.from_image(Image.fromarray(noisy, mode="RGB"))

    processor.apply_denoise(6)

    denoised = np.array(processor.image, dtype=np.float32)
    assert denoised.std() < noisy.astype(np.float32).std() * 0.5


def test_apply_denoise_color_radius_independently_changes_the_result():
    """color_radius drives fastNlMeansDenoisingColored's separate hColor parameter -- confirms it
    actually changes the result relative to leaving it at its 0 ("same strength as radius")
    default, not just that the parameter is accepted."""
    rng = np.random.default_rng(1)
    base = np.full((60, 60, 3), 128, dtype=np.float32)
    noisy = np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)

    default_result = PhotoProcessor.from_image(Image.fromarray(noisy, mode="RGB")).apply_denoise(4).as_numpy()
    color_boosted_result = PhotoProcessor.from_image(Image.fromarray(noisy, mode="RGB")).apply_denoise(4, 9).as_numpy()

    assert not np.array_equal(default_result, color_boosted_result)


def test_apply_post_crop_vignette_delegates_to_lens_corrections_module(tmp_path):
    source = tmp_path / "flat.png"
    Image.new("RGB", (60, 60), color=(200, 200, 200)).save(source)
    processor = PhotoProcessor(source)

    result = processor.apply_post_crop_vignette(-80, 30)

    assert result is processor
    pixels = np.array(processor.image, dtype=np.int32)
    assert pixels[:4, :4].mean() < pixels[28:32, 28:32].mean() - 20  # corners darkened


def test_apply_ai_upscale_delegates_to_ai_upscale_module(tmp_path, monkeypatch):
    """Mocks app.image_processor's own `ai_upscale` name (the module-level `from app.ai_upscale
    import upscale as ai_upscale` alias apply_ai_upscale() delegates into) rather than calling the
    real Real-ESRGAN inference -- this test doesn't need the optional torch/realesrgan packages or
    a GPU, matching every other delegation test in this file (e.g. apply_dehaze), just with a fake
    implementation standing in since the real one is an optional dependency."""
    from app import image_processor as image_processor_module

    calls = []

    def fake_ai_upscale(image, factor):
        calls.append(factor)
        return image.resize((image.width * factor, image.height * factor))

    monkeypatch.setattr(image_processor_module, "ai_upscale", fake_ai_upscale)

    source = tmp_path / "small.png"
    Image.new("RGB", (10, 8), color=(50, 60, 70)).save(source)
    processor = PhotoProcessor(source)

    result = processor.apply_ai_upscale(2)

    assert result is processor
    assert calls == [2]
    assert processor.image.size == (20, 16)


def test_apply_film_grain_delegates_to_effects_module(tmp_path):
    source = tmp_path / "flat.png"
    Image.new("RGB", (40, 40), color=(120, 120, 120)).save(source)
    processor = PhotoProcessor(source)
    before = processor.as_numpy().copy()

    result = processor.apply_film_grain(80)

    assert result is processor
    assert not np.array_equal(before, processor.as_numpy())


def test_apply_lens_corrections_delegates_to_lens_corrections_module(tmp_path):
    source = tmp_path / "grid.png"
    Image.new("RGB", (40, 40), color=(150, 150, 150)).save(source)
    processor = PhotoProcessor(source)

    result = processor.apply_lens_corrections(vignette_amount=-80, vignette_midpoint=30)

    assert result is processor
    pixels = np.array(processor.image, dtype=np.int32)
    assert pixels[:4, :4].mean() < pixels[18:22, 18:22].mean() - 10  # corners darkened


def test_apply_lens_corrections_passes_perspective_through(tmp_path):
    source = tmp_path / "grid.png"
    image = Image.new("RGB", (60, 60), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    for x in range(0, 60, 6):
        draw.line([(x, 0), (x, 60)], fill=(220, 220, 220))
    image.save(source)
    processor = PhotoProcessor(source)
    before = processor.as_numpy().copy()

    result = processor.apply_lens_corrections(perspective_vertical=60)

    assert result is processor
    assert not np.array_equal(before, processor.as_numpy())


def test_apply_vibrance_delegates_to_professional_color(tmp_path):
    source = tmp_path / "pastel.png"
    Image.new("RGB", (12, 12), color=(180, 150, 150)).save(source)
    processor = PhotoProcessor(source)
    before = processor.as_numpy().copy()

    result = processor.apply_vibrance(80)

    assert result is processor
    assert not np.array_equal(before, processor.as_numpy())


def test_auto_enhance_is_available(tmp_path):
    source = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(120, 110, 100)).save(source)
    processor = PhotoProcessor(source)
    processed = processor.auto_enhance()

    assert processed is processor
    assert processed.image.size[0] > 0


def test_named_preset_changes_the_image(tmp_path):
    source = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(120, 110, 100)).save(source)
    processor = PhotoProcessor(source)
    original = processor.as_numpy().copy()

    apply_preset(processor, "portrait-natural")

    assert "portrait-natural" in PRESETS
    assert not np.array_equal(original, processor.as_numpy())


def test_batch_processing_writes_supported_images(tmp_path):
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    Image.new("RGB", (8, 8), color=(100, 110, 120)).save(source_dir / "one.png")
    Image.new("RGB", (8, 8), color=(120, 110, 100)).save(source_dir / "two.jpg")
    (source_dir / "notes.txt").write_text("skip me", encoding="utf-8")

    outputs = process_folder(source_dir, output_dir, preset="travel-clean")

    assert [path.name for path in outputs] == ["one.png", "two.jpg"]
    assert all(path.exists() for path in outputs)


def test_delivery_export_converts_to_requested_format(tmp_path):
    destination = tmp_path / "delivery" / "final.jpg"
    source = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(120, 110, 100)).save(source)

    output = save_for_delivery(PhotoProcessor(source), destination, quality=90)

    assert output == destination
    assert Image.open(output).format == "JPEG"


def test_exposure_contrast_highlights_shadows_texture_and_clarity(tmp_path):
    source = tmp_path / "controls.png"
    Image.new("RGB", (20, 10), color=(90, 100, 110)).save(source)

    processor = PhotoProcessor(source)
    original = processor.as_numpy().copy()
    processor.apply_exposure(1.1).apply_contrast(1.2)
    processor.apply_highlights_shadows(-15, 20).apply_texture_clarity(30, 15)

    assert processor.image.size == (20, 10)
    assert not np.array_equal(original, processor.as_numpy())


def test_crop_and_aspect_ratio_framing(tmp_path):
    source = tmp_path / "framing.png"
    Image.new("RGB", (40, 20), color=(10, 20, 30)).save(source)

    processor = PhotoProcessor(source).crop(5, 2, 35, 18)
    assert processor.image.size == (30, 16)

    processor.fit_aspect_ratio(1, 1)
    assert processor.image.size == (16, 16)


def test_temperature_tint_and_curves_change_pixels(tmp_path):
    source = tmp_path / "color.png"
    Image.new("RGB", (12, 12), color=(100, 110, 120)).save(source)

    processor = PhotoProcessor(source)
    original = processor.as_numpy().copy()
    processor.apply_temperature_tint(30, -20).apply_curves([(0, 10), (128, 150), (255, 245)])

    assert processor.image.size == (12, 12)
    assert not np.array_equal(original, processor.as_numpy())


def test_object_removal_and_masked_adjustments(tmp_path):
    source = tmp_path / "scene.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (20, 20), color=(100, 100, 100)).save(source)
    mask = Image.new("L", (20, 20), color=0)
    mask.putpixel((10, 10), 255)
    mask.save(mask_path)

    processor = PhotoProcessor(source)
    original = processor.as_numpy().copy()
    processor.apply_masked_adjustments(mask_path, exposure=1.5)

    assert np.array_equal(original[0, 0], processor.as_numpy()[0, 0])
    assert not np.array_equal(original[10, 10], processor.as_numpy()[10, 10])
    processor.remove_objects(mask_path)
    assert processor.image.size == (20, 20)


def test_cinematic_preset_is_registered(tmp_path):
    source = tmp_path / "preset.png"
    Image.new("RGB", (12, 12), color=(100, 110, 120)).save(source)
    processor = PhotoProcessor(source)

    apply_preset(processor, "cinematic-soft")

    assert "cinematic-soft" in PRESETS
    assert processor.image.size == (12, 12)


def test_edit_history_supports_undo_redo_and_recipe_round_trip():
    history = EditHistory()
    history.record({"exposure": 1.0})
    history.record({"exposure": 1.2})
    history.record({"exposure": 1.4})

    assert history.undo() == {"exposure": 1.2}
    assert history.redo() == {"exposure": 1.4}

    restored = EditHistory.from_json(history.to_json())
    assert restored.current == {"exposure": 1.4}


def test_normal_image_loading_uses_raw_aware_adapter(tmp_path):
    source = tmp_path / "normal.jpg"
    Image.new("RGB", (12, 8), color=(100, 110, 120)).save(source)

    processor = PhotoProcessor(source)

    assert processor.image.size == (12, 8)
