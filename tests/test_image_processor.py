from pathlib import Path

import numpy as np
from PIL import Image

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
