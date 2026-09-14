from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.edit_history import EditHistory
from app.export import save_for_delivery
from app.image_processor import PhotoProcessor
from app.presets import apply_preset
from app.project import PhotoProject

ASSET_DIR = Path(__file__).parent / "assets"
FEATURE_CASES = [
    ("154446-naruto-naruto-wallpaper.jpg", "exposure"),
    ("850008.jpg", "contrast"),
    ("9.jpg", "highlights-shadows"),
    ("9i-QVZzaSIVuIJgALYY7upwbLflzGh-bk5Ex1S0uthE.jpg", "texture-clarity"),
    ("download.jpg", "temperature-tint"),
    ("dQM5l3.jpg", "curves"),
    ("eIVhqz.jpg", "crop"),
    ("FB_IMG_1529372941771.jpg", "preset"),
    ("Levi-Ackerman-shingeki-no-kyojin-attack-on-titan-37276165-500-379.png", "masking"),
]

# The repository's sample.png is user-modified and currently is not a readable image.
@pytest.mark.parametrize("filename,feature", FEATURE_CASES)
def test_real_asset_feature_smoke(tmp_path, filename, feature):
    source = ASSET_DIR / filename
    processor = PhotoProcessor(source)
    original_size = processor.image.size

    if feature == "exposure":
        processor.apply_exposure(1.08)
    elif feature == "contrast":
        processor.apply_contrast(1.12)
    elif feature == "highlights-shadows":
        processor.apply_highlights_shadows(-12, 18)
    elif feature == "texture-clarity":
        processor.apply_texture_clarity(20, 10)
    elif feature == "temperature-tint":
        processor.apply_temperature_tint(12, -5)
    elif feature == "curves":
        processor.apply_curves([(0, 8), (128, 138), (255, 248)])
    elif feature == "crop":
        processor.fit_aspect_ratio(1, 1)
    elif feature == "preset":
        apply_preset(processor, "cinematic-soft")
    elif feature == "masking":
        mask = np.zeros((processor.image.height, processor.image.width), dtype=np.uint8)
        mask[processor.image.height // 4 : processor.image.height // 2, processor.image.width // 4 : processor.image.width // 2] = 255
        processor.apply_masked_adjustments(mask, exposure=1.08)

    output = tmp_path / f"{feature}.jpg"
    save_for_delivery(processor, output, quality=88, max_size=(1600, 1600))
    result = Image.open(output)

    assert result.format == "JPEG"
    assert result.width > 0 and result.height > 0
    if feature == "crop":
        assert result.width == result.height
    else:
        assert processor.image.size == original_size


def test_real_assets_are_readable_except_known_modified_sample():
    unreadable = []
    for asset in ASSET_DIR.iterdir():
        if asset.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
            continue
        try:
            with Image.open(asset) as image:
                image.verify()
        except Exception:
            unreadable.append(asset.name)

    assert unreadable == ["sample.png"]


def test_real_asset_project_and_recipe_round_trip(tmp_path):
    source = ASSET_DIR / "FB_IMG_1539485356370.jpg"
    source_bytes = source.read_bytes()
    recipe = {"preset": "portrait-natural", "exposure": 1.1}
    project_path = tmp_path / "real-image.photoproject"

    PhotoProject(source.name, source_bytes, recipe).save(project_path)
    restored = PhotoProject.load(project_path)
    history = EditHistory()
    history.record(restored.recipe)

    assert restored.source_bytes == source_bytes
    assert history.current == recipe
