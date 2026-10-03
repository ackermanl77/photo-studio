"""Feature smoke tests on photo-like inputs, plus decode tests for the formats that need special
handling (16-bit grayscale, camera RAW).

These used to run on copyrighted wallpapers and personal Facebook downloads committed under
tests/assets, plus a `sample.png` whose bytes had been mangled into an unreadable file that one test
asserted *stayed* unreadable. Every input is now generated (tests/synthetic_assets.py), so nothing
here depends on third-party images (T3/T6 in the improvement report).
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.edit_history import EditHistory
from app.export import write_export
from app.image_processor import PhotoProcessor
from app.pipeline import decode_source
from app.presets import apply_preset
from tests.legacy_project import PhotoProject
from app.raw_import import SUPPORTED_EXTENSIONS, open_bytes, uploader_types
from tests.synthetic_assets import dng_bytes, encode, gray16_png, photo_like

ASSET_DIR = Path(__file__).parent / "assets"
FEATURES = ["exposure", "contrast", "highlights-shadows", "texture-clarity", "temperature-tint", "curves", "crop", "preset", "masking"]


@pytest.mark.parametrize("feature", FEATURES)
def test_feature_smoke_on_a_photo_like_image(tmp_path, feature):
    source = tmp_path / "photo.jpg"
    source.write_bytes(encode(photo_like((400, 300), seed=FEATURES.index(feature)), "JPEG", quality=90))
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
    write_export(processor.image, output, {"quality": 88, "long_edge": 1600})
    result = Image.open(output)

    assert result.format == "JPEG"
    assert result.width > 0 and result.height > 0
    if feature == "crop":
        assert result.width == result.height
    else:
        assert processor.image.size == original_size


def test_no_test_reads_any_binary_fixture_but_the_public_domain_portrait():
    """No test may depend on a binary fixture in tests/assets other than the public-domain
    portrait (the old third-party images were deleted; this keeps new ones from creeping back)."""
    allowed = {"portrait_public_domain.jpg"}
    fixtures = {path.name for path in ASSET_DIR.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS} - allowed
    for test_file in Path(__file__).parent.glob("*.py"):
        text = test_file.read_text(encoding="utf-8")
        if "assets" not in text:
            continue  # e.g. a test that writes its own image of the same name into tmp_path
        used = sorted(name for name in fixtures if f'"{name}"' in text)
        assert not used, f"{test_file.name} still reads {used}"
    with Image.open(ASSET_DIR / "portrait_public_domain.jpg") as image:
        image.verify()


def test_project_and_recipe_round_trip(tmp_path):
    source_bytes = encode(photo_like(), "JPEG")
    recipe = {"preset": "portrait-natural", "exposure": 1.1}
    project_path = tmp_path / "photo.photoproject"

    PhotoProject("photo.jpg", source_bytes, recipe).save(project_path)
    restored = PhotoProject.load(project_path)
    history = EditHistory()
    history.record(restored.recipe)

    assert restored.source_bytes == source_bytes
    assert history.current == recipe


# --- B7: 16-bit grayscale ---------------------------------------------------------------------------

def test_16_bit_grayscale_png_decodes_as_a_gradient_not_white():
    image = open_bytes(gray16_png(), "scan.png")
    row = np.asarray(image)[0, :, 0].astype(int)
    assert image.mode == "RGB"
    assert row[0] <= 1 and row[-1] >= 254
    assert np.all(np.diff(row) >= 0)  # monotonic, like the source gradient
    assert (np.asarray(image) == 255).mean() < 0.02  # used to be ~99.6 % white


def test_16_bit_grayscale_tiff_decodes_through_the_pipeline():
    gradient = np.tile(np.linspace(0, 65535, 200).astype(np.uint16), (10, 1))
    content = encode(Image.fromarray(gradient), "TIFF")
    image = decode_source(content, "scan.tif")
    values = np.asarray(image)[:, :, 0]
    assert 120 <= values.mean() <= 135


# --- R3: one supported-format list ------------------------------------------------------------------

def test_every_raw_brand_is_in_the_uploader_list():
    types = uploader_types()
    for extension in ("raf", "rw2", "orf", "pef", "srw", "cr3", "dng", "jpg", "tiff", "webp"):
        assert extension in types
    assert "raf" not in uploader_types(include_raw=False)


# --- T2: real RAW decode (needs the optional rawpy; CI runs it in its own job) ----------------------

def test_a_real_dng_decodes_through_libraw():
    pytest.importorskip("rawpy")
    image = decode_source(dng_bytes(), "tiny.dng")
    array = np.asarray(image).astype(int)
    assert image.size == (64, 48)
    # The synthetic sensor data has red rising left to right; the demosaiced image must too.
    assert array[:, -8:, 0].mean() > array[:, :8, 0].mean() + 20
