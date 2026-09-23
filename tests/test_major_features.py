import sqlite3

import numpy as np
import pytest
from PIL import Image

from app.major_features import BLEND_MODE_LABELS, Layer, LayerStack, PhotoCatalog, heal_patch, preview_image


def test_layer_stack_composites_opacity_and_blend_modes():
    base = Image.new("RGB", (8, 8), (100, 100, 100))
    overlay = Image.new("RGBA", (8, 8), (200, 0, 0, 255))

    result = LayerStack(base).add(Layer("Color", overlay, opacity=0.5, blend_mode="normal")).composite()

    pixels = np.asarray(result)
    assert pixels.shape == (8, 8, 4)
    assert 140 <= int(pixels[4, 4, 0]) <= 160


def test_layer_masks_and_preview():
    base = Image.new("RGB", (20, 10), (100, 100, 100))
    overlay = Image.new("RGBA", (20, 10), (255, 0, 0, 255))
    mask = Image.new("L", (20, 10), 0)
    mask.putpixel((10, 5), 255)

    result = LayerStack(base).add(Layer("Local", overlay, mask=mask)).composite()

    assert result.size == (20, 10)
    assert preview_image(result, (10, 10)).size == (10, 5)


def _flat_blend(base_value: int, source_value: int, blend_mode: str) -> int:
    """Composite a flat base color against a flat, fully-opaque source color under one blend
    mode (opacity=1, no mask) and return the resulting single-channel value -- the simplest
    possible probe for a blend mode's own formula, independent of alpha/mask compositing."""
    base = Image.new("RGB", (2, 2), (base_value, base_value, base_value))
    overlay = Image.new("RGBA", (2, 2), (source_value, source_value, source_value, 255))
    result = LayerStack(base).add(Layer("Blend", overlay, blend_mode=blend_mode)).composite()
    return int(np.asarray(result)[0, 0, 0])


def test_blend_mode_registry_matches_labels_and_layer_stack_modes():
    """BLEND_MODE_LABELS is the frontend's single source of truth for "which modes exist and what
    do we call them" -- it must never drift from what LayerStack.MODES actually accepts, or the
    Layers panel could offer a mode that raises ValueError when applied, or vice versa."""
    assert set(BLEND_MODE_LABELS) == LayerStack.MODES


def test_darken_and_lighten_blend_modes_pick_the_lower_or_higher_value():
    assert _flat_blend(200, 80, "darken") == 80
    assert _flat_blend(200, 80, "lighten") == 200


def test_linear_dodge_and_linear_burn_blend_modes_add_or_subtract():
    assert 145 <= _flat_blend(100, 60, "linear_dodge") <= 165  # ~100+60=160
    assert 0 <= _flat_blend(100, 60, "linear_burn") <= 10  # 100+60-255 clamps to 0


def test_overlay_blend_mode_multiplies_below_midpoint_and_screens_above():
    # base=0.2 (51) is below the 0.5 midpoint -> overlay behaves like multiply: 2*0.2*0.8=0.32
    assert 70 <= _flat_blend(51, 204, "overlay") <= 90
    # base=0.8 (204) is above the midpoint -> overlay behaves like screen: 1-2*0.2*0.8=0.68
    assert 165 <= _flat_blend(204, 51, "overlay") <= 185


def test_hard_light_is_overlay_with_base_and_source_swapped():
    assert _flat_blend(51, 204, "overlay") == _flat_blend(204, 51, "hard_light")


def test_soft_light_blend_mode_matches_known_formula():
    # base=0.5, source=1.0 -> soft_light lightens toward sqrt(0.5) ~= 0.707 -> ~180
    assert 170 <= _flat_blend(128, 255, "soft_light") <= 190


def test_color_dodge_and_color_burn_blend_modes():
    assert _flat_blend(100, 255, "color_dodge") == 255  # dividing by (1-1)=0 clamps to white
    assert _flat_blend(100, 0, "color_burn") == 0  # dividing by 0 clamps to black


def test_difference_and_exclusion_blend_modes():
    assert _flat_blend(120, 120, "difference") == 0  # |b-s|=0 for identical colors
    assert _flat_blend(255, 255, "exclusion") == 0  # b+s-2bs=0 only at the extremes
    assert 120 <= _flat_blend(120, 120, "exclusion") <= 135  # softer than Difference: ~0.5 midtone, not 0
    assert _flat_blend(50, 200, "difference") == _flat_blend(200, 50, "difference")  # symmetric


def test_subtract_and_divide_blend_modes():
    assert 40 <= _flat_blend(150, 100, "subtract") <= 60  # 150-100=50
    assert _flat_blend(100, 0, "divide") == 255  # dividing by 0 clamps to white


def test_darker_color_and_lighter_color_compare_whole_pixel_not_per_channel():
    """Unlike Darken/Lighten (per-channel min/max), Darker/Lighter Color pick one whole source
    pixel or the other based on overall luminosity -- verified here with mixed-channel colors
    where a per-channel min/max would disagree with a whole-pixel comparison."""
    base = Image.new("RGB", (2, 2), (250, 10, 10))  # bright-looking (high R) but low luminosity
    overlay = Image.new("RGBA", (2, 2), (10, 200, 10, 255))  # greener, higher luminosity

    darker = np.asarray(LayerStack(base).add(Layer("B", overlay, blend_mode="darker_color")).composite())
    lighter = np.asarray(LayerStack(base).add(Layer("B", overlay, blend_mode="lighter_color")).composite())

    assert tuple(darker[0, 0, :3]) == (250, 10, 10)  # base has the lower luminosity -> kept whole
    assert tuple(lighter[0, 0, :3]) == (10, 200, 10)  # overlay has the higher luminosity -> kept whole


def test_color_and_hue_non_separable_blend_modes_keep_the_base_layers_luminosity():
    """Color/Hue take the source's hue+saturation but must preserve the base's own luminosity --
    the defining property of Photoshop's non-separable modes (e.g. "colorize a photo without
    changing its tonal range")."""
    base = Image.new("RGB", (2, 2), (30, 30, 30))  # dark, desaturated base
    overlay = Image.new("RGBA", (2, 2), (255, 0, 0, 255))

    def luminosity(pixel):
        r, g, b = pixel
        return 0.3 * r + 0.59 * g + 0.11 * b

    result = np.asarray(LayerStack(base).add(Layer("C", overlay, blend_mode="color")).composite())[0, 0, :3]

    assert abs(luminosity(result) - luminosity((30, 30, 30))) < 2
    assert result[0] > result[1]  # still reddish (took the overlay's hue), just at the base's dark luminosity


def test_saturation_and_luminosity_non_separable_blend_modes():
    base = Image.new("RGB", (2, 2), (200, 100, 100))  # saturated red-ish base
    grayscale_overlay = Image.new("RGBA", (2, 2), (128, 128, 128, 255))  # zero saturation

    result = np.asarray(LayerStack(base).add(Layer("S", grayscale_overlay, blend_mode="saturation")).composite())[0, 0, :3]

    assert int(result[0]) == int(result[1]) == int(result[2])  # borrowed the overlay's zero saturation


def test_every_registered_blend_mode_composites_without_error_and_stays_in_range():
    base = Image.new("RGB", (6, 6), (180, 90, 40))
    overlay = Image.new("RGBA", (6, 6), (40, 200, 210, 255))
    for mode in LayerStack.MODES:
        result = np.asarray(LayerStack(base).add(Layer("X", overlay, opacity=0.7, blend_mode=mode)).composite())
        assert result.shape == (6, 6, 4)
        assert result.dtype == np.uint8


def test_healing_patch_preserves_dimensions():
    image = Image.new("RGB", (40, 20), (120, 120, 120))

    result = heal_patch(image, (0, 0, 10, 10), (20, 5, 30, 15))

    assert result.size == image.size


def test_photo_catalog_supports_rating_tags_and_search(tmp_path):
    catalog = PhotoCatalog(tmp_path / "photos.sqlite")
    catalog.upsert("portraits/one.jpg", rating=5, tags=["portrait", "favorite"])
    catalog.upsert("travel/two.jpg", rating=3, tags=["travel"])

    results = catalog.search("portrait", minimum_rating=4)

    assert results == [{"path": "portraits/one.jpg", "rating": 5, "tags": ["favorite", "portrait"]}]


def test_photo_catalog_get_returns_a_default_for_an_unseen_path(tmp_path):
    catalog = PhotoCatalog(tmp_path / "photos.sqlite")
    catalog.upsert("portraits/one.jpg", rating=4, tags=["favorite"])

    assert catalog.get("portraits/one.jpg") == {"path": "portraits/one.jpg", "rating": 4, "tags": ["favorite"]}
    assert catalog.get("never/scanned.jpg") == {"path": "never/scanned.jpg", "rating": 0, "tags": []}


def test_photo_catalog_closes_every_connection_it_opens(tmp_path, monkeypatch):
    """Regression test: PhotoCatalog._connect() used to return a bare sqlite3.Connection used via
    `with self._connect() as connection:` -- a Connection's own context manager only commits or
    rolls back on exit, it does NOT close the connection, so every upsert()/get()/search() call
    leaked an open connection/file handle. Wraps sqlite3.connect() to track every connection
    opened through it and asserts each one was actually closed by the time this returns."""
    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)

    catalog = PhotoCatalog(tmp_path / "photos.sqlite")
    catalog.upsert("portraits/one.jpg", rating=5, tags=["favorite"])
    catalog.get("portraits/one.jpg")
    catalog.search("portrait")

    assert len(opened) >= 4  # __init__'s CREATE TABLE + upsert + get + search, each its own connection
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")  # a closed connection raises on any further use
