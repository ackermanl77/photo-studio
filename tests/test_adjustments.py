"""Photoshop's remaining Image > Adjustments (PS1, app.adjustments): each one's defining property,
the pipeline step in 8-bit and 16-bit mode, and that everything at its default is a no-op."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app import adjustments as adj
from app.edit_schema import parse_options
from app.pipeline import run_pipeline


def _ramp() -> np.ndarray:
    """A 16x16 grey ramp from black to white, as float RGB."""
    values = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    return np.repeat(values[..., None], 3, axis=2)


def _photo() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.random((12, 16, 3), dtype=np.float32)


def test_exposure_offset_and_gamma():
    ramp = _ramp()
    assert adj.exposure_offset_gamma(ramp, 0.0, 1.0) == pytest.approx(ramp)
    lifted = adj.exposure_offset_gamma(ramp, 0.1, 1.0)
    assert lifted[0, 0, 0] == pytest.approx(0.1) and lifted[-1, -1, 0] == pytest.approx(1.0)
    brighter = adj.exposure_offset_gamma(ramp, 0.0, 2.0)
    assert brighter[8, 0, 0] > ramp[8, 0, 0] and brighter[0, 0, 0] == 0.0 and brighter[-1, -1, 0] == pytest.approx(1.0)


def test_levels_maps_input_points_and_output_range():
    ramp = _ramp()
    result = adj.levels(ramp, 64, 192, 1.0, 0, 255)
    flat = result[..., 0].ravel()
    assert flat[:65].max() == 0.0 and flat[192:].min() == pytest.approx(1.0)  # clipped below/above the points
    faded = adj.levels(ramp, 0, 255, 1.0, 40, 200)
    assert faded.min() == pytest.approx(40 / 255) and faded.max() == pytest.approx(200 / 255)
    assert adj.levels(ramp, 0, 255, 2.0, 0, 255)[8, 0, 0] > ramp[8, 0, 0]  # gamma above 1 brightens midtones


def test_channel_mixer_identity_and_swap():
    photo = _photo()
    identity = adj.mixer_matrix({})
    assert adj.channel_mixer(photo, identity) == pytest.approx(photo)
    swapped = adj.mixer_matrix({"mixer_r_r": 0, "mixer_r_b": 100, "mixer_b_b": 0, "mixer_b_r": 100})
    result = adj.channel_mixer(photo, swapped)
    assert result[..., 0] == pytest.approx(photo[..., 2]) and result[..., 2] == pytest.approx(photo[..., 0])


def test_photo_filter_tints_and_can_keep_brightness():
    grey = np.full((4, 4, 3), 0.6, dtype=np.float32)
    warm = adj.photo_filter(grey, adj.PHOTO_FILTERS["Warming Filter (85)"], 50, preserve_luminosity=False)
    assert warm[0, 0, 0] > warm[0, 0, 2]  # orange: more red than blue
    kept = adj.photo_filter(grey, adj.PHOTO_FILTERS["Cooling Filter (80)"], 50, preserve_luminosity=True)
    assert kept[0, 0, 2] > kept[0, 0, 0]
    assert float(kept[0, 0] @ adj._LUMA) == pytest.approx(0.6, abs=1e-3)


def test_colorize_gives_one_hue_and_keeps_light_and_shade():
    import cv2

    result = adj.colorize(_photo(), hue=30, saturation=40, lightness=0)
    hls = cv2.cvtColor(result.astype(np.float32), cv2.COLOR_RGB2HLS)
    coloured = hls[..., 2] > 0.05
    assert np.allclose(hls[..., 0][coloured], 30, atol=1.5)
    assert np.corrcoef(result.mean(axis=2).ravel(), (_photo() @ adj._LUMA).ravel())[0, 1] > 0.95


def test_gradient_map_runs_from_the_shadow_to_the_highlight_colour():
    ramp = _ramp()
    mapped = adj.gradient_map(ramp, "#FF0000", "#0000FF", 100)
    assert mapped[0, 0] == pytest.approx([1, 0, 0]) and mapped[-1, -1] == pytest.approx([0, 0, 1], abs=1e-6)
    assert adj.gradient_map(ramp, "#FF0000", "#0000FF", 0) == pytest.approx(ramp)


def test_posterize_threshold_and_equalize():
    ramp = _ramp()
    assert len(np.unique(adj.posterize(ramp, 4))) == 4
    binary = adj.threshold(ramp, 128)
    assert set(np.unique(binary)) == {0.0, 1.0} and binary[0, 0, 0] == 0.0 and binary[-1, -1, 0] == 1.0
    dark = ramp * 0.3  # a dark, low-contrast photo
    spread = adj.equalize(dark)
    assert spread.max() == pytest.approx(1.0) and spread.min() == pytest.approx(0.0, abs=1e-3)


def test_defaults_do_nothing():
    assert not adj.adjustments_active(parse_options({}))
    photo = Image.fromarray((_photo() * 255).astype(np.uint8), mode="RGB")
    assert np.array_equal(np.asarray(run_pipeline(photo, parse_options({}))), np.asarray(run_pipeline(photo, parse_options({"levels_in_black": 40}))))


def test_the_pipeline_applies_them_in_8_and_16_bit():
    photo = Image.fromarray((_photo() * 255).astype(np.uint8), mode="RGB")
    options = parse_options({"threshold_enabled": True, "threshold_level": 128})
    result = np.asarray(run_pipeline(photo, options))
    assert set(np.unique(result)) <= {0, 255}
    high = run_pipeline(photo, parse_options({"threshold_enabled": True, "bit_depth": 16}))
    assert set(np.unique(np.asarray(high))) <= {0, 255}


def test_every_option_is_validated():
    with pytest.raises(ValueError):
        parse_options({"photo_filter": "Not a filter"})
    with pytest.raises(ValueError):
        parse_options({"levels_gamma": 20})
    assert parse_options({"photo_filter": "Sepia", "gradient_map_shadow": "#102030"})["photo_filter"] == "Sepia"
