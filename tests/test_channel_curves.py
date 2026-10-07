"""F13: per-channel R/G/B tone curves (the "curve_channels" option)."""
from __future__ import annotations

import numpy as np
import pytest

from app.controls import curve_channels_from
from app.edit_schema import OptionsError, parse_options, recipe_defaults
from app.pipeline import run_pipeline
from app.recipe_options import recipe_to_options
from tests.synthetic_assets import photo_like


def test_a_channel_curve_changes_only_its_own_channel():
    image = photo_like()
    base = np.asarray(run_pipeline(image, {}), dtype=np.int16)
    lifted = np.asarray(run_pipeline(image, {"curve_channels": {"b": [[0, 40], [128, 170], [255, 255]]}}), dtype=np.int16)
    assert np.array_equal(lifted[..., :2], base[..., :2])
    assert (lifted[..., 2] - base[..., 2]).mean() > 20


def test_channel_curves_match_in_8_and_16_bit():
    image = photo_like()
    options = {"curve_channels": {"r": [[0, 0], [128, 100], [255, 255]], "g": [[0, 20], [255, 235]]}}
    eight = np.asarray(run_pipeline(image, options), dtype=np.int16)
    sixteen = np.asarray(run_pipeline(image, dict(options, bit_depth=16)), dtype=np.int16)
    assert np.abs(eight - sixteen).max() <= 1


@pytest.mark.parametrize("bad", [{"x": [[0, 0], [255, 255]]}, {"r": [[0, 0]]}, {"g": [[0, 300], [255, 255]]}, [[0, 0], [255, 255]]])
def test_invalid_channel_curves_are_rejected(bad):
    with pytest.raises(OptionsError):
        parse_options({"curve_channels": bad})


def test_cli_channel_curve_flags():
    from app.cli import build_parser, options_from_args

    args = build_parser().parse_args(["--input", "a.jpg", "--output", "b.jpg", "--curve-blue", "0:30,255:255"])
    assert options_from_args(args)["curve_channels"] == {"b": [(0, 30), (255, 255)]}


def test_only_bent_channels_become_options_and_only_while_the_curve_is_enabled():
    recipe = dict(recipe_defaults(), curve_enabled=True, curve_g_mid=150)
    assert curve_channels_from(recipe) == {"g": [(0, 0), (64, 64), (128, 150), (192, 192), (255, 255)]}
    assert recipe_to_options(recipe)["curve_channels"] == {"g": [(0, 0), (64, 64), (128, 150), (192, 192), (255, 255)]}
    assert curve_channels_from(dict(recipe, curve_enabled=False)) is None
    assert curve_channels_from(dict(recipe_defaults(), curve_enabled=True)) is None
