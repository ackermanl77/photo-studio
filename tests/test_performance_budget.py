"""Opt-in performance budgets at 24 MP (T5 in the improvement report). Skipped unless
PHOTO_PERF_TESTS=1 -- they take tens of seconds and depend on the machine, so they're meant for a
scheduled run on the target hardware (a Windows desktop), not every CI push:

    PHOTO_PERF_TESTS=1 python -m pytest -q -m perf

Budgets are generous upper bounds; a failure means something got several times slower.
"""
import os
import time

import pytest
from PIL import Image

from app.pipeline import run_pipeline
from tests.synthetic_assets import photo_like

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(os.getenv("PHOTO_PERF_TESTS") != "1", reason="set PHOTO_PERF_TESTS=1 to run the performance budgets"),
]

TYPICAL_EDIT = {
    "exposure_ev": 0.3, "contrast": 1.1, "highlights": -20, "shadows": 20, "temperature": 10, "vibrance": 20,
    "hsl_bands": {"Blue": {"hue": 0, "sat": -10, "lum": 0}, "Orange": {"hue": 5, "sat": 10, "lum": 0}},
    "texture": 15, "clarity": 20, "denoise": 3, "sharpen": True,
}


@pytest.fixture(scope="module")
def photo_24mp():
    return photo_like((1500, 1000)).resize((6000, 4000), Image.Resampling.BILINEAR)


def _timed(function):
    start = time.perf_counter()
    function()
    return time.perf_counter() - start


def test_full_resolution_typical_edit_stays_under_the_export_budget(photo_24mp):
    assert _timed(lambda: run_pipeline(photo_24mp, TYPICAL_EDIT)) < 30.0


def test_the_interactive_proxy_stays_interactive(photo_24mp):
    proxy = photo_24mp.copy()
    proxy.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    options = {**TYPICAL_EDIT, "preview_scale": proxy.width / photo_24mp.width}
    assert _timed(lambda: run_pipeline(proxy, options)) < 3.0
