"""Parity and golden-image tests (T4 in the improvement report).

- Every recipe control reaches the pipeline's options (or is explicitly photo-specific / UI-only),
  and every pipeline option is actually read by the pipeline -- the drift class behind several past
  "a control silently did nothing" bugs.
- The CLI and the backend produce identical pixels for the same full set of options.
- Small golden images pin each operation's output, compared by PSNR, so a refactor can't shift
  results silently. Regenerate them deliberately with PHOTO_REGENERATE_GOLDEN=1 after an intended
  change to an algorithm.
"""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.edit_schema import OPTION_SPECS, RECIPE_CONTROLS, recipe_defaults
from app.pipeline import run_pipeline
from app.recipe_options import RECIPE_KEYS_NOT_IN_OPTIONS, recipe_to_options
from tests.synthetic_assets import encode, photo_like

GOLDEN_DIR = Path(__file__).parent / "golden"
REGENERATE = os.getenv("PHOTO_REGENERATE_GOLDEN") == "1"


def _non_default(control):
    if control.kind == "bool":
        return not control.default
    if control.kind == "choice":
        return [choice for choice in control.choices if choice != control.default][-1]
    return control.maximum if control.default != control.maximum else control.minimum


@pytest.mark.parametrize("key", sorted(RECIPE_CONTROLS))
def test_every_recipe_control_reaches_the_pipeline_options(key):
    if key in RECIPE_KEYS_NOT_IN_OPTIONS:
        pytest.skip(f"{key} is photo-specific or UI-only by design")
    base_recipe = recipe_defaults()
    if key.startswith("curve_") and key != "curve_enabled":
        base_recipe["curve_enabled"] = True
    if key in ("crop_custom_w", "crop_custom_h", "crop_swap"):
        base_recipe["crop_aspect"] = "Custom"
    changed = dict(base_recipe, **{key: _non_default(RECIPE_CONTROLS[key])})
    assert recipe_to_options(changed) != recipe_to_options(base_recipe), f"{key} never reaches the pipeline"


def test_every_pipeline_option_is_read_by_the_pipeline():
    # app.adjustments reads its own options (PS1): run_pipeline() hands it the whole options dict.
    source = "".join(Path(f"app/{name}.py").read_text(encoding="utf-8") for name in ("pipeline", "geometry", "adjustments"))
    unread = [key for key, spec in OPTION_SPECS.items() if spec.stage != "request" and f'"{key}"' not in source]
    assert unread == []


FULL_OPTIONS = {
    "rotate": 90, "flip_horizontal": True, "straighten": 2.5, "constrain_crop": True,
    "lens_distortion": -12, "lens_vignette_amount": -20, "lens_chromatic_aberration": 10, "lens_perspective_v": 15,
    "exposure_ev": 0.4, "contrast": 1.1, "highlights": -30, "shadows": 25, "preset": "cinematic-soft",
    "temperature": 12, "tint": -4, "saturation": 1.1, "vibrance": 20,
    "hsl_bands": {"Blue": {"hue": 10, "sat": -20, "lum": 5}},
    "cg_shadows_hue": 200, "cg_shadows_intensity": 20, "cg_balance": 10,
    "curve": [[0, 5], [128, 135], [255, 250]],
    "texture": 20, "clarity": 30, "dehaze": 10, "denoise": 3, "sharpen": True,
    "portrait_smoothing": 0.3, "aspect": "4:5", "effects_vignette_amount": -15,  # (grain is random by design)
}


def test_cli_and_backend_produce_identical_pixels_for_the_full_option_set(tmp_path, monkeypatch):
    from app import cli
    from app.backend import app as flask_app

    source_bytes = encode(photo_like((240, 180), seed=7), "PNG")
    response = flask_app.test_client().post(
        "/process",
        data={"image": (io.BytesIO(source_bytes), "in.png"), "options": json.dumps(FULL_OPTIONS)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.json
    backend_pixels = np.asarray(Image.open(io.BytesIO(response.data)).convert("RGB"))

    source = tmp_path / "in.png"
    source.write_bytes(source_bytes)
    recipe = tmp_path / "options.json"
    recipe.write_text(json.dumps(FULL_OPTIONS), encoding="utf-8")
    output = tmp_path / "out.png"
    monkeypatch.setattr(sys, "argv", ["cli", "--input", str(source), "--output", str(output), "--options", str(recipe)])
    cli.main()
    assert np.array_equal(np.asarray(Image.open(output).convert("RGB")), backend_pixels)


GOLDEN_CASES = {
    "tone": {"exposure_ev": 0.6, "contrast": 1.2, "highlights": -40, "shadows": 40},
    "color": {"temperature": 25, "tint": -10, "vibrance": 30, "hsl_bands": {"Orange": {"hue": 15, "sat": 30, "lum": -10}}, "cg_highlights_hue": 40, "cg_highlights_intensity": 30},
    "curve": {"curve": [[0, 10], [64, 50], [192, 210], [255, 245]]},
    "detail": {"texture": 40, "clarity": 50, "dehaze": 20, "sharpen": True},
    "denoise": {"denoise": 5, "denoise_color": 7},
    "geometry": {"rotate": 270, "straighten": -4, "constrain_crop": True, "lens_distortion": 20, "lens_perspective_h": 20},
    "effects": {"effects_vignette_amount": -40, "effects_vignette_midpoint": 30, "aspect": "1:1"},
}


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_golden_images(name):
    result = run_pipeline(photo_like((160, 120), seed=11), GOLDEN_CASES[name]).convert("RGB")
    path = GOLDEN_DIR / f"{name}.png"
    if REGENERATE or not path.exists():
        if not REGENERATE:
            pytest.fail(f"missing golden image {path.name} -- run with PHOTO_REGENERATE_GOLDEN=1 to create it")
        path.parent.mkdir(parents=True, exist_ok=True)
        result.save(path)
    golden = np.asarray(Image.open(path).convert("RGB"))
    assert golden.shape == np.asarray(result).shape
    # Grain is random, and NLM can differ by a level between OpenCV builds: PSNR, not equality.
    assert _psnr(np.asarray(result), golden) >= 40.0
