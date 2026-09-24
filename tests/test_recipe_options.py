from app.professional_color import HSL_BAND_ORDER
from app.recipe_options import ASPECT_RATIOS, merge_curve_points, recipe_to_options

# Mirrors app.frontend_unified_app.RECIPE_DEFAULTS's shape (not imported directly: that module is a
# Streamlit script with top-level st.* calls meant to run under AppTest/`streamlit run`, not to be
# imported as a plain library -- every other test file that exercises it does so via
# streamlit.testing.v1.AppTest instead). Kept in sync by hand; test_frontend_unified_app.py's own
# recipe-shaped assertions (e.g. Undo/Redo/Reset round-tripping every field) are what would catch a
# real drift between the two.
_BASE_RECIPE = {
    "exposure": 1.0, "contrast": 1.0, "highlights": 0, "shadows": 0, "preset": "None",
    "temperature": 0, "tint": 0, "saturation": 1.0, "vibrance": 0,
    "curve_enabled": False, "curve_black": 0, "curve_shadows": 64, "curve_mid": 128, "curve_highlights": 192, "curve_white": 255,
    "curve_extra_points": [],
    "texture": 0, "clarity": 0, "denoise": 0, "denoise_color": 0, "sharpen": False, "dehaze": 0,
    "ai_upscale": False, "ai_upscale_factor": 2,
    "effects_vignette_amount": 0, "effects_vignette_midpoint": 50, "effects_grain_amount": 0,
    "portrait_smoothing": 0.0,
    "mask_target_exposure": 1.0, "mask_target_contrast": 1.0, "mask_target_saturation": 1.0,
    "crop_aspect": "Original", "removal_radius": 3,
    "crop_free_x0": 0.0, "crop_free_y0": 0.0, "crop_free_x1": 1.0, "crop_free_y1": 1.0,
    "lens_profile": "None", "lens_distortion": 0, "lens_vignette_amount": 0, "lens_vignette_midpoint": 50,
    "lens_chromatic_aberration": 0, "lens_perspective_v": 0, "lens_perspective_h": 0,
    "cg_shadows_hue": 0, "cg_shadows_intensity": 0,
    "cg_midtones_hue": 0, "cg_midtones_intensity": 0,
    "cg_highlights_hue": 0, "cg_highlights_intensity": 0,
    "cg_balance": 0,
    "mask_layers": [], "removal_mask_layers": [],
}
for _band in HSL_BAND_ORDER:
    for _channel in ("hue", "sat", "lum"):
        _BASE_RECIPE[f"hsl_{_band.lower()}_{_channel}"] = 0


def _full_recipe(**overrides) -> dict:
    """A recipe shaped like app.frontend_unified_app._build_recipe()'s real output (every field a
    saved custom preset actually carries, all HSL bands included even at zero) -- confirms
    recipe_to_options() handles that full shape, not a hand-picked subset of fields."""
    recipe = dict(_BASE_RECIPE)
    recipe.update(overrides)
    return recipe


def test_recipe_to_options_passes_through_plain_fields():
    recipe = _full_recipe(exposure=1.2, vibrance=40, dehaze=30, lens_perspective_v=25)
    options = recipe_to_options(recipe)

    assert options["exposure"] == 1.2
    assert options["vibrance"] == 40
    assert options["dehaze"] == 30
    assert options["lens_perspective_v"] == 25


def test_recipe_to_options_passes_through_ai_upscale_fields():
    recipe = _full_recipe(ai_upscale=True, ai_upscale_factor=4)
    options = recipe_to_options(recipe)

    assert options["ai_upscale"] is True
    assert options["ai_upscale_factor"] == 4


def test_recipe_to_options_renames_mask_target_fields():
    recipe = _full_recipe(mask_target_exposure=1.3, mask_target_contrast=0.9, mask_target_saturation=1.1)
    options = recipe_to_options(recipe)

    assert options["mask_exposure"] == 1.3
    assert options["mask_contrast"] == 0.9
    assert options["mask_saturation"] == 1.1
    assert "mask_target_exposure" not in options


def test_recipe_to_options_builds_the_curve_only_when_enabled():
    disabled = recipe_to_options(_full_recipe(curve_enabled=False, curve_black=10))
    assert disabled.get("curve") is None

    enabled = recipe_to_options(_full_recipe(curve_enabled=True, curve_black=10, curve_shadows=70, curve_mid=130, curve_highlights=190, curve_white=250))
    assert enabled["curve"] == [(0, 10), (64, 70), (128, 130), (192, 190), (255, 250)]


def test_recipe_to_options_merges_extra_curve_points_in():
    recipe = _full_recipe(curve_enabled=True, curve_extra_points=[[30, 90], [150, 40]])
    options = recipe_to_options(recipe)

    assert options["curve"] == [(0, 0), (30, 90), (64, 64), (128, 128), (150, 40), (192, 192), (255, 255)]


def test_merge_curve_points_sorts_by_x():
    fixed = [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]
    result = merge_curve_points(fixed, [[200, 10], [20, 200]])
    assert result == [(0, 0), (20, 200), (64, 64), (128, 128), (192, 192), (200, 10), (255, 255)]


def test_merge_curve_points_drops_an_extra_point_colliding_with_a_fixed_points_x():
    """apply_curves() requires strictly increasing X -- a defensive drop, not a crash, for the one
    case the on-canvas editor doesn't hard-prevent client-side: an extra point dragged exactly
    onto an existing point's X."""
    fixed = [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]
    result = merge_curve_points(fixed, [[64, 200]])
    assert result == fixed  # the colliding extra point is silently dropped, fixed point kept


def test_merge_curve_points_with_no_extra_points_returns_the_fixed_points_unchanged():
    fixed = [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]
    assert merge_curve_points(fixed, []) == fixed


def test_recipe_to_options_builds_hsl_bands_from_flat_keys():
    recipe = _full_recipe(hsl_red_hue=20, hsl_red_sat=-30, hsl_red_lum=5)
    options = recipe_to_options(recipe)

    assert options["hsl_bands"]["Red"] == {"hue": 20, "sat": -30, "lum": 5}
    # Every other band is present too (all zero), matching what the live editor always sends.
    assert options["hsl_bands"]["Blue"] == {"hue": 0, "sat": 0, "lum": 0}


def test_recipe_to_options_translates_crop_aspect_label_to_the_short_code():
    for label, (width_ratio, height_ratio) in ASPECT_RATIOS.items():
        options = recipe_to_options(_full_recipe(crop_aspect=label))
        assert options["aspect"] == f"{width_ratio}:{height_ratio}"

    assert "aspect" not in recipe_to_options(_full_recipe(crop_aspect="Original"))
    # "Free-form" is excluded deliberately (see recipe_to_options()'s own docstring) even though
    # it's now persisted (crop_free_x0/_y0/_x1/_y1) -- falls back to no aspect option rather than
    # raising on the unrecognized label.
    assert "aspect" not in recipe_to_options(_full_recipe(crop_aspect="Free-form"))


def test_recipe_to_options_excludes_photo_specific_mask_and_layer_fields():
    recipe = _full_recipe(mask_layers=[{"kind": "Luminance"}], removal_mask_layers=[{"kind": "Color"}])
    options = recipe_to_options(recipe)

    assert "mask_layers" not in options
    assert "removal_mask_layers" not in options
    assert "layer" not in options
