from __future__ import annotations

from app.professional_color import HSL_BAND_ORDER

# Aspect ratios offered on the unified UI's Crop tab (width, height) -- the canonical definition,
# imported by app.frontend_unified_app (for the dropdown) and used here (to translate a saved
# recipe's crop_aspect label into the short "W:H" code app.backend.process_image() expects).
ASPECT_RATIOS = {
    "1:1 Square": (1, 1),
    "4:5 Portrait": (4, 5),
    "5:4 Landscape": (5, 4),
    "3:2 Landscape": (3, 2),
    "16:9 Widescreen": (16, 9),
    "9:16 Story": (9, 16),
}

# recipe key -> options key, for every field that carries over unchanged (same name, same value).
_DIRECT_PASSTHROUGH_KEYS = (
    "exposure", "contrast", "highlights", "shadows", "preset",
    "temperature", "tint", "saturation", "vibrance",
    "texture", "clarity", "denoise", "sharpen", "dehaze",
    "portrait_smoothing", "removal_radius",
    "cg_shadows_hue", "cg_shadows_intensity",
    "cg_midtones_hue", "cg_midtones_intensity",
    "cg_highlights_hue", "cg_highlights_intensity", "cg_balance",
    "lens_distortion", "lens_vignette_amount", "lens_vignette_midpoint", "lens_chromatic_aberration",
    "lens_perspective_v", "lens_perspective_h",
)

# recipe key -> differently-named options key.
_RENAMED_KEYS = {
    "mask_target_exposure": "mask_exposure",
    "mask_target_contrast": "mask_contrast",
    "mask_target_saturation": "mask_saturation",
}


def recipe_to_options(recipe: dict) -> dict:
    """Converts a saved recipe (app.frontend_unified_app._build_recipe()'s shape -- what a custom
    preset or a .photoproject's recipe field actually stores) into the flat `options` dict
    app.backend.process_image()/process_via_backend() expect. Used to apply a *saved* preset to a
    batch of photos (app.batch_release.process_files()'s `recipe` parameter) -- distinct from the
    live single-photo editor's own `options = {...}` construction in frontend_unified_app.py, which
    reads directly from each tab's widget return values rather than from a serialized recipe dict,
    so the two aren't natural candidates to unify into one function without a much larger rework of
    how that script's tabs are wired.

    Deliberately excludes `mask_layers`/`removal_mask_layers` (and a project's optional overlay
    `layer`, which `_build_recipe()` doesn't even include) -- those reference pixel coordinates
    and/or an image tied to the one photo they were made on, so applying them unchanged to a
    different photo in a batch would silently select or overlay the wrong region far more often
    than the right one. A saved preset's *global* tone/color/detail/lens settings still transfer
    correctly to a different photo (the whole point of "sync settings" across a shoot); only the
    two local/spatial fields are left out, the same trade-off Lightroom's own Sync dialog makes
    explicit by letting "Local Adjustments" be unchecked when photos differ.
    """
    options: dict = {}
    for key in _DIRECT_PASSTHROUGH_KEYS:
        if key in recipe:
            options[key] = recipe[key]
    for recipe_key, options_key in _RENAMED_KEYS.items():
        if recipe_key in recipe:
            options[options_key] = recipe[recipe_key]

    if recipe.get("curve_enabled"):
        options["curve"] = [
            (0, recipe.get("curve_black", 0)),
            (64, recipe.get("curve_shadows", 64)),
            (128, recipe.get("curve_mid", 128)),
            (192, recipe.get("curve_highlights", 192)),
            (255, recipe.get("curve_white", 255)),
        ]

    hsl_bands = {}
    for band in HSL_BAND_ORDER:
        band_key = band.lower()
        hue_key, sat_key, lum_key = f"hsl_{band_key}_hue", f"hsl_{band_key}_sat", f"hsl_{band_key}_lum"
        if hue_key in recipe or sat_key in recipe or lum_key in recipe:
            hsl_bands[band] = {
                "hue": recipe.get(hue_key, 0), "sat": recipe.get(sat_key, 0), "lum": recipe.get(lum_key, 0),
            }
    if hsl_bands:
        options["hsl_bands"] = hsl_bands

    crop_aspect = recipe.get("crop_aspect", "Original")
    if crop_aspect in ASPECT_RATIOS:
        width_ratio, height_ratio = ASPECT_RATIOS[crop_aspect]
        options["aspect"] = f"{width_ratio}:{height_ratio}"

    return options
