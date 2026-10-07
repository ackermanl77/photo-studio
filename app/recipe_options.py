from __future__ import annotations

from app.controls import ASPECT_RATIOS, RECIPE_KEYS_NOT_IN_OPTIONS, aspect_code_from, controls_to_options, curve_channels_from, curve_fixed_points, hsl_bands_from

__all__ = ["ASPECT_RATIOS", "PER_PHOTO_ONLY_KEYS", "RECIPE_KEYS_NOT_IN_OPTIONS", "merge_curve_points", "recipe_to_options", "without_per_photo_only_keys"]

# Recipe keys that belong to one photo's export, never to a reusable look: AI Upscale is an
# expensive (Real-ESRGAN, seconds to minutes per photo) change of output resolution, not a style --
# carried through a preset it silently ran on every preset thumbnail render and every photo of a
# batch. Stripped from a recipe before it's saved as, or applied from, a custom preset
# (without_per_photo_only_keys()).
PER_PHOTO_ONLY_KEYS = ("ai_upscale", "ai_upscale_factor")

# Version-1 recipe keys (multipliers, from before recipes were versioned) that still pass straight
# through as options, keeping their exact old meaning instead of a rounded conversion. Version-2
# keys (the app.controls registry) are mapped by controls_to_options().
_LEGACY_OPTION_KEYS = {
    "exposure": "exposure", "contrast": "contrast", "saturation": "saturation",
    "mask_target_exposure": "mask_exposure", "mask_target_contrast": "mask_contrast", "mask_target_saturation": "mask_saturation",
}


def without_per_photo_only_keys(recipe: dict) -> dict:
    return {key: value for key, value in recipe.items() if key not in PER_PHOTO_ONLY_KEYS}


def merge_curve_points(fixed_points: list[tuple[int, int]], extra_points: list) -> list[tuple[int, int]]:
    """Combines the tone curve's 5 fixed control points (Blacks/Shadows/Midtones/Highlights/
    Whites) with any user-added extra points (the on-canvas curve editor's click-to-add points --
    see app.frontend_unified_app._curve_editor_html), sorted by X. `PhotoProcessor.apply_curves()`
    requires strictly increasing X values; the on-canvas editor doesn't hard-prevent a drag landing
    an extra point exactly on another point's X, so this is the server-side safety net -- an extra
    point whose X exactly collides with an already-placed one (in sorted order) is silently dropped
    rather than raising, keeping whichever point came first."""
    combined = sorted([tuple(point) for point in fixed_points] + [(int(x), int(y)) for x, y in extra_points], key=lambda point: point[0])
    result: list[tuple[int, int]] = []
    for x, y in combined:
        if result and result[-1][0] == x:
            continue
        result.append((x, y))
    return result


def recipe_to_options(recipe: dict) -> dict:
    """Converts a saved recipe (app.frontend_unified_app._build_recipe()'s shape -- what a custom
    preset or a .photoproject's recipe field actually stores) into the flat `options` dict
    app.backend.process_image()/process_via_backend() expect. Used to apply a *saved* preset to a
    batch of photos (app.batch_release.process_files()'s `recipe` parameter) -- distinct from the
    live editor's options, which are built from the same registry (app.controls.controls_to_options)
    out of the widgets' values plus the photo-specific pieces (framing, the pixel crop box, masks).

    Deliberately excludes `mask_layers`/`removal_mask_layers` (and a project's optional overlay
    `layer`, which `_build_recipe()` doesn't even include) -- those reference pixel coordinates
    and/or an image tied to the one photo they were made on, so applying them unchanged to a
    different photo in a batch would silently select or overlay the wrong region far more often
    than the right one. A saved preset's *global* tone/color/detail/lens settings still transfer
    correctly to a different photo (the whole point of "sync settings" across a shoot); only the
    local/spatial fields are left out, the same trade-off Lightroom's own Sync dialog makes
    explicit by letting "Local Adjustments" be unchecked when photos differ. A Free-form crop
    (`crop_aspect == "Free-form"`, its box in `crop_free_x0`/`_y0`/`_x1`/`_y1`) is excluded for the
    same reason -- it's a specific photo's exact framing, not a pattern that generalizes -- even
    though, unlike masks/layers, it's stored as size-independent fractions and so genuinely could
    be translated into a per-photo pixel `crop_box` if a future caller had each batch photo's own
    dimensions on hand at conversion time; today's single call site (`process_files()`, called once
    per batch rather than once per file) doesn't, so it's left as an explicit gap rather than
    solved partially. Rotate/flip/straighten are left out for the same reason (they fix one
    photo's framing); "constrain crop" is a preference and carries over.
    """
    options: dict = {}
    for legacy_key, options_key in _LEGACY_OPTION_KEYS.items():
        if legacy_key in recipe:
            options[options_key] = recipe[legacy_key]
    # Every registry control with a direct option (app.controls), photo-specific framing excluded.
    options.update(controls_to_options(recipe, include_photo_specific=False))
    if recipe.get("curve_enabled"):
        options["curve"] = merge_curve_points(curve_fixed_points(recipe), recipe.get("curve_extra_points", []))
    curve_channels = curve_channels_from(recipe)
    if curve_channels:
        options["curve_channels"] = curve_channels
    hsl_bands = hsl_bands_from(recipe)
    if hsl_bands:
        options["hsl_bands"] = hsl_bands
    aspect = aspect_code_from(recipe)
    if aspect:
        options["aspect"] = aspect
    return options
