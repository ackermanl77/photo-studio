"""The one place that knows what every edit option and every recipe control accepts.

Two related but distinct contracts live here:

- OPTION_SPECS / parse_options(): the `options` dict app.pipeline.run_pipeline() consumes -- the
  /process API's JSON, the CLI's flags, a batch recipe. Validated strictly: an unknown key (a typo
  like "exposre", which used to return 200 and silently do nothing), a wrong type (`{"exposure":
  null}` used to be an HTML 500), or a value outside the option's valid *domain* is rejected with an
  OptionsError (a ValueError, so every existing "bad request" handler already turns it into a 400)
  that names the offending keys. Domain limits are deliberately wider than the UI's sliders: the
  CLI and API have always accepted e.g. `--exposure 2.0`, and narrowing that would break callers.

- RECIPE_CONTROLS / sanitize_recipe(): a saved recipe (Undo/Redo snapshot, custom preset, project,
  imported preset file) about to be written back into the Streamlit widgets. Lenient on purpose: a
  hand-edited, older, or newer file must never take the page down (a value above a slider's max
  raised StreamlitValueAboveMaxError), so out-of-range values are clamped to the *widget's* range,
  wrong types fall back to the control's default, and unknown keys are dropped.
  Both tables' control entries are derived from app.controls.CONTROLS (F15), the one registry the
  widgets are generated from too, so a widget's range and its sanitizer can't drift apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.controls import CONTROLS
from app.professional_color import HSL_BAND_ORDER

# Bumped whenever an option's meaning changes incompatibly; a request carrying a newer
# "schema_version" than this build understands is rejected instead of being misread.
SCHEMA_VERSION = 1
# Same idea for saved recipes (presets/projects) -- see migrate_recipe().
# 2: Exposure moved to EV ("exposure_ev"), Contrast/Saturation and the mask targets to -100..100
# ("*_amount") -- see migrate_legacy_keys().
RECIPE_VERSION = 2


class OptionsError(ValueError):
    """An invalid `options` payload -- a ValueError subclass so the backend's existing
    ValueError -> JSON 400 handling (and the CLI's own error path) covers it unchanged."""


@dataclass(frozen=True)
class OptionSpec:
    kind: str  # "number", "int", "bool", "choice", or a structured kind validated by name below
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple | None = None
    nullable: bool = False
    # Which half of the two-stage pipeline consumes it (see app.pipeline.run_pipeline): "detail"
    # (Lens/Tone/Color/Detail), "finish" (Repair onward), or "request" (transport/output control).
    stage: str = "finish"


def _num(minimum: float, maximum: float, stage: str = "finish") -> OptionSpec:
    return OptionSpec("number", minimum, maximum, stage=stage)


def _int(minimum: int, maximum: int, stage: str = "finish") -> OptionSpec:
    return OptionSpec("int", minimum, maximum, stage=stage)


def _control_option_spec(control) -> OptionSpec:
    """The /process option a registry control sets (app.controls): its kind and its validation
    domain -- the control's own range carried through its transform, unless the control declares a
    wider API domain (the CLI and API have always accepted e.g. `--contrast 2.0`)."""
    kind = control.option_kind or {"int": "number", "float": "number"}.get(control.kind, control.kind)
    if kind == "choice":
        return OptionSpec("choice", choices=control.choices, nullable=control.option_nullable, stage=control.stage)
    if kind == "bool":
        return OptionSpec("bool", nullable=control.option_nullable, stage=control.stage)
    if kind == "lut":
        return OptionSpec("lut", nullable=control.option_nullable, stage=control.stage)
    if kind == "color":
        return OptionSpec("color", nullable=control.option_nullable, stage=control.stage)
    low, high = control.option_domain or (control.to_option(control.minimum), control.to_option(control.maximum))
    return OptionSpec(kind, low, high, nullable=control.option_nullable, stage=control.stage)


# Options no single registry control maps to: CLI/API-only knobs, the composite options several
# controls feed (curve, hsl_bands, aspect, crop_box), per-selection data, and request control.
_OTHER_OPTION_SPECS: dict[str, OptionSpec] = {
    # Tone (brightness and the multiplier exposure are CLI/API-only; the UI's Exposure is exposure_ev)
    "brightness": _num(0, 10, "detail"),
    "exposure": _num(0, 10, "detail"),
    "hsl_bands": OptionSpec("hsl_bands", nullable=True, stage="detail"),
    "curve": OptionSpec("curve", nullable=True, stage="detail"),
    # Per-channel R/G/B curves (F13): {"r"|"g"|"b": [[x, y], ...]}, after the master curve.
    "curve_channels": OptionSpec("curve_channels", nullable=True, stage="detail"),
    # Repair / Masks
    "mask_exposure": _num(0, 10),
    # One adjustment set per selection that has its own (B8) -- paired, in order, with the masks in
    # /process's "local_masks" file. See LOCAL_ADJUSTMENT_SPECS.
    "local_adjustments": OptionSpec("local_adjustments", nullable=True),
    # Liquify (F23): parametric warp strokes (app.liquify), applied last in the Geometry step.
    "liquify": OptionSpec("liquify", nullable=True, stage="detail"),
    # Photoshop filter plug-ins (PS20, app.filter_plugins): replayed after Film Grain, in order.
    "plugin_filters": OptionSpec("plugin_filters", nullable=True),
    # Crop
    "aspect": OptionSpec("aspect", nullable=True),
    "crop_box": OptionSpec("crop_box", nullable=True),
    # AI Upscale (an export-time action in the UI)
    "ai_upscale": OptionSpec("bool"),
    "ai_upscale_factor": OptionSpec("choice", choices=(2, 4)),
    # Request / transport control
    "schema_version": OptionSpec("int", 0, SCHEMA_VERSION, stage="request"),
    "stop_after": OptionSpec("choice", choices=("detail",), nullable=True, stage="request"),
    "resume_from_detail": OptionSpec("bool", stage="request"),
    "preview_scale": OptionSpec("number", 0.01, 1.0, stage="request"),
    "output_format": OptionSpec("choice", choices=("PNG", "JPEG"), stage="request"),
    "quality": _int(1, 100, "request"),
    "max_width": _int(0, 100_000, "request"),
    "max_height": _int(0, 100_000, "request"),
    "strip_metadata": OptionSpec("bool", stage="request"),
    # 16 = run the pipeline in float32 high-precision mode for a genuine 16-bit PNG/TIFF export (B4).
    "bit_depth": OptionSpec("choice", choices=(8, 16), stage="request"),
    # The color space the pipeline works in (B14). "Display P3" for a Display P3 export, so colours
    # outside sRGB survive the decode/ICC step; app.backend sets it from the export settings.
    "working_space": OptionSpec("choice", choices=("sRGB", "Display P3"), stage="request"),
    "export": OptionSpec("export", nullable=True, stage="request"),
}

OPTION_SPECS: dict[str, OptionSpec] = {
    **{control.option: _control_option_spec(control) for control in CONTROLS if control.option},
    **_OTHER_OPTION_SPECS,
}

# A selection's own adjustment set (Masks tab, "Own adjustments"): UI units throughout.
LOCAL_ADJUSTMENT_SPECS = {
    "exposure_ev": _num(-5, 5), "contrast": _num(-100, 100), "saturation": _num(-100, 100),
    "temperature": _num(-100, 100), "tint": _num(-100, 100), "clarity": _num(-100, 100),
    "exposure_offset": _num(-0.5, 0.5), "exposure_gamma": _num(0.1, 3.0),
    "posterize_levels": _num(0, 64), "threshold_level": _num(0, 255),
}
LOCAL_ADJUSTMENT_DEFAULTS = {name: 1.0 if name == "exposure_gamma" else 0.0 for name in LOCAL_ADJUSTMENT_SPECS}
MAX_LOCAL_ADJUSTMENTS = 32


def clean_adjustment_set(value) -> dict:
    """An adjustment layer's set (F25): the per-selection keys, clamped; anything else dropped."""
    value = value if isinstance(value, dict) else {}
    out = {}
    for name, spec in LOCAL_ADJUSTMENT_SPECS.items():
        default = LOCAL_ADJUSTMENT_DEFAULTS[name]
        raw = value.get(name, default)
        out[name] = float(min(max(raw, spec.minimum), spec.maximum)) if _is_number(raw) else default
    return out

DETAIL_STAGE_OPTION_KEYS = tuple(key for key, spec in OPTION_SPECS.items() if spec.stage == "detail")


def parse_aspect(value: str) -> tuple[int, int] | None:
    """"Original" (or empty) -> None; "W:H" with positive integers -> (W, H). Any ratio is
    accepted, not only the Crop tab's named ones (see ASPECT_RATIOS), so the CLI's
    `--aspect W H` and the backend share one parser instead of a duplicated lookup table."""
    if value in (None, "", "Original"):
        return None
    parts = str(value).split(":")
    if len(parts) != 2:
        raise OptionsError(f"aspect must be 'Original' or 'W:H', got {value!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as error:
        raise OptionsError(f"aspect must be 'Original' or 'W:H', got {value!r}") from error
    if width <= 0 or height <= 0:
        raise OptionsError("aspect ratio values must be positive")
    return width, height


def is_hex_color(value) -> bool:
    """'#RRGGBB' (a "color" control's value)."""
    return isinstance(value, str) and len(value) == 7 and value[0] == "#" and all(c in "0123456789abcdefABCDEF" for c in value[1:])


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_value(key: str, spec: OptionSpec, value):
    if value is None:
        if spec.nullable:
            return None
        raise OptionsError(f"{key} must not be null")
    kind = spec.kind
    if kind in ("number", "int"):
        if not _is_number(value) or (kind == "int" and float(value) != int(value)):
            raise OptionsError(f"{key} must be {'an integer' if kind == 'int' else 'a number'}, got {value!r}")
        if not spec.minimum <= value <= spec.maximum:
            raise OptionsError(f"{key} must be between {spec.minimum:g} and {spec.maximum:g}, got {value!r}")
        return int(value) if kind == "int" else float(value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise OptionsError(f"{key} must be true or false, got {value!r}")
        return value
    if kind == "lut":
        # A LUT library name (app.lut) -- never a path; "" means none. Whether the file exists is
        # checked when the pipeline loads it, with a message naming the LUT.
        from app.lut import is_valid_name

        if value == "":
            return None
        if not is_valid_name(value):
            raise OptionsError(f"{key} must be the name of a LUT in the LUT library, got {value!r}")
        return value
    if kind == "color":
        if not is_hex_color(value):
            raise OptionsError(f"{key} must be a colour like '#RRGGBB', got {value!r}")
        return value.upper()
    if kind == "choice":
        if value not in spec.choices:
            if key == "preset":
                raise OptionsError(f"Unknown preset: {value} (choose from {', '.join(spec.choices)})")
            raise OptionsError(f"{key} must be one of {', '.join(map(str, spec.choices))}; got {value!r}")
        return value
    if kind == "hsl_bands":
        if not isinstance(value, dict):
            raise OptionsError("hsl_bands must be an object mapping band names to {hue, sat, lum}")
        unknown = sorted(set(value) - set(HSL_BAND_ORDER))
        if unknown:
            raise OptionsError(f"Unknown HSL band(s): {', '.join(map(str, unknown))} (expected {', '.join(HSL_BAND_ORDER)})")
        bands = {}
        for band, channels in value.items():
            if not isinstance(channels, dict) or set(channels) - {"hue", "sat", "lum"}:
                raise OptionsError(f"hsl_bands.{band} must be an object with only hue/sat/lum")
            bands[band] = {name: _check_value(f"hsl_bands.{band}.{name}", _num(-100, 100), channels.get(name, 0)) for name in ("hue", "sat", "lum")}
        return bands
    if kind == "curve":
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise OptionsError("curve must be a list of at least two [x, y] points")
        points = []
        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(_is_number(v) and 0 <= v <= 255 for v in point):
                raise OptionsError(f"curve points must be [x, y] pairs within 0-255, got {point!r}")
            points.append((int(point[0]), int(point[1])))
        return points
    if kind == "curve_channels":
        if not isinstance(value, dict) or set(value) - {"r", "g", "b"}:
            raise OptionsError('curve_channels must be an object with only "r", "g" and "b" curves')
        return {channel: _check_value(f"curve_channels.{channel}", OptionSpec("curve"), points) for channel, points in value.items()}
    if kind == "crop_box":
        if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_is_number(v) for v in value):
            raise OptionsError("crop_box must be [left, top, right, bottom]")
        left, top, right, bottom = (int(v) for v in value)
        if right <= left or bottom <= top:
            raise OptionsError("crop_box must describe a non-empty region")
        return [left, top, right, bottom]
    if kind == "local_adjustments":
        if not isinstance(value, list) or len(value) > MAX_LOCAL_ADJUSTMENTS:
            raise OptionsError(f"local_adjustments must be a list of at most {MAX_LOCAL_ADJUSTMENTS} objects")
        result = []
        for index, entry in enumerate(value):
            if not isinstance(entry, dict) or set(entry) - set(LOCAL_ADJUSTMENT_SPECS):
                raise OptionsError(f"local_adjustments[{index}] must be an object with only {', '.join(LOCAL_ADJUSTMENT_SPECS)}")
            result.append({
                name: _check_value(f"local_adjustments[{index}].{name}", spec, entry.get(name, LOCAL_ADJUSTMENT_DEFAULTS[name]))
                for name, spec in LOCAL_ADJUSTMENT_SPECS.items()
            })
        return result
    if kind == "aspect":
        parse_aspect(value)
        return value
    if kind == "liquify":
        from app.liquify import MAX_STROKES, clean_strokes

        if not isinstance(value, list) or len(value) > MAX_STROKES:
            raise OptionsError(f"liquify must be a list of at most {MAX_STROKES} strokes")
        strokes = clean_strokes(value)
        if len(strokes) != len(value):
            raise OptionsError('liquify strokes must be {"tool", "size", "strength", "points": [[x, y], ...]}')
        return strokes or None
    if kind == "plugin_filters":
        from app.filter_plugins import clean_steps

        try:
            return clean_steps(value) or None
        except ValueError as error:
            raise OptionsError(str(error)) from None
    if kind == "export":
        from app.export import validate_export_settings

        return validate_export_settings(value)
    raise AssertionError(f"unhandled option kind {kind}")


def parse_options(raw) -> dict:
    """Validate and normalize an `options` payload (see the module docstring). Returns a new dict
    with only the keys that were given (defaults stay the pipeline's business), numbers normalized
    to float/int. Raises OptionsError listing every unknown key at once."""
    if not isinstance(raw, dict):
        raise OptionsError(f"options must be a JSON object, got {type(raw).__name__}")
    unknown = sorted(key for key in raw if key not in OPTION_SPECS)
    if unknown:
        raise OptionsError(f"Unknown option(s): {', '.join(map(str, unknown))}")
    version = raw.get("schema_version")
    if _is_number(version) and version > SCHEMA_VERSION:
        raise OptionsError(f"options schema_version {version} is newer than this build supports ({SCHEMA_VERSION})")
    return {key: _check_value(key, OPTION_SPECS[key], value) for key, value in raw.items()}


# ----------------------------------------------------------------------------------------------
# Recipe controls (the widget-facing contract -- see the module docstring)


@dataclass(frozen=True)
class RecipeControl:
    kind: str  # "int", "float", "bool", "choice"
    default: object
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple | None = None


def _ri(minimum: int, maximum: int, default: int = 0) -> RecipeControl:
    return RecipeControl("int", default, minimum, maximum)


def _rf(minimum: float, maximum: float, default: float) -> RecipeControl:
    return RecipeControl("float", default, minimum, maximum)


# One entry per registry control (app.controls) -- the widgets' own kind, range and default.
RECIPE_CONTROLS: dict[str, RecipeControl] = {
    control.key: RecipeControl(control.kind, control.default, control.minimum, control.maximum, control.choices)
    for control in CONTROLS
}

# Recipe entries that aren't single widgets -- lists of per-selection/per-layer dicts, validated
# structurally by sanitize_recipe() rather than clamped.
RECIPE_LIST_KEYS = ("mask_layers", "removal_mask_layers", "curve_extra_points", "layers", "liquify_strokes", "plugin_filters")
SELECTION_KINDS = ("Luminance", "Color", "Horizontal gradient", "Vertical gradient", "Radial", "Painted", "Magic Wand")


def recipe_defaults() -> dict:
    return {key: control.default for key, control in RECIPE_CONTROLS.items()}


def _sanitize_control(control: RecipeControl, value):
    if control.kind == "bool":
        return value if isinstance(value, bool) else control.default
    if control.kind == "choice":
        return value if value in control.choices else control.default
    if control.kind == "lut":
        from app.lut import is_valid_name

        return value if value == "" or is_valid_name(value) else control.default
    if control.kind == "color":
        return value.upper() if is_hex_color(value) else control.default
    if not _is_number(value):
        return control.default
    clamped = min(max(float(value), control.minimum), control.maximum)
    return int(round(clamped)) if control.kind == "int" else float(clamped)


def _sanitize_selection(entry) -> dict | None:
    if not isinstance(entry, dict) or entry.get("kind") not in SELECTION_KINDS or not isinstance(entry.get("params", {}), dict):
        return None
    clean = dict(entry)
    clean.setdefault("params", {})
    clean["name"] = str(clean.get("name") or clean["kind"])
    clean["enabled"] = clean.get("enabled", True) if isinstance(clean.get("enabled", True), bool) else True
    clean["invert"] = clean.get("invert", False) if isinstance(clean.get("invert", False), bool) else False
    clean["soften"] = _sanitize_control(_ri(0, 40), clean.get("soften", 0))
    clean["opacity"] = _sanitize_control(_rf(0.0, 1.0, 1.0), clean.get("opacity", 1.0))
    clean["combine"] = clean.get("combine") if clean.get("combine") in ("add", "subtract", "intersect") else "add"
    clean["adjust"] = clean.get("adjust") if isinstance(clean.get("adjust"), bool) else False
    adjustments = clean.get("adjustments") if isinstance(clean.get("adjustments"), dict) else {}
    clean["adjustments"] = {
        name: _sanitize_control(_rf(-5.0, 5.0, 0.0) if name == "exposure_ev" else _ri(-100, 100), adjustments.get(name, 0))
        for name in LOCAL_ADJUSTMENT_SPECS
    }
    return clean


# Recipe version 1 keys (multipliers) -> version 2 keys (EV / -100..100), see migrate_legacy_keys().
_LEGACY_EV_KEYS = {"exposure": "exposure_ev", "mask_target_exposure": "mask_target_exposure_ev"}
_LEGACY_AMOUNT_KEYS = {
    "contrast": "contrast_amount", "saturation": "saturation_amount",
    "mask_target_contrast": "mask_target_contrast_amount", "mask_target_saturation": "mask_target_saturation_amount",
}


def migrate_legacy_keys(recipe: dict) -> dict:
    """Convert a version-1 recipe's multiplier controls to version 2's units: an exposure
    multiplier m on gamma-encoded values is about 2.2*log2(m) stops (rounded to the slider's 0.05
    step), a contrast/saturation multiplier c becomes (c - 1) * 100. Keys already in the new form
    win; anything unconvertible is dropped (sanitize_recipe then falls back to the default)."""
    clean = dict(recipe)
    for old, new in _LEGACY_EV_KEYS.items():
        value = clean.pop(old, None)
        if new not in clean and _is_number(value) and value > 0:
            clean[new] = round(round(2.2 * math.log2(value) / 0.05) * 0.05, 2)
    for old, new in _LEGACY_AMOUNT_KEYS.items():
        value = clean.pop(old, None)
        if new not in clean and _is_number(value):
            clean[new] = round((value - 1.0) * 100)
    return clean


def sanitize_recipe(recipe) -> dict:
    """Make a recipe safe to write into the widgets (see the module docstring): keeps only known
    keys, clamps each control to its widget's range and coerces its type, drops malformed
    selection/curve-point entries. A key absent from `recipe` stays absent -- callers that want
    "everything else back to default" (Reset, Load Project) merge over recipe_defaults() first."""
    if not isinstance(recipe, dict):
        return {}
    recipe = migrate_legacy_keys(recipe)
    clean: dict = {}
    for key, value in recipe.items():
        if key in RECIPE_CONTROLS:
            clean[key] = _sanitize_control(RECIPE_CONTROLS[key], value)
        elif key in ("mask_layers", "removal_mask_layers"):
            entries = value if isinstance(value, list) else []
            clean[key] = [entry for entry in map(_sanitize_selection, entries) if entry is not None]
        elif key == "curve_extra_points":
            points = value if isinstance(value, list) else []
            clean[key] = [
                [int(point[0]), int(point[1])] for point in points
                if isinstance(point, (list, tuple)) and len(point) == 2 and all(_is_number(v) and 0 <= v <= 255 for v in point)
            ]
        elif key == "layers":
            clean[key] = [layer for layer in value if isinstance(layer, dict)] if isinstance(value, list) else []
        elif key == "liquify_strokes":
            from app.liquify import clean_strokes

            clean[key] = clean_strokes(value)
        elif key == "plugin_filters":
            # Each valid step keeps its UI id (a string); an invalid list is dropped whole rather than
            # half-kept, since the filters run in order.
            from app.filter_plugins import clean_steps

            entries = value if isinstance(value, list) else []
            try:
                cleaned = clean_steps([{k: v for k, v in entry.items() if k != "id"} for entry in entries if isinstance(entry, dict)])
            except ValueError:
                cleaned = []
            ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
            clean[key] = [{**step, **({"id": ids[index]} if isinstance(ids[index], str) else {})}
                          for index, step in enumerate(cleaned)] if len(cleaned) == len(ids) else []
    return clean


def migrate_recipe(recipe: dict, version: int | None) -> dict:
    """Bring a recipe saved at `version` (None = saved before recipes were versioned, which is
    RECIPE_VERSION 1's shape) up to RECIPE_VERSION: version 1 -> 2 converts the multiplier controls
    (migrate_legacy_keys). A recipe from a *newer* build is refused, not guessed at."""
    if version is not None and (not _is_number(version) or version > RECIPE_VERSION):
        raise ValueError(f"This file was saved by a newer version of Photo Studio (recipe version {version}); update the app to open it.")
    return migrate_legacy_keys(recipe) if (version or 1) < 2 else dict(recipe)
