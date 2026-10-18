"""The one control registry (F15): every editable recipe control declared once.

A Control carries everything the rest of the app used to keep in separate hand-written copies:

- the widget: label, kind, range, step, default, help text, widget type and session_state key
  (app/ui/widgets.py::render_control() builds the Streamlit widget from it);
- the recipe: its key, default and valid range (app.edit_schema.RECIPE_CONTROLS, which
  sanitize_recipe() clamps restored values against, is derived from this table);
- the pipeline: the option it sets, that option's stage (Stage 1 "detail" or Stage 2 "finish") and
  validation domain (app.edit_schema.OPTION_SPECS' entries for these options are generated here),
  and how a widget value becomes an option value (`transform`);
- grouping: the `section` used by the per-section "↺ Reset" buttons and the "modified" dots.

Adding a control is one Control(...) line: its slider, Undo/Redo/preset/project restore, options
mapping (live editor and app.recipe_options.recipe_to_options()), stage split, /process validation
and the parity tests in tests/test_parity_golden.py all follow. Widget keys are part of saved
state, so an existing control's `widget_key` never changes.

Streamlit-free: imported by the backend's schema as well as by the UI.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.geometry import ROTATIONS
from app.lens_corrections import LENS_PROFILES
from app.presets import PRESETS
from app.professional_color import HSL_BAND_ORDER

# Aspect ratios offered on the Crop controls (width, height) -- the canonical definition (also
# re-exported by app.recipe_options for older imports).
ASPECT_RATIOS = {
    "1:1 Square": (1, 1),
    "4:5 Portrait": (4, 5),
    "5:4 Landscape": (5, 4),
    "3:2 Landscape": (3, 2),
    "16:9 Widescreen": (16, 9),
    "9:16 Story": (9, 16),
}

# Per-channel tone curves (F13): "r", "g", "b", in that order.
CURVE_CHANNELS = ("r", "g", "b")
CURVE_POINTS = ("black", "shadows", "mid", "highlights", "white")
CURVE_X = (0, 64, 128, 192, 255)

# Widget value -> option value conversions (`Control.transform`).
TRANSFORMS = {
    # -100..100 amount -> the pipeline's multiplier (Contrast, Saturation, the mask targets).
    "amount": lambda value: max(0.0, 1.0 + float(value) / 100.0),
}


@dataclass(frozen=True)
class Control:
    key: str  # recipe key
    widget_key: str | None  # session_state key of its widget (None: kept in session_state by an editor)
    label: str
    kind: str  # "int", "float", "bool", "choice", "lut" (a LUT library name, app.lut; "" = none)
    default: object
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple | None = None
    help: str | None = None
    section: str | None = None
    widget: str | None = None  # "slider", "checkbox", "selectbox", "number_input"; None = from kind
    # Pipeline side. `option` is the /process option this control sets directly; `composite` names
    # the option it feeds together with other controls (curve, hsl_bands, aspect, crop_box).
    option: str | None = None
    stage: str = "finish"  # "detail" (Lens/Tone/Color/Detail) or "finish" (Repair onward)
    transform: str | None = None  # a TRANSFORMS name
    option_kind: str | None = None  # "number"/"int"/"bool"/"choice"; None = from kind
    option_domain: tuple | None = None  # the API's (wider) range; None = the widget's range, transformed
    option_nullable: bool = False
    composite: str | None = None
    # One photo's framing (rotation, flips, straighten, the free-form box): applied live, but never
    # carried to another photo by recipe_to_options() (batch "sync settings", CLI --recipe).
    photo_specific: bool = False

    @property
    def widget_type(self) -> str:
        if self.widget:
            return self.widget
        return {"bool": "checkbox", "choice": "selectbox", "lut": "selectbox"}.get(self.kind, "slider")

    def to_option(self, value):
        return TRANSFORMS[self.transform](value) if self.transform else value


def _slider(key, widget_key, label, minimum, maximum, default=0, *, section, option=None, stage="finish", step=None, help=None, **extra):
    kind = "float" if any(isinstance(v, float) for v in (minimum, maximum, default, step)) else "int"
    if option is True:
        option = key
    return Control(key, widget_key, label, kind, default, minimum, maximum, step, help=help, section=section, option=option, stage=stage, **extra)


def _check(key, widget_key, label, default=False, *, section, option=None, stage="finish", help=None, **extra):
    if option is True:
        option = key
    return Control(key, widget_key, label, "bool", default, help=help, section=section, option=option, stage=stage, **extra)


def _choice(key, widget_key, label, choices, default, *, section, option=None, stage="finish", help=None, **extra):
    if option is True:
        option = key
    return Control(key, widget_key, label, "choice", default, choices=tuple(choices), help=help, section=section, option=option, stage=stage, **extra)


_D = "detail"

CONTROLS: tuple[Control, ...] = (
    # --- Tone -------------------------------------------------------------------------------------
    _choice("preset", "tone-preset", "Preset Look", ("None", *sorted(PRESETS)), "None", section="tone", option=True, stage=_D, option_nullable=True),
    _slider("exposure_ev", "t-ev", "Exposure (EV)", -5.0, 5.0, 0.0, step=0.05, section="tone", option=True, stage=_D,
            help="In stops of light: +1 doubles the light, -1 halves it. Bright highlights roll off smoothly instead of clipping."),
    _slider("contrast_amount", "t-contrast", "Contrast", -100, 100, section="tone", option="contrast", stage=_D, transform="amount", option_domain=(0, 10),
            help="Spreads tones apart around midtone gray -- positive makes shadows darker and highlights brighter, negative flattens them."),
    _slider("highlights", "t-high", "Highlights", -100, 100, section="tone", option=True, stage=_D, option_domain=(-255, 255),
            help="Recovers or boosts the brightest parts of the image -- negative pulls blown highlights back, positive brightens them further."),
    _slider("shadows", "t-shad", "Shadows", -100, 100, section="tone", option=True, stage=_D, option_domain=(-255, 255),
            help="Lifts or deepens the darkest parts of the image -- positive opens up shadow detail, negative deepens shadows."),
    _slider("whites", "t-whites", "Whites", -100, 100, section="tone", option=True, stage=_D,
            help="Sets the white point -- positive pushes the brightest tones toward pure white, negative pulls them down. Midtones barely move."),
    _slider("blacks", "t-blacks", "Blacks", -100, 100, section="tone", option=True, stage=_D,
            help="Sets the black point -- negative deepens the darkest tones toward pure black, positive lifts them. Midtones barely move."),
    # --- Color ------------------------------------------------------------------------------------
    _slider("temperature", "c-temp", "Temperature (Cool/Warm)", -100, 100, section="color", option=True, stage=_D,
            help="Shifts the overall white balance -- negative cools the image toward blue, positive warms it toward orange."),
    _slider("tint", "c-tint", "Tint (Green/Magenta)", -100, 100, section="color", option=True, stage=_D,
            help="Corrects a color cast along the green/magenta axis, independent of Temperature."),
    _slider("saturation_amount", "c-saturation", "Saturation", -100, 100, section="color", option="saturation", stage=_D, transform="amount", option_domain=(0, 10),
            help="Uniformly boosts or reduces color intensity across the whole image -- -100 is grayscale, 0 is unchanged."),
    _slider("vibrance", "c-vib", "Vibrance", -100, 100, section="color", option=True, stage=_D,
            help="A smarter saturation boost -- protects skin tones and already-saturated colors, pushing washed-out ones harder instead."),
    # --- Black & White (a per-colour brightness mix of the gray conversion) -------------------------
    _check("black_white", "bw-enabled", "Convert to Black & White", section="bw", option=True, stage=_D,
           help="Converts the photo to black and white. The sliders below brighten or darken the gray of each original colour."),
    *(_slider(f"bw_{band.lower()}", f"bw-{band.lower()}", band, -100, 100, section="bw", option=True, stage=_D)
      for band in HSL_BAND_ORDER),
    # --- Tone curve (feeds the "curve" option together with the extra points) ---------------------
    _check("curve_enabled", "c-curve-en", "Enable RGB Curve", section="curve", composite="curve"),
    _slider("curve_black", "c-p0", "Blacks", 0, 64, 0, section="curve", composite="curve"),
    _slider("curve_shadows", "c-p1", "Shadows", 0, 128, 64, section="curve", composite="curve"),
    _slider("curve_mid", "c-p2", "Midtones", 64, 192, 128, section="curve", composite="curve"),
    _slider("curve_highlights", "c-p3", "Highlights", 128, 255, 192, section="curve", composite="curve"),
    _slider("curve_white", "c-p4", "Whites", 192, 255, 255, section="curve", composite="curve"),
    # Per-channel R/G/B curves (F13): the same five points per channel, feeding "curve_channels".
    *(_slider(f"curve_{channel}_{point}", f"c-{channel}-p{index}", label, low, high, default, section="curve", composite="curve_channels")
      for channel in CURVE_CHANNELS
      for index, (point, label, low, high, default) in enumerate((
          ("black", "Blacks", 0, 64, 0), ("shadows", "Shadows", 0, 128, 64), ("mid", "Midtones", 64, 192, 128),
          ("highlights", "Highlights", 128, 255, 192), ("white", "Whites", 192, 255, 255)))),
    # --- Presence and Detail ----------------------------------------------------------------------
    _slider("texture", "d-text", "Texture", -100, 100, section="presence", option=True, stage=_D,
            help="Enhances or softens fine surface detail (skin pores, fabric, foliage) without affecting large tonal shapes."),
    _slider("clarity", "d-clar", "Clarity", -100, 100, section="presence", option=True, stage=_D,
            help="Boosts or reduces midtone contrast at a larger scale than Texture -- adds a sense of punch/depth; negative gives a soft, dreamy look."),
    _slider("dehaze", "d-haze", "Dehaze", 0, 100, section="presence", option=True, stage=_D,
            help="Reduces atmospheric haze (Dark Channel Prior, a classical algorithm -- not AI). Pushed too high, it can crush shadows or halo around strong edges."),
    _slider("denoise", "d-den", "Denoise", 0, 9, step=2, section="detail", option=True, stage=_D, option_kind="int", option_domain=(0, 20),
            help="Smooths luminance noise/grain -- higher removes more noise but can soften fine detail."),
    _slider("denoise_color", "d-den-color", "Color Noise", 0, 9, step=2, section="detail", option=True, stage=_D, option_kind="int", option_domain=(0, 20),
            help="Independent chroma (color speckle) denoising strength, on top of Denoise's luminance/detail smoothing -- push higher than Denoise to clean up color blotches without softening detail as much. Has no effect while Denoise is 0."),
    _check("sharpen", "d-sharp", "Sharpen (Unsharp Mask)", section="detail", option=True, stage=_D,
           help="Increases edge contrast to make the image look crisper -- applied last in the Detail group, after Texture/Clarity/Denoise."),
    # --- Effects (after the crop) -----------------------------------------------------------------
    _slider("effects_vignette_amount", "fx-vignette-amt", "Amount", -100, 100, section="effects", option=True,
            help="Negative darkens the corners; positive brightens them."),
    _slider("effects_vignette_midpoint", "fx-vignette-mid", "Midpoint", 0, 100, 50, section="effects", option=True),
    _slider("effects_grain_amount", "fx-grain-amt", "Amount", 0, 100, section="effects", option=True),
    # --- Portrait ---------------------------------------------------------------------------------
    _slider("portrait_smoothing", "port-sm", "Skin Smoothing", 0.0, 1.0, 0.0, step=0.01, section="portrait", option=True,
            help="0 leaves skin untouched. Increase to smooth detected skin tones."),
    _check("portrait_faces_only", "port-faces", "Only around detected faces", True, section="portrait", option=True,
           help="Smooth skin tones only around the faces found in the photo, so walls, wood or sand in "
                "skin-like colours stay sharp. With no face found, every skin-toned area is smoothed."),
    # --- Masks: the shared selection's target adjustments -------------------------------------------
    _slider("mask_target_exposure_ev", "m-tev", "Target Exposure (EV)", -5.0, 5.0, 0.0, step=0.05, section="mask_targets", option="mask_exposure_ev",
            help="Exposure change in stops, applied only inside the combined mask's selected area."),
    _slider("mask_target_contrast_amount", "m-tcontrast", "Target Contrast", -100, 100, section="mask_targets", option="mask_contrast", transform="amount", option_domain=(0, 10),
            help="Contrast adjustment applied only inside the combined mask's selected area."),
    _slider("mask_target_saturation_amount", "m-tsaturation", "Target Saturation", -100, 100, section="mask_targets", option="mask_saturation", transform="amount", option_domain=(0, 10),
            help="Saturation adjustment applied only inside the combined mask's selected area."),
    # --- Crop (feeds "aspect" / "crop_box") ---------------------------------------------------------
    _choice("crop_aspect", "crop-aspect", "Aspect Ratio", ("Original", "Free-form", "Custom", *ASPECT_RATIOS), "Original", section="crop", composite="aspect"),
    _slider("crop_custom_w", "crop-custom-w", "Ratio width", 1, 100, 3, section="crop", composite="aspect", widget="number_input"),
    _slider("crop_custom_h", "crop-custom-h", "Ratio height", 1, 100, 2, section="crop", composite="aspect", widget="number_input"),
    _check("crop_swap", "crop-swap", "Swap orientation (portrait ↔ landscape)", section="crop", composite="aspect"),
    # The free-form box is drawn in the crop editor (or typed in pixels); its fractions live in
    # these session_state keys rather than in widgets of their own.
    *(_slider(f"crop_free_{axis}", f"crop-free-{axis}", f"Free-form crop {axis}", 0.0, 1.0, 1.0 if axis.endswith("1") else 0.0,
              section="crop", composite="crop_box", photo_specific=True, widget="state")
      for axis in ("x0", "y0", "x1", "y1")),
    # --- Geometry ----------------------------------------------------------------------------------
    _choice("rotate", "geo-rotate", "Rotation", ROTATIONS, 0, section="geometry", option=True, stage=_D, photo_specific=True),
    _check("flip_horizontal", "geo-flip-h", "⇆ Flip horizontal", section="geometry", option=True, stage=_D, photo_specific=True),
    _check("flip_vertical", "geo-flip-v", "⇅ Flip vertical", section="geometry", option=True, stage=_D, photo_specific=True),
    _slider("straighten", "geo-straighten", "Straighten (°)", -45.0, 45.0, 0.0, step=0.05, section="geometry", option=True, stage=_D, photo_specific=True,
            help="Rotates the photo by a small angle to level a tilted horizon -- positive turns it clockwise."),
    _check("constrain_crop", "geo-constrain", "Constrain crop", True, section="geometry", option=True, stage=_D,
           help="Crop to the largest rectangle that Lens corrections, keystone and straighten leave fully covered -- no black or smeared corners."),
    # --- Repair --------------------------------------------------------------------------------------
    _slider("removal_radius", "rm-radius", "Repair Softness", 1, 15, 3, section="repair", option=True, option_domain=(1, 50),
            help="How far the repair blends into the surrounding area -- higher softens a wider edge."),
    _choice("removal_method", "rm-method", "Repair Method", ("Fill", "Heal"), "Fill", section="repair", option=True,
            help="Fill paints the area over from its edges (best for wires, small objects on smooth areas). Heal copies the best-matching nearby patch and blends it in, keeping real texture (best for spots and blemishes on skin, sand or grass)."),
    # --- Lens ------------------------------------------------------------------------------------------
    _choice("lens_profile", "lens-profile", "Lens Profile", tuple(LENS_PROFILES), "None", section="lens"),
    _slider("lens_distortion", "lens-distortion", "Distortion", -100, 100, section="lens", option=True, stage=_D,
            help="Negative corrects barrel distortion (straight lines bowing outward); positive corrects pincushion (bowing inward)."),
    _slider("lens_vignette_amount", "lens-vignette-amt", "Vignette Amount", -100, 100, section="lens", option=True, stage=_D,
            help="Negative darkens the corners (adds a vignette); positive brightens them (removes one)."),
    _slider("lens_vignette_midpoint", "lens-vignette-mid", "Vignette Midpoint", 0, 100, 50, section="lens", option=True, stage=_D,
            help="How far from center the vignette effect starts -- higher concentrates it closer to the corners."),
    _slider("lens_chromatic_aberration", "lens-ca", "Chromatic Aberration", -100, 100, section="lens", option=True, stage=_D,
            help="Corrects red/cyan and blue/yellow fringing near high-contrast edges."),
    _slider("lens_perspective_v", "lens-persp-v", "Vertical Keystone", -100, 100, section="lens", option=True, stage=_D,
            help="Corrects converging vertical lines (e.g. a building shot from below) -- positive narrows the top edge, negative narrows the bottom."),
    _slider("lens_perspective_h", "lens-persp-h", "Horizontal Keystone", -100, 100, section="lens", option=True, stage=_D,
            help="Corrects converging horizontal lines (e.g. a wall shot at an angle) -- positive narrows the left edge, negative narrows the right."),
    # --- Color grading -----------------------------------------------------------------------------------
    *(control for tone_range, short in (("shadows", "sh"), ("midtones", "mid"), ("highlights", "hi")) for control in (
        _slider(f"cg_{tone_range}_hue", f"cg-{short}-hue", "Hue", 0, 360, section="grading", option=True, stage=_D),
        _slider(f"cg_{tone_range}_intensity", f"cg-{short}-int", "Intensity", 0, 100, section="grading", option=True, stage=_D),
    )),
    _slider("cg_balance", "cg-balance", "Balance", -100, 100, section="grading", option=True, stage=_D,
            help="Shifts where shadows end and highlights begin -- negative gives shadows more of the tonal range, positive gives highlights more."),
    # --- LUT (F17): a .cube look from the user's LUT library, last in the colour chain ------------------
    Control("lut", "lut-name", "LUT", "lut", "", section="lut", option="lut", stage=_D, option_nullable=True,
            help="A 3D LUT (.cube) -- a film emulation or creative look -- applied after the other colour and curve adjustments."),
    _slider("lut_amount", "lut-amount", "LUT strength", 0, 100, 100, section="lut", option=True, stage=_D,
            help="How much of the LUT's look to apply: 100 is the LUT as designed, lower blends it with the photo."),
    # --- HSL color mixer (feeds "hsl_bands") --------------------------------------------------------------
    *(_slider(f"hsl_{band.lower()}_{channel}", f"hsl-{band.lower()}-{channel[0]}", label, -100, 100, section="hsl", composite="hsl_bands")
      for band in HSL_BAND_ORDER for channel, label in (("hue", "Hue"), ("sat", "Saturation"), ("lum", "Luminance"))),
)

CONTROLS_BY_KEY: dict[str, Control] = {control.key: control for control in CONTROLS}
if len(CONTROLS_BY_KEY) != len(CONTROLS):
    raise RuntimeError("duplicate control key in app.controls.CONTROLS")

# recipe key -> widget session_state key (Undo/Redo, presets, projects capture and restore these).
RECIPE_WIDGET_KEYS: dict[str, str] = {control.key: control.widget_key for control in CONTROLS}

# Recipe keys a saved recipe never turns into options (see recipe_to_options): photo-specific
# framing, and UI-only state with neither an option nor a composite (the lens profile's name --
# its values are already in the lens_* controls).
RECIPE_KEYS_NOT_IN_OPTIONS = frozenset(
    control.key for control in CONTROLS if control.photo_specific or (control.option is None and control.composite is None)
)


def section_keys(section: str) -> tuple[str, ...]:
    """Recipe keys of one section, in declaration order (reset buttons, modified dots)."""
    return tuple(control.key for control in CONTROLS if control.section == section)


def controls_to_options(values: dict, *, include_photo_specific: bool = True) -> dict:
    """The options every directly-mapped control in `values` (recipe key -> value) sets."""
    options = {}
    for key, value in values.items():
        control = CONTROLS_BY_KEY.get(key)
        if control is None or control.option is None or (control.photo_specific and not include_photo_specific):
            continue
        options[control.option] = control.to_option(value)
    return options


def hsl_bands_from(values: dict) -> dict:
    """The "hsl_bands" option from the 24 HSL controls present in `values`."""
    bands = {}
    for band in HSL_BAND_ORDER:
        keys = [f"hsl_{band.lower()}_{channel}" for channel in ("hue", "sat", "lum")]
        if any(key in values for key in keys):
            bands[band] = {channel: values.get(key, 0) for channel, key in zip(("hue", "sat", "lum"), keys)}
    return bands


def curve_fixed_points(values: dict) -> list[tuple[int, int]]:
    """The tone curve's five fixed points (x positions 0/64/128/192/255) from the curve controls."""
    return [
        (x, values.get(key, CONTROLS_BY_KEY[key].default))
        for x, key in zip((0, 64, 128, 192, 255), ("curve_black", "curve_shadows", "curve_mid", "curve_highlights", "curve_white"))
    ]


def curve_channels_from(values: dict) -> dict | None:
    """The "curve_channels" option: {channel: five [x, y] points} for each R/G/B curve that isn't a
    straight line, or None when none is (or the curve is disabled)."""
    if not values.get("curve_enabled"):
        return None
    channels = {}
    for channel in CURVE_CHANNELS:
        points = [(x, values.get(f"curve_{channel}_{point}", CONTROLS_BY_KEY[f"curve_{channel}_{point}"].default)) for x, point in zip(CURVE_X, CURVE_POINTS)]
        if any(x != y for x, y in points):
            channels[channel] = points
    return channels or None


def aspect_code_from(values: dict) -> str | None:
    """"W:H" for a named or custom aspect ratio (swap applied), else None (Original / Free-form)."""
    choice = values.get("crop_aspect", "Original")
    if choice in ASPECT_RATIOS:
        width, height = ASPECT_RATIOS[choice]
    elif choice == "Custom":
        width, height = int(values.get("crop_custom_w", 3)), int(values.get("crop_custom_h", 2))
    else:
        return None
    if values.get("crop_swap"):
        width, height = height, width
    return f"{width}:{height}"
