from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from app.batch import process_folder
from app.edit_schema import OptionsError, migrate_recipe, parse_options, sanitize_recipe
from app.export import METADATA_POLICIES, write_export
from app.pipeline import decode_source, run_pipeline
from app.presets import PRESETS
from app.professional_color import HSL_BAND_ORDER
from app.recipe_options import recipe_to_options, without_per_photo_only_keys

# --auto-enhance sets exactly what the GUI's "Auto Enhance" button writes into the recipe (see
# app/frontend_unified_app.py), so it now runs at those controls' normal pipeline positions
# instead of as an extra pass after everything else. Any of these given explicitly as a flag wins.
AUTO_ENHANCE_OPTIONS = {"exposure": 1.12, "contrast": 1.15, "saturation": 1.08, "denoise": 3, "sharpen": True}

# flag attribute -> options key, for every flag that maps 1:1 onto an app.pipeline option.
_FLAG_OPTIONS = {
    "brightness": "brightness", "exposure": "exposure", "contrast": "contrast", "saturation": "saturation",
    "highlights": "highlights", "shadows": "shadows", "whites": "whites", "blacks": "blacks", "texture": "texture", "clarity": "clarity",
    "temperature": "temperature", "tint": "tint", "vibrance": "vibrance", "dehaze": "dehaze",
    "denoise": "denoise", "denoise_color": "denoise_color", "preset": "preset",
    "mask_exposure": "mask_exposure", "mask_contrast": "mask_contrast", "mask_saturation": "mask_saturation",
    "exposure_ev": "exposure_ev", "rotate": "rotate", "straighten": "straighten",
}


def parse_curve(value: str) -> list[tuple[int, int]]:
    try:
        return [tuple(int(part) for part in pair.split(":")) for pair in value.split(",")]
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Curves must use x:y pairs, for example 0:0,128:140,255:255") from error


def parse_black_white(value: str) -> dict:
    """--black-white's "Blue=-60,Red=30" -> the black_white/bw_<band> options."""
    options = {"black_white": True}
    for item in filter(None, (part.strip() for part in value.split(","))):
        band, separator, amount = item.partition("=")
        key = f"bw_{band.strip().lower()}"
        if not separator or key not in {f"bw_{name.lower()}" for name in HSL_BAND_ORDER}:
            raise ValueError(f"--black-white expects BAND=AMOUNT pairs with bands {', '.join(HSL_BAND_ORDER)}; got {item!r}")
        options[key] = float(amount)
    return options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Professional photo editing pipeline -- the same step order as the GUI (app/pipeline.py)",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input image path (JPG/PNG/TIFF/WEBP or camera RAW)")
    input_group.add_argument("--input-dir", help="Folder of input images; every other flag applies to each one")
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", help="Output image path (.jpg/.png/.webp/.tif)")
    output_group.add_argument("--output-dir", help="Folder for processed images")
    # Numeric defaults are None ("not given") so a --recipe's value or --auto-enhance's isn't
    # overwritten by a flag the user never typed; the pipeline's own defaults apply otherwise.
    parser.add_argument("--recipe", help="Saved look to apply: a custom preset recipe JSON, or a presets file exported from the app")
    parser.add_argument("--recipe-name", help="Which preset to use when --recipe is an exported file holding several")
    parser.add_argument("--options", help="JSON file holding a raw pipeline options object (the /process API's \"options\"); applied after --recipe, before the other flags")
    parser.add_argument("--brightness", type=float, help="Brightness multiplier (default: 1.0)")
    parser.add_argument("--contrast", type=float, help="Contrast multiplier (default: 1.0)")
    parser.add_argument("--saturation", type=float, help="Saturation multiplier (default: 1.0)")
    parser.add_argument("--exposure", type=float, help="Exposure multiplier on gamma-encoded values (default: 1.0)")
    parser.add_argument("--exposure-ev", type=float, help="Exposure in stops, in linear light, -5 to 5 (the GUI's Exposure slider)")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), help="Rotate clockwise by 90/180/270 degrees")
    parser.add_argument("--straighten", type=float, help="Straighten angle in degrees, -45 to 45 (positive = clockwise)")
    parser.add_argument("--constrain-crop", action="store_true", default=None, help="Crop away the borders lens/perspective/straighten corrections leave uncovered")
    parser.add_argument("--highlights", type=float, help="Highlights adjustment from -255 to 255")
    parser.add_argument("--shadows", type=float, help="Shadows adjustment from -255 to 255")
    parser.add_argument("--whites", type=float, help="White point from -100 to 100 (the GUI's Whites slider)")
    parser.add_argument("--blacks", type=float, help="Black point from -100 to 100 (the GUI's Blacks slider)")
    parser.add_argument("--texture", type=float, help="Texture adjustment from -100 to 100")
    parser.add_argument("--clarity", type=float, help="Clarity adjustment from -100 to 100")
    parser.add_argument("--temperature", type=float, help="Temperature from -100 cool to 100 warm")
    parser.add_argument("--tint", type=float, help="Tint from -100 green to 100 magenta")
    parser.add_argument("--vibrance", type=float, help="Vibrance from -100 to 100")
    parser.add_argument("--dehaze", type=float, help="Dehaze strength from 0 to 100")
    for channel in ("red", "green", "blue"):
        parser.add_argument(f"--curve-{channel}", type=parse_curve, help=f"{channel.title()} channel curve points, e.g. 0:0,128:140,255:255")
    parser.add_argument("--black-white", nargs="?", const="", metavar="BAND=AMOUNT,...",
                        help="Convert to black & white; optionally brighten/darken the gray of colour bands, e.g. Blue=-60,Red=30 (bands: Red, Orange, Yellow, Green, Aqua, Blue, Purple, Magenta; -100..100)")
    parser.add_argument("--curve", type=parse_curve, help="RGB curve points such as 0:0,128:140,255:255")
    parser.add_argument("--remove-mask", help="Grayscale mask path for CPU object removal (single image only)")
    parser.add_argument("--mask", help="Grayscale mask path for local adjustments (single image only)")
    parser.add_argument("--mask-exposure", type=float, help="Masked exposure multiplier")
    parser.add_argument("--mask-contrast", type=float, help="Masked contrast multiplier")
    parser.add_argument("--mask-saturation", type=float, help="Masked saturation multiplier")
    parser.add_argument("--denoise", type=int, help="Non-Local Means denoise strength, 0-9 like the GUI slider (default: 0)")
    parser.add_argument("--denoise-color", type=int, help="Separate color-noise strength for --denoise (default: same as --denoise)")
    parser.add_argument("--sharpen", action="store_true", default=None, help="Apply sharpening")
    parser.add_argument("--auto-enhance", action="store_true", help="Apply the GUI's Auto Enhance values (exposure/contrast/saturation/denoise/sharpen)")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Apply a named editing preset")
    parser.add_argument("--quality", type=int, default=92, help="JPEG/WEBP delivery quality from 1 to 100")
    parser.add_argument("--metadata", choices=METADATA_POLICIES, default="all", help="EXIF to keep: all, camera (copyright & camera only), or none")
    parser.add_argument("--remove-gps", action="store_true", help="Drop GPS location from the kept EXIF")
    parser.add_argument(
        "--gpu-export",
        action="store_true",
        help="Use CUDA for final export resizing and sharpening when available; encoding stays on the CPU and unavailable/failed CUDA falls back to CPU",
    )
    parser.add_argument(
        "--bit-depth", type=int, choices=(8, 16), default=8,
        help="16 = process in float32 and write a genuine 16-bit PNG/TIFF (16-bit and RAW sources keep their extra precision); JPEG/WEBP outputs stay 8-bit",
    )
    parser.add_argument(
        "--color-space", choices=("sRGB", "Display P3"), default="sRGB",
        help="Output color space: sRGB (default) or Display P3 (pixels converted, a P3 profile embedded)",
    )
    parser.add_argument("--crop", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"), help="Crop rectangle in pixels")
    parser.add_argument("--aspect", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), help="Center-crop to an aspect ratio")
    parser.add_argument(
        "--ai-upscale", type=int, choices=(2, 4), default=0,
        help="Real-ESRGAN neural upscale factor (2 or 4) -- optional, requires requirements-gpu.txt "
        "(torch, realesrgan); omit for no upscaling. The one deliberate exception to this app's "
        "otherwise CPU-only, classical-algorithms design -- see docs/PROJECT_STATUS.md.",
    )
    return parser


def load_recipe(path: str | Path, name: str | None = None) -> dict:
    """A recipe file is either a single recipe object (what the Presets tab stores per preset) or
    an exported presets file ({"version": 1, "presets": {name: recipe}}) -- then `name` picks one,
    optional when it holds exactly one. Migrated and sanitized like every other restore path."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A recipe file must contain a JSON object")
    version = payload.get("recipe_version")
    if isinstance(payload.get("presets"), dict):
        presets = payload["presets"]
        if name is None:
            if len(presets) != 1:
                raise ValueError(f"{path} holds {len(presets)} presets -- pick one with --recipe-name ({', '.join(sorted(presets))})")
            name = next(iter(presets))
        if name not in presets:
            raise ValueError(f"No preset named {name!r} in {path}")
        payload = presets[name]
    return sanitize_recipe(without_per_photo_only_keys(migrate_recipe(payload, version)))


def options_from_args(args: argparse.Namespace) -> dict:
    """The app.pipeline options dict for these flags: --recipe first, then --auto-enhance's
    values, then every explicitly given flag on top. Validated (app.edit_schema.parse_options)."""
    options: dict = {}
    if args.recipe:
        options.update(recipe_to_options(load_recipe(args.recipe, args.recipe_name)))
    if args.options:
        raw = json.loads(Path(args.options).read_text(encoding="utf-8"))
        options.update(parse_options(raw))
    if args.auto_enhance:
        options.update(AUTO_ENHANCE_OPTIONS)
    if args.constrain_crop:
        options["constrain_crop"] = True
    for attribute, key in _FLAG_OPTIONS.items():
        value = getattr(args, attribute)
        if value is not None:
            options[key] = value
    if args.black_white is not None:
        options.update(parse_black_white(args.black_white))
    if args.sharpen:
        options["sharpen"] = True
    if args.curve:
        options["curve"] = [list(point) for point in args.curve]
    channel_curves = {name[0]: [list(point) for point in getattr(args, f"curve_{name}")] for name in ("red", "green", "blue") if getattr(args, f"curve_{name}")}
    if channel_curves:
        options["curve_channels"] = channel_curves
    if args.crop:
        options["crop_box"] = list(args.crop)
    if args.aspect:
        options["aspect"] = f"{args.aspect[0]}:{args.aspect[1]}"
    if args.ai_upscale:
        options["ai_upscale"] = True
        options["ai_upscale_factor"] = args.ai_upscale
    if getattr(args, "bit_depth", 8) == 16:
        options["bit_depth"] = 16
    return parse_options(options)


def main() -> None:
    from app.observability import configure_logging

    configure_logging("cli")
    parser = build_parser()
    args = parser.parse_args()
    try:
        options = options_from_args(args)
    except (OptionsError, ValueError, OSError) as error:
        parser.error(str(error))
    export_settings = {
        "quality": max(1, min(100, args.quality)), "metadata": args.metadata,
        "remove_gps": args.remove_gps, "gpu_export": args.gpu_export, "color_space": args.color_space,
    }
    if options.get("bit_depth") == 16:
        export_settings["bit_depth"] = 16
    if args.color_space == "Display P3":
        options["working_space"] = "Display P3"  # B14: keep colours outside sRGB

    if args.input_dir:
        if not args.output_dir:
            raise SystemExit("--output-dir is required when using --input-dir")
        if args.mask or args.remove_mask:
            parser.error("--mask/--remove-mask are drawn for one specific photo and can't be applied to a whole folder")
        outputs = process_folder(args.input_dir, args.output_dir, options=options, export_settings=export_settings)
        print(f"Processed {len(outputs)} image(s) into: {args.output_dir}")
        return
    if not args.output:
        raise SystemExit("--output is required when using --input")

    image = decode_source(Path(args.input).read_bytes(), Path(args.input).name, high_bit=options.get("bit_depth") == 16,
                          working_space=options.get("working_space", "sRGB"))
    removal_mask = Image.open(args.remove_mask) if args.remove_mask else None
    local_mask = Image.open(args.mask) if args.mask else None
    result = run_pipeline(image, options, removal_mask=removal_mask, local_mask=local_mask)
    output_path = write_export(result, args.output, export_settings)
    print(f"Saved edited image to: {output_path}")


if __name__ == "__main__":
    main()
