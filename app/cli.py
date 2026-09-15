from __future__ import annotations

import argparse

from app.batch import process_folder
from app.export import save_for_delivery
from app.image_processor import PhotoProcessor
from app.presets import PRESETS, apply_preset


def parse_curve(value: str) -> list[tuple[int, int]]:
    try:
        return [tuple(int(part) for part in pair.split(":")) for pair in value.split(",")]
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Curves must use x:y pairs, for example 0:0,128:140,255:255") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Professional photo editing pipeline")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input image path")
    input_group.add_argument("--input-dir", help="Folder of input images")
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", help="Output image path")
    output_group.add_argument("--output-dir", help="Folder for processed images")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness multiplier (default: 1.0)")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier (default: 1.0)")
    parser.add_argument("--saturation", type=float, default=1.0, help="Saturation multiplier (default: 1.0)")
    parser.add_argument("--exposure", type=float, default=1.0, help="Exposure multiplier (default: 1.0)")
    parser.add_argument("--highlights", type=float, default=0.0, help="Highlights adjustment from -255 to 255")
    parser.add_argument("--shadows", type=float, default=0.0, help="Shadows adjustment from -255 to 255")
    parser.add_argument("--texture", type=float, default=0.0, help="Texture adjustment from -100 to 100")
    parser.add_argument("--clarity", type=float, default=0.0, help="Clarity adjustment from -100 to 100")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature from -100 cool to 100 warm")
    parser.add_argument("--tint", type=float, default=0.0, help="Tint from -100 green to 100 magenta")
    parser.add_argument("--curve", type=parse_curve, help="RGB curve points such as 0:0,128:140,255:255")
    parser.add_argument("--remove-mask", help="Grayscale mask path for CPU object removal")
    parser.add_argument("--mask", help="Grayscale mask path for local adjustments")
    parser.add_argument("--mask-exposure", type=float, default=1.0, help="Masked exposure multiplier")
    parser.add_argument("--mask-contrast", type=float, default=1.0, help="Masked contrast multiplier")
    parser.add_argument("--mask-saturation", type=float, default=1.0, help="Masked saturation multiplier")
    parser.add_argument("--denoise", type=int, default=0, help="Median filter denoise strength (default: 0)")
    parser.add_argument("--sharpen", action="store_true", help="Apply sharpening")
    parser.add_argument("--auto-enhance", action="store_true", help="Apply a standard portrait-friendly enhancement pass")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Apply a named editing preset")
    parser.add_argument("--quality", type=int, default=92, help="JPEG delivery quality from 1 to 100")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"), help="Crop rectangle in pixels")
    parser.add_argument("--aspect", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), help="Center-crop to an aspect ratio")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.input_dir:
        if not args.output_dir:
            raise SystemExit("--output-dir is required when using --input-dir")
        outputs = process_folder(args.input_dir, args.output_dir, preset=args.preset)
        print(f"Processed {len(outputs)} image(s) into: {args.output_dir}")
        return

    processor = PhotoProcessor(args.input)

    processor.apply_brightness(args.brightness)
    processor.apply_contrast(args.contrast)
    processor.apply_saturation(args.saturation)
    processor.apply_exposure(args.exposure)
    processor.apply_highlights_shadows(args.highlights, args.shadows)
    # Matches app/backend.py::process_image()'s pipeline order: preset is applied right after this
    # Tone group and before Color/Detail/Crop, not last. A preset (app/presets.py) applies its own
    # exposure/contrast/saturation/curve/sharpen adjustments, which are order-sensitive relative to
    # every step below -- applying it last (as this used to) meant --preset combined with any other
    # flag produced a visibly different result via the CLI than the same combination produced via
    # the GUI/backend, on the exact same source image.
    if args.preset:
        apply_preset(processor, args.preset)
    processor.apply_texture_clarity(args.texture, args.clarity)
    processor.apply_temperature_tint(args.temperature, args.tint)
    if args.curve:
        processor.apply_curves(args.curve)
    if args.remove_mask:
        processor.remove_objects(args.remove_mask)
    if args.mask:
        processor.apply_masked_adjustments(
            args.mask,
            exposure=args.mask_exposure,
            contrast=args.mask_contrast,
            saturation=args.mask_saturation,
        )

    if args.denoise > 0:
        processor.apply_denoise(args.denoise)

    if args.sharpen:
        processor.apply_sharpen()

    if args.auto_enhance:
        processor.auto_enhance()

    if args.crop:
        processor.crop(*args.crop)
    if args.aspect:
        processor.fit_aspect_ratio(*args.aspect)

    output_path = save_for_delivery(processor, args.output, quality=args.quality)
    print(f"Saved edited image to: {output_path}")


if __name__ == "__main__":
    main()
