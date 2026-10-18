from __future__ import annotations

from pathlib import Path

from app.export import write_export
from app.pipeline import decode_source, run_pipeline
from app.raw_import import STANDARD_EXTENSIONS, SUPPORTED_EXTENSIONS


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    preset: str | None = None,
    options: dict | None = None,
    export_settings: dict | None = None,
) -> list[Path]:
    """Run every supported image in `input_dir` through app.pipeline.run_pipeline() with the same
    `options` (plus `preset`, if given) and write it to `output_dir` under the same name via the
    metadata-aware writer (app.export.write_export -- quality 92 and the source's EXIF/ICC by
    default; this used to save with PIL's defaults: JPEG quality 75, no metadata). Only `preset`
    used to be applied here, which is why the CLI's `--input-dir` silently ignored every other
    flag. A RAW source is written as .jpg (RAW can't be written back)."""
    source = Path(input_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run_options = dict(options or {})
    if preset:
        run_options["preset"] = preset
    if (export_settings or {}).get("color_space") == "Display P3":
        run_options["working_space"] = "Display P3"  # B14, like the backend's export path
    outputs: list[Path] = []

    for input_path in sorted(source.iterdir()):
        suffix = input_path.suffix.lower()
        if not input_path.is_file() or suffix not in SUPPORTED_EXTENSIONS:
            continue
        output_path = destination / (input_path.name if suffix in STANDARD_EXTENSIONS else f"{input_path.stem}.jpg")
        image = decode_source(input_path.read_bytes(), input_path.name, high_bit=run_options.get("bit_depth") == 16,
                              working_space=run_options.get("working_space", "sRGB"))
        write_export(run_pipeline(image, run_options), output_path, export_settings)
        outputs.append(output_path)

    return outputs
