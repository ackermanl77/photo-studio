from __future__ import annotations

from pathlib import Path

from app.image_processor import PhotoProcessor
from app.presets import apply_preset

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    preset: str | None = None,
) -> list[Path]:
    source = Path(input_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for input_path in sorted(source.iterdir()):
        if not input_path.is_file() or input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        output_path = destination / input_path.name
        processor = PhotoProcessor(input_path)
        if preset:
            apply_preset(processor, preset)
        processor.save(output_path)
        outputs.append(output_path)

    return outputs
