from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.image_processor import PhotoProcessor


def save_for_delivery(
    processor: PhotoProcessor,
    output_path: str | Path,
    quality: int = 92,
    max_size: tuple[int, int] | None = None,
    strip_metadata: bool = False,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    quality = min(100, max(1, int(quality)))
    image = processor.image.copy()
    if max_size:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

    if output.suffix.lower() in {".jpg", ".jpeg"}:
        save_options = {"format": "JPEG", "quality": quality, "optimize": True}
        if not strip_metadata and image.info.get("icc_profile"):
            save_options["icc_profile"] = image.info["icc_profile"]
        image.convert("RGB").save(output, **save_options)
    else:
        save_options = {"optimize": True}
        if not strip_metadata:
            save_options["exif"] = image.getexif().tobytes()
            if image.info.get("icc_profile"):
                save_options["icc_profile"] = image.info["icc_profile"]
        image.save(output, **save_options)
    return output
