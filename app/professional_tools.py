from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter

def preview_image(image: Image.Image, max_size: tuple[int, int] = (1600, 1600)) -> Image.Image:
    """Return a bounded copy for responsive UI previews."""
    result = image.copy()
    result.thumbnail(max_size, Image.Resampling.LANCZOS)
    return result


EXPORT_PROFILES = {
    "Web": {"format": "JPEG", "quality": 82, "max_size": (2048, 2048), "strip_metadata": True},
    "Social": {"format": "JPEG", "quality": 88, "max_size": (1080, 1080), "strip_metadata": True},
    "Print": {"format": "TIFF", "quality": 100, "max_size": None, "strip_metadata": False},
    "Archive": {"format": "PNG", "quality": 100, "max_size": None, "strip_metadata": False},
}


def zoom_preview(image: Image.Image, zoom: float = 1.0, center: tuple[float, float] = (0.5, 0.5), max_size: tuple[int, int] = (1200, 900)) -> Image.Image:
    zoom = max(0.25, min(8.0, float(zoom)))
    if zoom <= 1.0:
        result = image.copy()
        result.thumbnail(max_size, Image.Resampling.LANCZOS)
        return result
    width, height = image.size
    crop_width = max(1, int(width / zoom))
    crop_height = max(1, int(height / zoom))
    cx, cy = max(0.0, min(1.0, center[0])), max(0.0, min(1.0, center[1]))
    left = max(0, min(width - crop_width, int(width * cx - crop_width / 2)))
    top = max(0, min(height - crop_height, int(height * cy - crop_height / 2)))
    result = image.crop((left, top, left + crop_width, top + crop_height))
    result.thumbnail(max_size, Image.Resampling.LANCZOS)
    return result


def normalize_mask(mask: Image.Image | np.ndarray, size: tuple[int, int], invert: bool = False, feather: int = 0) -> Image.Image:
    result = mask if isinstance(mask, Image.Image) else Image.fromarray(np.asarray(mask).astype(np.uint8), mode="L")
    result = result.convert("L").resize(size, Image.Resampling.NEAREST)
    if invert:
        result = ImageChops.invert(result)
    if feather > 0:
        result = result.filter(ImageFilter.GaussianBlur(max(1, int(feather))))
    return result


def apply_mask_overlay(image: Image.Image, mask: Image.Image, color: tuple[int, int, int] = (220, 55, 35), opacity: int = 35) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (*color, max(0, min(255, int(opacity)))))
    alpha = normalize_mask(mask, base.size)
    overlay.putalpha(alpha.point(lambda value: value * overlay.getchannel("A").getextrema()[1] // 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def export_profile(image: Image.Image, profile_name: str = "Web") -> tuple[bytes, str, str]:
    if profile_name not in EXPORT_PROFILES:
        raise ValueError(f"Unknown export profile: {profile_name}")
    profile = EXPORT_PROFILES[profile_name]
    result = image.copy()
    if profile["max_size"]:
        result.thumbnail(profile["max_size"], Image.Resampling.LANCZOS)
    output = io.BytesIO()
    format_name = profile["format"]
    save_kwargs = {"format": format_name}
    if format_name == "JPEG":
        save_kwargs.update(quality=profile["quality"], optimize=True)
        result = result.convert("RGB")
    result.save(output, **save_kwargs)
    extension = {"JPEG": ".jpg", "TIFF": ".tif", "PNG": ".png"}[format_name]
    return output.getvalue(), extension, "image/jpeg" if format_name == "JPEG" else "image/" + format_name.lower()


def validate_project_bytes(content: bytes) -> dict:
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Project manifest is not valid JSON") from error
    if payload.get("version") not in (1, 2) or not isinstance(payload.get("recipe", {}), dict):
        raise ValueError("Unsupported or invalid project manifest")
    return payload
