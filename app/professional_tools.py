from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

# Fractional (0..1) line positions each named grid style draws on BOTH axes -- shared between
# draw_guides_and_grid()'s display overlay and the free-form crop editor's on-canvas snapping (see
# app/frontend_unified_app.py), so the two can never disagree about where a grid line actually sits.
GRID_PRESETS: dict[str, tuple[float, ...]] = {
    "None": (),
    "Rule of Thirds": (1 / 3, 2 / 3),
    "Grid 4x4": (0.25, 0.5, 0.75),
    "Golden Ratio": (0.381966, 0.618034),
}


def grid_line_fractions(grid_type: str) -> tuple[float, ...]:
    return GRID_PRESETS.get(grid_type, ())


def draw_guides_and_grid(
    image: Image.Image,
    grid_type: str = "None",
    guides: list[dict] | None = None,
    grid_color: tuple[int, int, int] = (255, 255, 255),
    guide_color: tuple[int, int, int] = (0, 220, 255),
) -> Image.Image:
    """A view-only composition aid -- grid lines (Rule of Thirds/4x4/Golden Ratio) plus any number
    of user-placed horizontal/vertical guides, drawn semi-transparent over the image. Purely a
    display overlay: never sent through app/backend.py's pipeline and never part of an exported
    file, the same "computed locally, canvas-only" pattern as the Masks tab's selection preview and
    the ICC gamut/clipping warning overlays. Each guide is a dict with "orientation"
    ("horizontal"/"vertical") and "position_pct" (0-100, that axis's own fraction of the frame)."""
    base = image.convert("RGBA")
    width, height = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for fraction in grid_line_fractions(grid_type):
        x, y = int(round(fraction * width)), int(round(fraction * height))
        draw.line([(x, 0), (x, height)], fill=(*grid_color, 110), width=1)
        draw.line([(0, y), (width, y)], fill=(*grid_color, 110), width=1)

    for guide in guides or []:
        position = max(0.0, min(100.0, float(guide.get("position_pct", 50.0)))) / 100.0
        if guide.get("orientation") == "horizontal":
            y = int(round(position * height))
            draw.line([(0, y), (width, y)], fill=(*guide_color, 200), width=1)
        else:
            x = int(round(position * width))
            draw.line([(x, 0), (x, height)], fill=(*guide_color, 200), width=1)

    return Image.alpha_composite(base, overlay).convert("RGB")


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


def compute_histogram(image: Image.Image, bins: int = 64) -> dict[str, np.ndarray]:
    """Per-channel pixel-value counts, binned for a compact UI chart (e.g. st.line_chart)."""
    rgb = np.asarray(image.convert("RGB"))
    return {
        channel: np.histogram(rgb[..., index], bins=bins, range=(0, 255))[0]
        for index, channel in enumerate(("Red", "Green", "Blue"))
    }


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
