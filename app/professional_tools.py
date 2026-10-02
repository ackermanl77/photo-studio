from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

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


def zoom_viewport(size: tuple[int, int], zoom: float, center: tuple[float, float] = (0.5, 0.5), max_size: tuple[int, int] = (1200, 900)) -> tuple[int, int, int, int]:
    """The (left, top, right, bottom) region of an image of `size` that a `zoom` view centered at
    `center` (fractions) shows in a `max_size` viewport -- the whole image for Fit (zoom < 1)."""
    width, height = size
    if zoom < 1.0:
        return (0, 0, width, height)
    crop_width = max(1, min(width, int(max_size[0] / zoom)))
    crop_height = max(1, min(height, int(max_size[1] / zoom)))
    cx, cy = max(0.0, min(1.0, center[0])), max(0.0, min(1.0, center[1]))
    left = max(0, min(width - crop_width, int(width * cx - crop_width / 2)))
    top = max(0, min(height - crop_height, int(height * cy - crop_height / 2)))
    return (left, top, left + crop_width, top + crop_height)


def draw_viewport_box(image: Image.Image, box: tuple[int, int, int, int], color: tuple[int, int, int] = (251, 191, 36)) -> Image.Image:
    """The zoom navigator's overview: `image` with the visible region outlined."""
    from PIL import ImageDraw

    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    left, top, right, bottom = box
    draw.rectangle((left, top, max(left + 1, right - 1), max(top + 1, bottom - 1)), outline=color, width=2)
    return result


def zoom_preview(image: Image.Image, zoom: float = 0.0, center: tuple[float, float] = (0.5, 0.5), max_size: tuple[int, int] = (1200, 900)) -> Image.Image:
    """The canvas view. `zoom` < 1 is Fit (the whole image, downscaled to `max_size`); 1.0 is a true
    1:1 view -- a viewport-sized region at the image's own pixels, never resampled; 2.0/4.0 enlarge
    that region with nearest-neighbour so individual pixels stay visible. (100 % used to mean "fit",
    and 200 % cropped a region and then *downscaled* it, so actual pixels were never shown.)"""
    zoom = max(0.0, min(8.0, float(zoom)))
    if zoom < 1.0:
        result = image.copy()
        result.thumbnail(max_size, Image.Resampling.LANCZOS)
        return result
    result = image.crop(zoom_viewport(image.size, zoom, center, max_size))
    if zoom > 1.0:
        result = result.resize((max(1, int(result.width * zoom)), max(1, int(result.height * zoom))), Image.Resampling.NEAREST)
    return result


def compute_histogram(image: Image.Image, bins: int = 64) -> dict[str, np.ndarray]:
    """Per-channel pixel-value counts, binned for a compact UI chart (e.g. st.line_chart)."""
    rgb = np.asarray(image.convert("RGB"))
    return {
        channel: np.histogram(rgb[..., index], bins=bins, range=(0, 255))[0]
        for index, channel in enumerate(("Red", "Green", "Blue"))
    }


WAVEFORM_MODES: tuple[str, ...] = ("Luma", "RGB")


def compute_waveform(image: Image.Image, mode: str = "Luma", height: int = 256) -> Image.Image:
    """A video-style waveform monitor: horizontal pixel position (x) against the distribution of
    luminance ("Luma") or per-channel ("RGB") intensity found in that column, the same concept as
    a real waveform scope used for exposure/levels analysis. Unlike compute_histogram() (one
    aggregate count per value bin across the whole image), this keeps the x-axis so a gradient,
    a blown-out sky, or a clipped column stands out at the position it actually occupies in the
    frame.

    Returns a ready-to-display RGB image sized (image.width, height) -- row 0 is the brightest
    value (255) and the bottom row is the darkest (0), matching how a real waveform monitor
    plots white at the top. Computed entirely with NumPy (one vectorized 2D-histogram-style
    scatter per channel via `np.add.at`, no per-column Python loop) since this can run on every
    canvas rerun just like compute_histogram(). Each channel's own column-independent pixel
    counts are normalized against that channel's single brightest bin (not each column
    separately) and gamma-compressed (`sqrt`) for display, then combined across channels with a
    per-pixel maximum (a "screen"-like combine) rather than a sum, so a bright trace in one
    channel doesn't wash out or clip against another's."""
    if mode not in WAVEFORM_MODES:
        mode = "Luma"
    rgb = np.asarray(image.convert("RGB")).astype(np.float64)
    src_height, width = rgb.shape[0], rgb.shape[1]
    bins = max(1, int(height))

    if mode == "Luma":
        channels = {"Luma": rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114}
        colors = {"Luma": (235.0, 235.0, 235.0)}
    else:
        channels = {"Red": rgb[..., 0], "Green": rgb[..., 1], "Blue": rgb[..., 2]}
        colors = {"Red": (255.0, 70.0, 70.0), "Green": (70.0, 255.0, 70.0), "Blue": (70.0, 130.0, 255.0)}

    canvas = np.zeros((bins, width, 3), dtype=np.float64)
    col_indices = np.broadcast_to(np.arange(width), (src_height, width))

    for name, values in channels.items():
        rows = np.clip(np.round((255.0 - values) / 255.0 * (bins - 1)).astype(np.int64), 0, bins - 1)
        counts = np.zeros((bins, width), dtype=np.float64)
        np.add.at(counts, (rows.ravel(), col_indices.ravel()), 1.0)
        peak = counts.max()
        normalized = np.sqrt(counts / peak) if peak > 0 else counts
        color = np.array(colors[name], dtype=np.float64)
        canvas = np.maximum(canvas, normalized[..., None] * color[None, None, :])

    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")


def normalize_mask(mask: Image.Image | np.ndarray, size: tuple[int, int], invert: bool = False, feather: int = 0) -> Image.Image:
    result = mask if isinstance(mask, Image.Image) else Image.fromarray(np.asarray(mask).astype(np.uint8), mode="L")
    result = result.convert("L").resize(size, Image.Resampling.NEAREST)
    if invert:
        result = ImageChops.invert(result)
    if feather > 0:
        result = result.filter(ImageFilter.GaussianBlur(max(1, int(feather))))
    return result


OVERLAY_STYLES = ("Tint", "Hatched", "Outline")


def apply_mask_overlay(image: Image.Image, mask: Image.Image, color: tuple[int, int, int] = (220, 55, 35), opacity: int = 35, style: str = "Tint") -> Image.Image:
    """Show a selection on `image`. "Tint" washes it with `color`; "Hatched" draws diagonal stripes
    and "Outline" its edge -- both readable without telling colors apart (F11)."""
    base = image.convert("RGBA")
    alpha = normalize_mask(mask, base.size)
    if style == "Hatched":
        width, height = base.size
        yy, xx = np.mgrid[0:height, 0:width]
        stripes = (((xx + yy) // 6) % 2 == 0).astype(np.float32)
        strength = np.asarray(alpha, dtype=np.float32) / 255.0 * stripes * 0.85
        layer = Image.new("RGBA", base.size, (*color, 255))
        layer.putalpha(Image.fromarray((strength * 255).astype(np.uint8), mode="L"))
        return Image.alpha_composite(base, layer).convert("RGB")
    if style == "Outline":
        binary = (np.asarray(alpha) >= 128).astype(np.uint8)
        edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edge = cv2.dilate(edge, np.ones((2, 2), np.uint8))
        layer = Image.new("RGBA", base.size, (*color, 255))
        layer.putalpha(Image.fromarray(edge * 255, mode="L"))
        return Image.alpha_composite(base, layer).convert("RGB")
    overlay = Image.new("RGBA", base.size, (*color, max(0, min(255, int(opacity)))))
    overlay.putalpha(alpha.point(lambda value: value * overlay.getchannel("A").getextrema()[1] // 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


WATERMARK_POSITIONS: tuple[str, ...] = ("Top Left", "Top Right", "Bottom Left", "Bottom Right", "Center")


def apply_watermark(
    image: Image.Image,
    text: str,
    position: str = "Bottom Right",
    opacity: int = 50,
    font_size: int = 32,
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Stamp semi-transparent text onto a copy of `image` -- a lightweight branding/proofing
    mark for exports, not a tamper-resistant signature. A blank/whitespace-only `text` is a
    no-op (returns an unmodified copy), so callers can wire this straight to an optional export
    control without a separate "watermark enabled" checkbox. Drawn on its own full-size RGBA
    layer and alpha-composited once, the same approach apply_mask_overlay() already uses, so the
    text's own antialiasing blends correctly at any opacity instead of each glyph's edge pixels
    being individually thresholded. `opacity` is 0-100 (a percentage, matching a UI slider
    labeled "Opacity %"), unlike apply_mask_overlay()'s own `opacity`, which is a direct 0-255
    alpha value -- the two aren't meant to share a caller."""
    if not text or not text.strip():
        return image.copy()
    base = image.convert("RGBA")
    width, height = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    size = max(1, int(font_size))
    # The same TrueType lookup Text layers use (app.layer_workspace._load_font), so a watermark
    # and a text layer look alike; it falls back to Pillow's own font when none is installed.
    from app.layer_workspace import _load_font

    font = _load_font(size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = max(4, int(size * 0.4))
    positions = {
        "Top Left": (margin, margin),
        "Top Right": (max(margin, width - text_width - margin), margin),
        "Bottom Left": (margin, max(margin, height - text_height - margin)),
        "Bottom Right": (max(margin, width - text_width - margin), max(margin, height - text_height - margin)),
        "Center": (max(0, (width - text_width) // 2), max(0, (height - text_height) // 2)),
    }
    x, y = positions.get(position, positions["Bottom Right"])
    alpha = max(0, min(255, int(round(opacity / 100.0 * 255))))
    draw.text((x - bbox[0], y - bbox[1]), text, font=font, fill=(*color, alpha))
    return Image.alpha_composite(base, overlay).convert("RGB")


def export_profile(
    image: Image.Image,
    profile_name: str = "Web",
    watermark_text: str | None = None,
    watermark_position: str = "Bottom Right",
    watermark_opacity: int = 50,
    watermark_font_size: int = 32,
    exif: bytes | None = None,
    remove_gps: bool = False,
) -> tuple[bytes, str, str]:
    """`remove_gps` drops the GPS block from a profile that keeps EXIF (Print/Archive; Web and
    Social strip all EXIF anyway). `exif` overrides `image.info["exif"]` -- the frontend passes the *source* photo's EXIF,
    since the processed canvas image it exports isn't guaranteed to carry any (the regression
    test this used to have only passed because it injected image.info["exif"] by hand)."""
    if profile_name not in EXPORT_PROFILES:
        raise ValueError(f"Unknown export profile: {profile_name}")
    profile = EXPORT_PROFILES[profile_name]
    result = image.copy()
    if profile["max_size"]:
        result.thumbnail(profile["max_size"], Image.Resampling.LANCZOS)
    if watermark_text and watermark_text.strip():
        # Applied after the resize above (not before) so position/margin/font-size are relative
        # to the actual exported pixel dimensions of each profile, not the source's.
        result = apply_watermark(result, watermark_text, watermark_position, watermark_opacity, watermark_font_size)
    output = io.BytesIO()
    format_name = profile["format"]
    save_kwargs = {"format": format_name}
    if format_name == "JPEG":
        save_kwargs.update(quality=profile["quality"], optimize=True)
        result = result.convert("RGB")
    # Metadata is read from the *input* image's info (a resize/watermark/convert above may not
    # carry it). The profile's strip_metadata flag used to be ignored entirely -- EXIF was never
    # written, even for Print/Archive. Now EXIF (camera data, capture time, possibly GPS) is kept
    # unless the profile strips it; the ICC profile is always kept, since it isn't personal data
    # and dropping it would change how a non-sRGB export's colors are displayed.
    # prepare_exif() fixes the kept EXIF up for the exported file (Orientation 1, the exported
    # pixel dimensions, no stale embedded thumbnail) -- see app.export.
    from app.export import prepare_exif

    if image.info.get("icc_profile"):
        save_kwargs["icc_profile"] = image.info["icc_profile"]
    exif_out = prepare_exif(exif if exif is not None else image.info.get("exif"), result.size, "none" if profile["strip_metadata"] else "all", remove_gps)
    if exif_out:
        save_kwargs["exif"] = exif_out
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
