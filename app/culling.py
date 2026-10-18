"""Culling scores for the Library (F20): how sharp a photo's sharpest part is, and how much of it is
clipped -- the classical half of Lightroom's Assisted Culling, no model needed.

- **Focus** (0-100): the variance of the Laplacian per tile of a ~1024 px preview, taking the 90th
  percentile tile, so a shallow-depth-of-field portrait with a sharp face scores as sharp even when
  most of the frame is deliberately soft. The flattest tiles' variance (the noise floor) is
  subtracted and a light blur comes first, so sensor grain doesn't read as detail. The scale is
  logarithmic: below BLURRY_FOCUS a photo is flagged as likely blurry.
- **Highlights / Shadows**: the share (percent) of pixels at the luminance extremes (>= 250, <= 5).

Scores are computed during "Scan Folder" from the same preview the thumbnails use (RAW: the embedded
JPEG) and cached in the catalog against the file's modification time. Streamlit-free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.raw_import import is_raw_path

PREVIEW_EDGE = 1024
TILES = 8
# Focus = 100 * (log10(detail) - 1) / 3, clipped: detail (Laplacian variance on 0-255 gray, above
# the noise floor) of 10 -> 0, 10,000 -> 100. Calibrated on real photos at preview size: sharp ones
# score ~80-100, a 1 px blur ~55-85, 2 px ~30-60, 4 px under ~25.
FOCUS_LOG_RANGE = (1.0, 4.0)
BLURRY_FOCUS = 35.0


def score_image(image: Image.Image) -> dict[str, float]:
    """{"focus": 0-100, "highlights": %, "shadows": %} for a PIL image of any size."""
    import cv2

    preview = image.convert("RGB")
    if max(preview.size) > PREVIEW_EDGE:
        preview = preview.copy()
        preview.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.BILINEAR)
    rgb = np.asarray(preview, dtype=np.float32)
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = cv2.Laplacian(cv2.GaussianBlur(luma, (3, 3), 0), cv2.CV_32F, ksize=3)
    height, width = lap.shape
    tiles = []
    for ty in range(TILES):
        for tx in range(TILES):
            tile = lap[ty * height // TILES:(ty + 1) * height // TILES, tx * width // TILES:(tx + 1) * width // TILES]
            if tile.size:
                tiles.append(float(tile.var()))
    # The sharpest tiles' detail above the flattest tiles' (the noise floor, ~ sensor grain).
    detail = max(0.0, float(np.percentile(tiles, 90) - np.percentile(tiles, 10))) if tiles else 0.0
    low, high = FOCUS_LOG_RANGE
    focus = 100.0 * min(1.0, max(0.0, (np.log10(detail + 1e-9) - low) / (high - low)))
    return {
        "focus": round(float(focus), 1),
        "highlights": round(100.0 * float((luma >= 250).mean()), 2),
        "shadows": round(100.0 * float((luma <= 5).mean()), 2),
    }


def preview_for_scoring(path: str | Path) -> Image.Image:
    """A preview of `path` about PREVIEW_EDGE px across, decoded cheaply: JPEG via draft mode, RAW via
    its embedded preview (falling back to a half-size demosaic), orientation applied."""
    source = Path(path)
    if is_raw_path(source):
        from app.library import _raw_preview

        preview = _raw_preview(source)
        if preview is None:
            raise OSError("RAW support (rawpy) isn't installed")
        return preview
    with Image.open(source) as image:
        image.draft("RGB", (PREVIEW_EDGE, PREVIEW_EDGE))
        return ImageOps.exif_transpose(image).convert("RGB")


def score_file(path: str | Path) -> dict[str, float]:
    return score_image(preview_for_scoring(path))


def is_likely_blurry(focus: float | None) -> bool:
    return focus is not None and 0 <= focus < BLURRY_FOCUS
