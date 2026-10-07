"""Spot healing (F13): replace each selected spot with the best-matching nearby patch, blended in
with Poisson (gradient-domain) cloning -- Lightroom's "Heal" next to Repair's inpaint "Fill".

For every connected region of the mask, candidate source patches of the same size are tried on a
grid around it (none overlapping any selected pixel); the one whose surroundings best match the
target's surroundings (sum of squared differences over the ring between the region and its padded
box) is copied over with cv2.seamlessClone, which matches its colours and brightness to the
target's border. Real texture (skin pores, grass, sand) comes back, where inpaint smears it. A
region too large for any clean source patch falls back to inpaint. CPU-only, classical.
"""
from __future__ import annotations

import cv2
import numpy as np

# Search window around a region, in multiples of its size; grid steps per region size.
_SEARCH_SPAN = 3.0
_STEPS_PER_SIZE = 4
# Largest region (as a fraction of the image's pixels) healed by patch; bigger ones are inpainted.
_MAX_REGION_FRACTION = 0.08


def _best_source(image: np.ndarray, selected: np.ndarray, box: tuple[int, int, int, int], ring: np.ndarray) -> tuple[int, int] | None:
    """Top-left corner of the source patch whose ring best matches the target's, or None."""
    height, width = selected.shape
    x0, y0, w, h = box
    target = image[y0:y0 + h, x0:x0 + w].astype(np.float32)
    span = int(_SEARCH_SPAN * max(w, h))
    step = max(1, max(w, h) // _STEPS_PER_SIZE)
    # A summed-area table makes "does this window touch the selection?" O(1) per candidate.
    integral = cv2.integral(selected.astype(np.uint8))
    best, best_score = None, np.inf
    weights = ring[..., None].astype(np.float32)
    for sy in range(max(0, y0 - span), min(height - h, y0 + span) + 1, step):
        for sx in range(max(0, x0 - span), min(width - w, x0 + span) + 1, step):
            if integral[sy + h, sx + w] - integral[sy, sx + w] - integral[sy + h, sx] + integral[sy, sx]:
                continue
            candidate = image[sy:sy + h, sx:sx + w].astype(np.float32)
            score = float((((candidate - target) ** 2) * weights).sum())
            if score < best_score:
                best, best_score = (sx, sy), score
    return best


def heal(image: np.ndarray, mask: np.ndarray, feather: float = 3.0) -> np.ndarray:
    """Heal every region of `mask` (uint8, >127 = selected) in `image` (uint8 RGB, H x W x 3).
    `feather` (pixels) widens each region a little before cloning so its edge blends."""
    selected = mask > 127
    if not selected.any():
        return image
    height, width = selected.shape
    radius = max(1, int(round(feather)))
    grown = cv2.dilate(selected.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    result = image.copy()
    leftovers = np.zeros_like(grown)
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        pad = max(3, int(round(0.25 * max(w, h))))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
        region = labels[y0:y1, x0:x1] == label
        # seamlessClone needs the region clear of the patch border, and a clean source.
        clear_of_border = x0 > 0 and y0 > 0 and x1 < width and y1 < height and region[[0, -1], :].sum() == 0 and region[:, [0, -1]].sum() == 0
        source = None
        if clear_of_border and area <= _MAX_REGION_FRACTION * width * height:
            ring = ~region & ~selected[y0:y1, x0:x1]
            source = _best_source(result, grown.astype(bool), (x0, y0, x1 - x0, y1 - y0), ring)
        if source is None:
            leftovers[labels == label] = 255
            continue
        sx, sy = source
        patch = result[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
        patch_mask = region.astype(np.uint8) * 255
        center = (x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2)
        try:
            # OpenCV's seamlessClone works in BGR; the channel order doesn't matter to the math.
            result = cv2.seamlessClone(np.ascontiguousarray(patch), result, patch_mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            leftovers[labels == label] = 255
    if leftovers.any():
        result = cv2.inpaint(result, leftovers, float(radius), cv2.INPAINT_TELEA)
    return result
