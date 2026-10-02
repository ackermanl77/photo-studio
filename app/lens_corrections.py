from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Deliberately not a real per-lens-model profile database (Adobe's and Lensfun's are built from
# calibrating thousands of specific lens/camera combinations against external data this app has no
# access to, offline-first as it is). These are generic, typical-of-the-category starting points a
# user can fine-tune with the sliders afterward -- the same fallback role Lightroom's own "Manual"
# lens-correction tab serves when no matching profile exists.
LENS_PROFILES = {
    "None": {"distortion": 0, "vignette_amount": 0, "vignette_midpoint": 50, "chromatic_aberration": 0},
    "Wide-angle prime (heavy barrel)": {"distortion": -35, "vignette_amount": -25, "vignette_midpoint": 45, "chromatic_aberration": 15},
    "Standard zoom (mild barrel)": {"distortion": -12, "vignette_amount": -12, "vignette_midpoint": 50, "chromatic_aberration": 8},
    "Telephoto zoom (mild pincushion)": {"distortion": 8, "vignette_amount": -8, "vignette_midpoint": 55, "chromatic_aberration": 5},
    "Fisheye": {"distortion": -60, "vignette_amount": -35, "vignette_midpoint": 40, "chromatic_aberration": 20},
}


def correct_distortion(image: Image.Image, amount: float) -> Image.Image:
    """Radial barrel/pincushion correction. `amount` -100..100: negative pulls a barrel-distorted
    image's edges back in (corrects barrel), positive pushes them back out (corrects pincushion) --
    0 is a no-op. Uses cv2.undistort with a synthetic camera matrix (no real per-lens calibration,
    just a single radial term (k1) driven by the slider) and a focal length long enough that the
    correction stays gentle across the whole -100..100 range instead of blowing up at the extremes.
    """
    amount = max(-100.0, min(100.0, float(amount)))
    if amount == 0:
        return image
    array = np.array(image.convert("RGB"))
    height, width = array.shape[:2]
    focal = float(max(width, height))
    camera_matrix = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float32)
    k1 = (amount / 100.0) * 0.5
    dist_coeffs = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    corrected = cv2.undistort(array, camera_matrix, dist_coeffs)
    return Image.fromarray(corrected, mode="RGB")


def correct_vignette(image: Image.Image, amount: float, midpoint: float = 50.0) -> Image.Image:
    """Radial vignette correction. `amount` -100..100: negative darkens the corners (adds a
    vignette), positive brightens them (removes one) -- 0 is a no-op. `midpoint` 0..100 controls
    how far from center the effect starts (higher = concentrated closer to the corners only)."""
    amount = max(-100.0, min(100.0, float(amount)))
    if amount == 0:
        return image
    array = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = array.shape[:2]
    center_x, center_y = width / 2.0, height / 2.0
    max_distance = np.hypot(center_x, center_y)
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.hypot(xx - center_x, yy - center_y) / max_distance
    midpoint_norm = max(0.01, min(0.99, float(midpoint) / 100.0))
    falloff = np.clip((distance - midpoint_norm) / (1.0 - midpoint_norm), 0.0, 1.0)
    factor = 1.0 + (amount / 100.0) * falloff
    array *= factor[..., None]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def correct_chromatic_aberration(image: Image.Image, amount: float) -> Image.Image:
    """Fixes red/cyan and blue/yellow fringing near high-contrast edges by radially scaling the Red
    and Blue channels a small amount in opposite directions relative to Green, which stays fixed as
    the reference channel. `amount` -100..100 (0 is a no-op); the max scale at +-100 is deliberately
    small (1%) since real fringing offsets are subtle -- a heavy-handed shift just introduces new
    fringing in the other direction instead of removing the original one."""
    amount = max(-100.0, min(100.0, float(amount)))
    if amount == 0:
        return image
    array = np.array(image.convert("RGB"))
    height, width = array.shape[:2]
    center_x, center_y = width / 2.0, height / 2.0
    scale = (amount / 100.0) * 0.01
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    def _radially_scaled(channel: np.ndarray, factor: float) -> np.ndarray:
        map_x = center_x + (grid_x - center_x) * (1.0 - factor)
        map_y = center_y + (grid_y - center_y) * (1.0 - factor)
        return cv2.remap(channel, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    red = _radially_scaled(array[..., 0], scale)
    blue = _radially_scaled(array[..., 2], -scale)
    corrected = np.stack([red, array[..., 1], blue], axis=-1)
    return Image.fromarray(corrected, mode="RGB")


def _perspective_warp(array: np.ndarray, src: np.ndarray, dst: np.ndarray, border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    height, width = array.shape[:2]
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(array, matrix, (width, height), borderMode=border)


def correct_perspective(image: Image.Image, vertical: float = 0.0, horizontal: float = 0.0, border: int = cv2.BORDER_REPLICATE) -> Image.Image:
    """Perspective/keystone correction for converging lines -- darktable's rotate-and-perspective
    module ("corrects converging lines by warping the image so lines in question become parallel
    to the image frame"), RawTherapee's/Lightroom's Vertical/Horizontal Perspective sliders. Unlike
    the radial corrections above, this needs no lens/camera data at all -- it's purely a function of
    the shooting angle, so there's no companion LENS_PROFILES entry for it (a keystone-corrected
    profile default wouldn't mean anything; every photo needs its own amount).

    `vertical` -100..100 corrects converging vertical lines (e.g. a building shot from below, where
    verticals lean inward toward the top): positive narrows the top edge toward the bottom edge's
    width, negative narrows the bottom edge instead. `horizontal` -100..100 does the same for
    converging horizontal lines (a wall or shelf shot at an angle): positive narrows the left edge's
    height toward the right edge's, negative narrows the right edge instead. Applied as two
    independent single-axis perspective warps (cv2.getPerspectiveTransform/warpPerspective) one
    after the other rather than solving one combined quad -- easier to reason about and test each
    axis in isolation, and the two are visually independent enough in practice that composing them
    sequentially is indistinguishable from a single combined solve. 0/0 is a no-op. The maximum
    shift at +-100 is capped at 30% of the corrected edge's own length so even the extreme end of
    the slider stays a recognizable keystone correction rather than degenerating toward a sliver.
    `border` is how uncovered edges are filled (edge pixels replicated by default); app.geometry
    passes BORDER_CONSTANT to measure the uncovered area for "constrain crop".
    """
    vertical = max(-100.0, min(100.0, float(vertical)))
    horizontal = max(-100.0, min(100.0, float(horizontal)))
    if not vertical and not horizontal:
        return image
    array = np.array(image.convert("RGB"))
    height, width = array.shape[:2]
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    if vertical:
        shift = abs(vertical) / 100.0 * (width * 0.3)
        if vertical > 0:
            dst = np.float32([[shift, 0], [width - shift, 0], [width, height], [0, height]])
        else:
            dst = np.float32([[0, 0], [width, 0], [width - shift, height], [shift, height]])
        array = _perspective_warp(array, src, dst, border)

    if horizontal:
        shift = abs(horizontal) / 100.0 * (height * 0.3)
        if horizontal > 0:
            dst = np.float32([[0, shift], [width, 0], [width, height], [0, height - shift]])
        else:
            dst = np.float32([[0, 0], [width, shift], [width, height - shift], [0, height]])
        array = _perspective_warp(array, src, dst, border)

    return Image.fromarray(array, mode="RGB")


def apply_lens_corrections(
    image: Image.Image,
    distortion: float = 0.0,
    vignette_amount: float = 0.0,
    vignette_midpoint: float = 50.0,
    chromatic_aberration: float = 0.0,
    perspective_vertical: float = 0.0,
    perspective_horizontal: float = 0.0,
) -> Image.Image:
    """Applies whichever corrections are non-zero, in the order a real lens-correction workflow
    would: geometric distortion first, then perspective/keystone (another geometric warp -- doing
    it right after distortion means it operates on already-straightened lines, matching how a real
    workflow corrects barrel/pincushion before worrying about converging verticals), then chromatic
    aberration (radial, assumes the now-corrected geometry), then vignette (pure brightness, order-
    independent relative to the others)."""
    if distortion:
        image = correct_distortion(image, distortion)
    if perspective_vertical or perspective_horizontal:
        image = correct_perspective(image, perspective_vertical, perspective_horizontal)
    if chromatic_aberration:
        image = correct_chromatic_aberration(image, chromatic_aberration)
    if vignette_amount:
        image = correct_vignette(image, vignette_amount, vignette_midpoint)
    return image
