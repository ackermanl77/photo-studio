"""Geometry: orientation (90° rotation, flips), lens/perspective warps, straighten, and "constrain
crop" -- the Geometry step of app.pipeline.run_pipeline(), and the base image the frontend's Crop,
Masks and Repair tools draw against (they must agree with the pipeline on which pixel is where).

Order: rotate/flip first (it's how the photo is oriented, like EXIF orientation), then the lens
corrections (distortion, perspective, chromatic aberration, vignette -- app.lens_corrections),
then the straighten rotation, then -- when `constrain_crop` is on -- a crop to the largest centered
rectangle with the frame's aspect ratio that contains no area the warps left uncovered. Without it,
distortion +35 left ~7 % pure-black corners and keystone/straighten smeared or blackened the edges
(B9 in the improvement report).

Everything here is resolution-independent (angles, fractions of the frame), so a reduced-size
preview proxy and the full-resolution export get the same geometry.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from app.lens_corrections import apply_lens_corrections, correct_distortion, correct_perspective

ROTATIONS = (0, 90, 180, 270)
STRAIGHTEN_LIMIT = 45.0
# Long edge of the white probe image the valid (covered) region is measured on -- a fixed size, so
# the crop fractions are identical whether the pipeline runs on a proxy or at full resolution.
_PROBE_LONG_EDGE = 600

GEOMETRY_OPTION_KEYS = (
    "rotate", "flip_horizontal", "flip_vertical", "straighten", "constrain_crop",
    "lens_distortion", "lens_vignette_amount", "lens_vignette_midpoint", "lens_chromatic_aberration",
    "lens_perspective_v", "lens_perspective_h",
)


def orient(image: Image.Image, rotate: int = 0, flip_horizontal: bool = False, flip_vertical: bool = False) -> Image.Image:
    """`rotate` is clockwise degrees (0/90/180/270) -- what a "rotate right" button adds. `image`
    may also be a float HxWx3 array (app.high_bit's 16-bit path)."""
    rotate = int(rotate) % 360
    if isinstance(image, np.ndarray):
        if rotate:
            image = np.rot90(image, -(rotate // 90))
        if flip_horizontal:
            image = image[:, ::-1]
        if flip_vertical:
            image = image[::-1]
        return np.ascontiguousarray(image)
    if rotate == 90:
        image = image.transpose(Image.Transpose.ROTATE_270)  # PIL rotates counter-clockwise
    elif rotate == 180:
        image = image.transpose(Image.Transpose.ROTATE_180)
    elif rotate == 270:
        image = image.transpose(Image.Transpose.ROTATE_90)
    if flip_horizontal:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_vertical:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image


def oriented_size(size: tuple[int, int], rotate: int = 0) -> tuple[int, int]:
    return (size[1], size[0]) if int(rotate) % 180 == 90 else tuple(size)


def straighten(image: Image.Image, angle: float, border: int = cv2.BORDER_CONSTANT) -> Image.Image:
    """Rotate by `angle` degrees about the center, keeping the frame size (positive = clockwise,
    the direction that levels a horizon tilted down to the right... i.e. counter-rotates a tilt)."""
    angle = max(-STRAIGHTEN_LIMIT, min(STRAIGHTEN_LIMIT, float(angle)))
    if not angle:
        return image
    is_array = isinstance(image, np.ndarray)
    array = np.ascontiguousarray(image, dtype=np.float32) if is_array else np.array(image.convert("RGB"))
    height, width = array.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -angle, 1.0)
    rotated = cv2.warpAffine(array, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=border)
    if is_array:
        return np.clip(rotated, 0.0, 1.0)  # cubic interpolation overshoots at hard edges
    return Image.fromarray(rotated, mode="RGB")


def _needs_constraint(options: dict) -> bool:
    return bool(options.get("lens_distortion") or options.get("lens_perspective_v") or options.get("lens_perspective_h") or options.get("straighten"))


@lru_cache(maxsize=64)
def _constrain_fractions_cached(aspect_milli: int, distortion: float, perspective_v: float, perspective_h: float, angle: float) -> tuple[float, float, float, float]:
    aspect = aspect_milli / 1000.0
    if aspect >= 1:
        width, height = _PROBE_LONG_EDGE, max(8, round(_PROBE_LONG_EDGE / aspect))
    else:
        width, height = max(8, round(_PROBE_LONG_EDGE * aspect)), _PROBE_LONG_EDGE
    probe = Image.new("RGB", (width, height), (255, 255, 255))
    if distortion:
        probe = correct_distortion(probe, distortion)  # cv2.undistort leaves black where uncovered
    if perspective_v or perspective_h:
        probe = correct_perspective(probe, perspective_v, perspective_h, border=cv2.BORDER_CONSTANT)
    if angle:
        probe = straighten(probe, angle, border=cv2.BORDER_CONSTANT)
    valid = (np.asarray(probe.convert("L")) >= 250).astype(np.uint8)
    integral = cv2.integral(valid)

    def fits(scale: float) -> bool:
        w, h = max(1, int(width * scale)), max(1, int(height * scale))
        x0, y0 = (width - w) // 2, (height - h) // 2
        x1, y1 = x0 + w, y0 + h
        total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        return total == w * h

    low, high = 0.05, 1.0
    if fits(high):
        return (0.0, 0.0, 1.0, 1.0)
    for _ in range(18):
        middle = (low + high) / 2
        if fits(middle):
            low = middle
        else:
            high = middle
    # A one-probe-pixel safety margin so interpolation at the very edge can't show a dark seam.
    scale = max(0.05, low - 1.0 / _PROBE_LONG_EDGE)
    margin_x, margin_y = (1 - scale) / 2, (1 - scale) / 2
    return (margin_x, margin_y, 1 - margin_x, 1 - margin_y)


def constrain_fractions(size: tuple[int, int], options: dict) -> tuple[float, float, float, float]:
    """The (x0, y0, x1, y1) fractions of an already-oriented frame of `size` that constrain crop
    keeps -- (0, 0, 1, 1) when it's off or nothing warps the frame."""
    if not options.get("constrain_crop") or not _needs_constraint(options):
        return (0.0, 0.0, 1.0, 1.0)
    aspect_milli = max(1, round(size[0] / size[1] * 1000))
    return _constrain_fractions_cached(
        aspect_milli, float(options.get("lens_distortion", 0.0)), float(options.get("lens_perspective_v", 0.0)),
        float(options.get("lens_perspective_h", 0.0)), float(options.get("straighten", 0.0)),
    )


def _crop_to_fractions(image: Image.Image, fractions: tuple[float, float, float, float]) -> Image.Image:
    if fractions == (0.0, 0.0, 1.0, 1.0):
        return image
    width, height = (image.shape[1], image.shape[0]) if isinstance(image, np.ndarray) else image.size
    x0, y0, x1, y1 = fractions
    box = (int(round(x0 * width)), int(round(y0 * height)), int(round(x1 * width)), int(round(y1 * height)))
    if isinstance(image, np.ndarray):
        return image[box[1]:max(box[3], box[1] + 1), box[0]:max(box[2], box[0] + 1)]
    return image.crop((box[0], box[1], max(box[2], box[0] + 1), max(box[3], box[1] + 1)))


def apply_geometry(image: Image.Image, options: dict) -> Image.Image:
    """Every geometry option in `options` (see GEOMETRY_OPTION_KEYS), in the module's order.
    `image` is a PIL image, or a float32 HxWx3 array (0..1) for the 16-bit path -- the result is
    the same type."""
    image = orient(image, options.get("rotate", 0), options.get("flip_horizontal", False), options.get("flip_vertical", False))
    oriented = (image.shape[1], image.shape[0]) if isinstance(image, np.ndarray) else image.size
    image = apply_lens_corrections(
        image,
        options.get("lens_distortion", 0.0),
        options.get("lens_vignette_amount", 0.0),
        options.get("lens_vignette_midpoint", 50.0),
        options.get("lens_chromatic_aberration", 0.0),
        options.get("lens_perspective_v", 0.0),
        options.get("lens_perspective_h", 0.0),
    )
    if options.get("straighten"):
        # Straighten without constrain crop replicates the edge pixels into the uncovered corners
        # rather than painting them black.
        border = cv2.BORDER_CONSTANT if options.get("constrain_crop") else cv2.BORDER_REPLICATE
        image = straighten(image, options["straighten"], border=border)
    return _crop_to_fractions(image, constrain_fractions(oriented, options))


def geometry_output_size(source_size: tuple[int, int], options: dict) -> tuple[int, int]:
    """The size apply_geometry() produces for a `source_size` image, without running it."""
    width, height = oriented_size(source_size, options.get("rotate", 0))
    x0, y0, x1, y1 = constrain_fractions((width, height), options)
    if (x0, y0, x1, y1) == (0.0, 0.0, 1.0, 1.0):
        return width, height
    return max(1, int(round(x1 * width)) - int(round(x0 * width))), max(1, int(round(y1 * height)) - int(round(y0 * height)))


def estimate_straighten_angle(image: Image.Image) -> float:
    """Auto-level: the straighten angle that makes the photo's dominant near-horizontal/near-
    vertical lines level, from Hough line segments on a small grayscale proxy. 0.0 when no
    confident set of lines is found. Classical (Canny + probabilistic Hough), no ML."""
    proxy = image.convert("L")
    proxy.thumbnail((800, 800), Image.Resampling.LANCZOS)
    gray = np.asarray(proxy)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    min_length = max(20, int(min(gray.shape) * 0.15))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60, minLineLength=min_length, maxLineGap=8)
    if lines is None:
        return 0.0
    deviations, weights = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # (N, 1, 4) on OpenCV 4, (N, 4) on 5
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))  # image y points down
        angle = (angle + 90) % 180 - 90  # -90..90
        if abs(angle) <= 30:  # near horizontal
            deviation = angle
        elif abs(abs(angle) - 90) <= 30:  # near vertical
            deviation = angle - 90 if angle > 0 else angle + 90
        else:
            continue
        deviations.append(deviation)
        weights.append(np.hypot(x2 - x1, y2 - y1))
    if len(deviations) < 2:
        return 0.0
    order = np.argsort(deviations)
    cumulative = np.cumsum(np.asarray(weights)[order])
    median = float(np.asarray(deviations)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
    # A line tilted clockwise (positive deviation, y down) is leveled by rotating counter-clockwise.
    return round(max(-STRAIGHTEN_LIMIT, min(STRAIGHTEN_LIMIT, -median)), 2)
