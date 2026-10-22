"""Relight (F32): light the photo from a new direction, CPU-only and classical.

Photos' Relight estimates a normal map with a neural model. This one uses the classical
shape-from-shading shortcut instead: brighter is nearer -- a heavily smoothed luminance becomes a
height field, its gradients give a surface normal per pixel, and a Lambertian term (N . L) for one
movable directional light, plus an ambient level, scales the photo's own light. It is a creative
approximation (dark objects read as hollows), not a 3D reconstruction; the GPU-pack route (depth
from a model) can replace estimate_normals() later without changing the rest.

Works on float RGB 0..1 in linear light, so a brighter side looks like more light rather than a
contrast boost. Spatial sizes are fractions of the image, so a preview proxy and the export match.
Streamlit-free.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from app.creative_filters import linear_to_srgb, srgb_to_linear

# The height field's smoothing (fraction of the long edge) and relief (how strongly its slopes tilt the normals).
_HEIGHT_SIGMA_FRACTION = 0.02
_RELIEF = 6.0
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def estimate_normals(array: np.ndarray) -> np.ndarray:
    """Unit surface normals (H x W x 3, z towards the viewer) from `array`'s smoothed luminance."""
    height, width = array.shape[:2]
    luminance = (array[..., :3] @ _LUMA).astype(np.float32)
    sigma = max(1.0, _HEIGHT_SIGMA_FRACTION * max(height, width))
    surface = cv2.GaussianBlur(luminance, (0, 0), sigma)
    # Slopes per fraction of the frame (not per pixel), so the result doesn't depend on resolution.
    gx = cv2.Sobel(surface, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * max(height, width)
    gy = cv2.Sobel(surface, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * max(height, width)
    scale = _RELIEF / max(1.0, float(np.percentile(np.hypot(gx, gy), 99)))
    normals = np.dstack((-gx * scale, -gy * scale, np.ones_like(surface)))
    return normals / np.linalg.norm(normals, axis=2, keepdims=True)


def light_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """The direction towards the light: azimuth 0 = from the right, 90 = from the top; elevation 0
    = grazing from the side, 90 = straight from the viewer."""
    azimuth, elevation = math.radians(azimuth_deg), math.radians(max(0.0, min(90.0, elevation_deg)))
    return np.array([math.cos(elevation) * math.cos(azimuth), -math.cos(elevation) * math.sin(azimuth), math.sin(elevation)], dtype=np.float32)


def relight(array: np.ndarray, amount: float, azimuth_deg: float = 135.0, elevation_deg: float = 35.0,
            ambient: float = 40.0, warmth: float = 0.0) -> np.ndarray:
    """`array` (float RGB 0..1) relit by one light at `amount` percent. `ambient` (0-100) is how
    much light reaches the side facing away; `warmth` (-100..100) tints the new light cooler or
    warmer. 0 amount returns the input."""
    strength = max(0.0, min(100.0, float(amount))) / 100.0
    if strength <= 0.0:
        return array
    normals = estimate_normals(array)
    lambert = np.clip(normals @ light_vector(azimuth_deg, elevation_deg), 0.0, 1.0)
    # Normalized so a surface facing the viewer keeps its brightness: the flat parts of the photo
    # stay put, the slopes facing the light brighten and the ones facing away darken.
    flat = float(light_vector(azimuth_deg, elevation_deg)[2]) or 1e-3
    ambient_level = max(0.0, min(100.0, float(ambient))) / 100.0
    gain = ambient_level + (1.0 - ambient_level) * lambert / max(flat, 0.2)
    gain = np.clip(gain, 0.0, 2.5)[..., None]
    tint = 1.0 + np.array([0.15, 0.03, -0.15], dtype=np.float32) * (max(-100.0, min(100.0, float(warmth))) / 100.0)
    linear = srgb_to_linear(array[..., :3])
    lit = linear * (1.0 + (gain * tint - 1.0) * strength)
    return np.clip(linear_to_srgb(np.clip(lit, 0.0, None)), 0.0, 1.0).astype(np.float32)
