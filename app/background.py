"""One-step Background (F31): blur, remove or replace with a colour what isn't the subject.

The subject mask comes from the AI masks (F16, app.ai_masks: DeepLabV3 through the GPU pack)
when they are installed, and otherwise from a classical OpenCV GrabCut seeded with a centred
rectangle -- the CPU-first fallback, good on a photo with one subject in the middle and a
distinct background. Blur reuses Effects' Lens Blur (app.creative_filters, linear light, round
aperture); Remove returns the mask, which app.pipeline turns into the export's alpha channel at
the very end; Color fills the background with a flat colour.

Run on the final (cropped) frame, just before the Effects. The mask is cached by the frame's
content, so a rerun that only changes the blur amount or the colour doesn't segment again.
Streamlit-free.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

BACKGROUND_MODES = ("None", "Blur", "Remove", "Color")
SUBJECT_KINDS = ("Subject", "Person")

_PROXY_LONG_EDGE = 640
_CACHE_ENTRIES = 4
_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_cache_lock = threading.Lock()


def _model_available() -> bool:
    from app import ai_masks

    return ai_masks.ai_masks_installed()


def grabcut_subject_mask(image: Image.Image) -> np.ndarray:
    """Classical subject mask, float 0..1 at the image's size: GrabCut on a <=640 px proxy, seeded
    with a rectangle 8 % in from each edge (the frame border is taken as background), softened."""
    rgb = image.convert("RGB")
    proxy = rgb.copy()
    proxy.thumbnail((_PROXY_LONG_EDGE, _PROXY_LONG_EDGE), Image.Resampling.LANCZOS)
    array = cv2.cvtColor(np.asarray(proxy), cv2.COLOR_RGB2BGR)
    height, width = array.shape[:2]
    if min(height, width) < 8:
        return np.ones((rgb.height, rgb.width), dtype=np.float32)
    margin_x, margin_y = max(1, int(width * 0.08)), max(1, int(height * 0.08))
    mask = np.zeros((height, width), np.uint8)
    rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
    background_model, foreground_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(array, mask, rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.ones((rgb.height, rgb.width), dtype=np.float32)
    subject = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    # Keep the largest connected region (GrabCut leaves specks in a busy background).
    count, labels, stats, _ = cv2.connectedComponentsWithStats((subject > 0).astype(np.uint8), 8)
    if count > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        subject = (labels == largest).astype(np.float32)
    subject = cv2.GaussianBlur(subject, (0, 0), max(1.0, 0.004 * max(height, width)))
    return cv2.resize(subject, (rgb.width, rgb.height), interpolation=cv2.INTER_LINEAR)


def subject_mask(image: Image.Image, kind: str = "Subject") -> np.ndarray:
    """The subject (1) / background (0) mask of `image`, float 0..1 at its size; cached by content."""
    if kind not in SUBJECT_KINDS:
        raise ValueError(f"Unknown background subject {kind!r} (choose from {', '.join(SUBJECT_KINDS)})")
    use_model = _model_available()
    digest = hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
    key = f"{digest}|{kind}|{use_model}"
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    if use_model:
        from app.ai_masks import make_ai_mask

        mask = np.asarray(make_ai_mask(image.convert("RGB"), kind).convert("L").resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    else:
        mask = grabcut_subject_mask(image)
    with _cache_lock:
        _cache[key] = mask
        while len(_cache) > _CACHE_ENTRIES:
            _cache.popitem(last=False)
    return mask


def parse_hex_color(value: str) -> tuple[float, float, float]:
    """'#RRGGBB' -> (r, g, b) floats 0..1."""
    text = str(value).lstrip("#")
    return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def apply_background(array: np.ndarray, mask: np.ndarray, mode: str, amount: float = 60.0, color: str = "#FFFFFF") -> np.ndarray:
    """`array` (float RGB 0..1) with its background (where `mask` is 0) blurred or filled with
    `color`. Remove leaves the pixels alone -- the caller applies the mask as alpha."""
    if mode in ("None", "Remove"):
        return array
    weight = (1.0 - np.clip(mask, 0.0, 1.0))[..., None].astype(np.float32)
    if mode == "Blur":
        from app.creative_filters import lens_blur

        replacement = lens_blur(array, max(1.0, amount))
    elif mode == "Color":
        replacement = np.broadcast_to(np.asarray(parse_hex_color(color), dtype=np.float32), array.shape)
    else:
        raise ValueError(f"Unknown background mode {mode!r} (choose from {', '.join(BACKGROUND_MODES)})")
    return array * (1.0 - weight) + replacement * weight
