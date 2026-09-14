from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not hasattr(cv2, "CascadeClassifier"):
        return []
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return []
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return [tuple(int(value) for value in face) for face in faces]


def skin_mask(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv_mask = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([35, 220, 255]))
    ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 180, 135]))
    combined = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    combined = cv2.medianBlur(combined, 5)
    return Image.fromarray(combined, mode="L")


def smooth_skin(image: Image.Image, strength: float = 0.25) -> Image.Image:
    strength = max(0.0, min(1.0, float(strength)))
    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    mask = np.asarray(skin_mask(image), dtype=np.float32) / 255.0
    blend = mask[..., None] * strength
    return Image.fromarray(np.clip(source * (1.0 - blend) + blurred * blend, 0, 255).astype(np.uint8), mode="RGB")
