from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

# YuNet (OpenCV Zoo, MIT license -- see face_detection_yunet_LICENSE.txt next to the model), a
# ~230 KB CNN face detector run through cv2.FaceDetectorYN (OpenCV's own DNN module, CPU-only, no
# extra dependency). The primary detector because the Haar cascade this app used to rely on
# exclusively is gone from OpenCV 5 (no cv2.CascadeClassifier) -- the version a fresh
# `pip install -r requirements.txt` resolves to -- which made detect_faces() silently return []
# and the Portrait tab always report "Detected faces: 0". FaceDetectorYN exists in both OpenCV
# 4.5.4+ and 5.x; the Haar cascade stays only as a fallback for an OpenCV build without it.
YUNET_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "face_detection_yunet_2023mar.onnx"
_YUNET_SCORE_THRESHOLD = 0.8
_YUNET_NMS_THRESHOLD = 0.3
# Detection runs on a proxy no larger than this on its long edge (boxes are scaled back to the
# original's coordinates): YuNet finds faces down to ~10 px, so a 1024 px proxy still catches any
# face big enough to retouch, at a fraction of the full-resolution cost.
DETECTION_LONG_EDGE = 1024
# FaceDetectorYN keeps per-call input-size state (setInputSize), so the one cached instance is
# guarded against concurrent use from the backend's/Streamlit's worker threads.
_YUNET_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _yunet_detector():
    if not hasattr(cv2, "FaceDetectorYN") or not YUNET_MODEL_PATH.exists():
        return None
    try:
        return cv2.FaceDetectorYN.create(str(YUNET_MODEL_PATH), "", (320, 320), _YUNET_SCORE_THRESHOLD, _YUNET_NMS_THRESHOLD, 5000)
    except cv2.error:
        return None


def _haar_cascade_path() -> Path | None:
    data = getattr(cv2, "data", None)
    if data is None or not hasattr(cv2, "CascadeClassifier"):
        return None
    path = Path(data.haarcascades) / "haarcascade_frontalface_default.xml"
    return path if path.exists() else None


def face_detection_backend() -> str:
    """Which detector detect_faces() will actually use: "yunet", "haar", or "none". Reported by
    app.diagnostics.check_environment() and the desktop --self-test, so a build or environment
    where face detection has silently stopped working shows up as a failed check instead of an
    always-zero face count."""
    if _yunet_detector() is not None:
        return "yunet"
    if _haar_cascade_path() is not None:
        return "haar"
    return "none"


def _proxy(image: Image.Image) -> tuple[Image.Image, float]:
    rgb = image.convert("RGB")
    long_edge = max(rgb.size)
    if long_edge <= DETECTION_LONG_EDGE:
        return rgb, 1.0
    scale = DETECTION_LONG_EDGE / long_edge
    size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    return rgb.resize(size, Image.Resampling.BILINEAR), scale


def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Face boxes as (x, y, width, height) in `image`'s own pixel coordinates. Runs on a
    DETECTION_LONG_EDGE proxy (see above); YuNet first, the Haar cascade only when YuNet isn't
    available. Returns [] when no detector is available -- check face_detection_backend() to tell
    "no faces" apart from "can't detect"."""
    proxy, scale = _proxy(image)
    bgr = np.ascontiguousarray(np.asarray(proxy)[:, :, ::-1])
    detector = _yunet_detector()
    if detector is not None:
        with _YUNET_LOCK:
            detector.setInputSize((proxy.width, proxy.height))
            _, faces = detector.detect(bgr)
        raw_boxes = [] if faces is None else [face[:4] for face in faces]
    else:
        cascade_path = _haar_cascade_path()
        if cascade_path is None:
            return []
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        raw_boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    boxes = []
    for x, y, width, height in raw_boxes:
        left, top = max(0, int(round(x / scale))), max(0, int(round(y / scale)))
        boxes.append((left, top, int(round(width / scale)), int(round(height / scale))))
    return boxes


def skin_mask(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv_mask = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([35, 220, 255]))
    ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 180, 135]))
    combined = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    combined = cv2.medianBlur(combined, 5)
    return Image.fromarray(combined, mode="L")


def smooth_skin(image: Image.Image, strength: float = 0.25, blur_radius: float = 2.0) -> Image.Image:
    """`blur_radius` is in `image`'s own pixels -- app.pipeline scales it down for a reduced-size
    interactive preview so the preview softens skin as much as the full-resolution export does."""
    strength = max(0.0, min(1.0, float(strength)))
    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(max(0.0, float(blur_radius)))), dtype=np.float32)
    mask = np.asarray(skin_mask(image), dtype=np.float32) / 255.0
    blend = mask[..., None] * strength
    return Image.fromarray(np.clip(source * (1.0 - blend) + blurred * blend, 0, 255).astype(np.uint8), mode="RGB")
