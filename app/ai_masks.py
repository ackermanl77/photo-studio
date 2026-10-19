"""AI masks (F16): Subject, Person and Sky selections, added to Masks/Repair as a Painted selection.

- **Subject / Person** -- optional, through the GPU pack's torch: torchvision's DeepLabV3 with a
  MobileNetV3-Large backbone (Pascal VOC's 20 object classes). Person is its "person" class;
  Subject is every object class (people, animals, vehicles, furniture...) against the background.
  The weights (~42 MB) are downloaded by torchvision into its own cache the first time a mask is
  made -- the UI says so before the first one -- and the model runs on the GPU when CUDA is there.
  Inference happens in the backend process (``POST /ai_mask``), like AI Upscale: the Streamlit
  process never imports torch.
- **Sky** -- classical, no model: bright, smooth, blue-or-white pixels connected to the top edge
  of the frame (sky_mask), softened at the edges. Works without the GPU pack.

The result is a grayscale mask at the photo's size; ``mask_to_painted_b64`` turns it into the
paint editor's format (white at the mask's alpha), so it becomes an ordinary editable Painted
selection -- invert it, combine it, soften it, use it for local adjustments, Repair or a layer mask.
"""
from __future__ import annotations

import base64
import io
import threading
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

AI_MASK_KINDS = ("Subject", "Person", "Sky")
MODEL_KINDS = ("Subject", "Person")
WEIGHTS_FILE = "deeplabv3_mobilenet_v3_large-fc3c493d.pth"
WEIGHTS_DOWNLOAD = "42 MB"
PERSON_CLASS = 15  # Pascal VOC
MODEL_SHORT_EDGE = 520
PAINTED_LONG_EDGE = 1024
_MODEL_LOCK = threading.Lock()


# --- Sky (classical) -------------------------------------------------------------------------------

def sky_mask(image: Image.Image) -> Image.Image:
    """Bright, low-texture, bluish or near-white pixels connected to the top edge (L, image size)."""
    work = image.convert("RGB")
    work.thumbnail((640, 640))
    rgb = np.asarray(work, np.float32) / 255.0
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat, val = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    grey = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    texture = cv2.blur(np.abs(cv2.Laplacian(grey, cv2.CV_32F)), (9, 9))
    blue = (hue >= 170) & (hue <= 260) & (sat > 0.12) & (val > 0.35)
    white = (sat < 0.18) & (val > 0.72)
    sunset = ((hue < 50) | (hue > 300)) & (val > 0.55) & (sat < 0.75)
    candidate = (blue | white | sunset) & (texture < 0.035)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents(candidate)
    top_labels = set(np.unique(labels[: max(1, labels.shape[0] // 40)]).tolist()) - {0}
    mask = np.isin(labels, list(top_labels)).astype(np.float32)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(mask.shape) / 200.0))
    return Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L").resize(image.size, Image.Resampling.BILINEAR)


# --- Subject / Person (DeepLabV3) ------------------------------------------------------------------

def ai_masks_installed() -> bool:
    """torch + torchvision present (the GPU pack, or this environment). Cheap: no import."""
    import importlib.util

    from app.gpu_pack import activate_gpu_pack

    activate_gpu_pack()
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


def weights_cached() -> bool:
    """Whether the model's weights are already in torch's hub cache (no download needed). Resolves
    the cache folder the way torch.hub.get_dir() does, without importing torch."""
    import os

    home = os.environ.get("TORCH_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "torch")
    return os.path.isfile(os.path.join(home, "hub", "checkpoints", WEIGHTS_FILE))


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large

    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), device


def class_probabilities(image: Image.Image) -> np.ndarray:
    """(21, h, w) softmax over Pascal VOC classes at a MODEL_SHORT_EDGE-px copy of `image`."""
    import torch

    with _MODEL_LOCK:
        model, device = _load_model()
        work = image.convert("RGB")
        scale = MODEL_SHORT_EDGE / min(work.size)
        if scale < 1:
            work = work.resize((round(work.width * scale), round(work.height * scale)), Image.Resampling.BILINEAR)
        array = (np.asarray(work, np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(tensor)["out"][0]
        return torch.softmax(logits, dim=0).float().cpu().numpy()


def model_mask(image: Image.Image, kind: str, probabilities=class_probabilities) -> Image.Image:
    """Subject (1 - background) or Person probability as an L mask at the image's size."""
    probs = probabilities(image)
    values = probs[PERSON_CLASS] if kind == "Person" else 1.0 - probs[0]
    values = cv2.GaussianBlur(values.astype(np.float32), (0, 0), 1.0)
    small = Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8), mode="L")
    return small.resize(image.size, Image.Resampling.BILINEAR)


def make_ai_mask(image: Image.Image, kind: str, probabilities=class_probabilities) -> Image.Image:
    if kind not in AI_MASK_KINDS:
        raise ValueError(f"Unknown AI mask {kind!r} (choose from {', '.join(AI_MASK_KINDS)})")
    if kind == "Sky":
        return sky_mask(image)
    return model_mask(image, kind, probabilities)


def mask_to_painted_b64(mask: Image.Image, long_edge: int = PAINTED_LONG_EDGE) -> str:
    """The paint editor's format: white RGBA whose alpha is the mask (app.ui.selections.decode_painted_mask)."""
    mask = mask.convert("L")
    if max(mask.size) > long_edge:
        scale = long_edge / max(mask.size)
        mask = mask.resize((max(1, round(mask.width * scale)), max(1, round(mask.height * scale))), Image.Resampling.BILINEAR)
    painted = Image.new("RGBA", mask.size, (255, 255, 255, 0))
    painted.putalpha(mask)
    buffer = io.BytesIO()
    painted.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
