from __future__ import annotations

"""AI Upscale -- the one learned image-processing exception to this app's CPU-first,
no-ML-dependency design (see "Known Issues" in docs/PROJECT_STATUS.md for why denoise/dehaze/lens
corrections stay classical). Final export resize/sharpen can also opt into a separate, non-ML CUDA
path (app.gpu_export). Real-ESRGAN (via the `realesrgan` PyPI package, itself built on PyTorch) is
an optional dependency -- see requirements-gpu.txt, never installed by
requirements.txt/CI -- so importing this module must never fail or slow down app startup just
because `torch`/`realesrgan` aren't installed. Every import of either lives inside a function body
(a "lazy import"), not at module scope, and `ai_upscale_available()` is the one function the rest
of the app (frontend and pipeline alike) should call before assuming this feature works at all.

Device selection mirrors the sibling PixelFlow project's own
`backend/nodes/io_nodes.py::_resolve_device_and_dtype()`: CUDA + float16 when
`torch.cuda.is_available()`, else CPU + float32 -- same reasoning (float16 halves VRAM/compute on
a GPU that supports it; float32 is the safe, universally-supported fallback for CPU inference,
which Real-ESRGAN still supports, just much slower).
"""

import importlib.util
import logging
import threading
from functools import lru_cache

from PIL import Image

_logger = logging.getLogger(__name__)

# Real-ESRGAN's general-purpose 4x model. RealESRGANer natively only ships 2x/4x/8x-trained
# weights; a 2x *request* is served by running this same 4x model and downscaling the result back
# down to 2x rather than bundling a second model file -- see upscale() below.
_MODEL_NAME = "RealESRGAN_x4plus"
_NATIVE_SCALE = 4


_REQUIRED_PACKAGES = ("torch", "realesrgan", "basicsr")


@lru_cache(maxsize=1)
def ai_upscale_installed() -> bool:
    """Cheap "are the optional packages present at all" probe for UI gating (the Detail tab's AI
    Upscale control) -- importlib.util.find_spec() only locates each package, it never imports it.
    ai_upscale_available() below really imports torch, which costs seconds and hundreds of MB, and
    the Streamlit process never runs inference itself (app/backend.py does), so gating the control
    on that made every Streamlit server process pay for torch just to decide whether to draw a
    checkbox. A broken-but-present install still shows the control here; the backend's own
    ai_upscale_available() check then reports it as a clean error when the upscale is requested.
    Never raises. A packaged build finds the packages in the optional GPU pack (app.gpu_pack),
    which this activates first."""
    from app.gpu_pack import activate_gpu_pack

    activate_gpu_pack()
    try:
        return all(importlib.util.find_spec(name) is not None for name in _REQUIRED_PACKAGES)
    except (ImportError, ValueError):
        return False


def ai_upscale_install_hint() -> str:
    """What the Export panel says when AI Upscale isn't installed: the packaged app takes the
    optional GPU pack (app.gpu_pack); a source checkout installs requirements-gpu.txt."""
    import sys

    from app.gpu_pack import gpu_pack_status

    problem = gpu_pack_status()["problem"]
    if problem:
        return f"AI Upscale is off: {problem}."
    if getattr(sys, "frozen", False):
        return "AI Upscale needs the optional GPU pack: extract it next to PhotoStudio.exe (a gpu-pack folder) and restart."
    return "Install requirements-gpu.txt to enable AI Upscale on export."


def weights_dir():
    """Where Real-ESRGAN's downloaded model weights live: the per-user app data folder
    (app.model_weights). RealESRGANer's own default is a "weights" folder inside the installed
    realesrgan package -- in the packaged desktop build that's under _internal/, i.e. inside
    Program Files, which a normal user can't write to, so the first AI Upscale failed there."""
    from app.model_weights import weights_dir as shared_weights_dir

    return shared_weights_dir()


@lru_cache(maxsize=1)
def ai_upscale_available() -> bool:
    """True only if `torch` and `realesrgan` are both actually importable in this environment.
    Cached (an import probe is not free) -- safe to cache since installed packages don't change
    within a single process's lifetime. Never raises: any import failure (missing package, a
    partially-broken install, an incompatible torch/CUDA combination that errors on import) is
    treated the same as "not installed," which is what lets the frontend and the pipeline both
    treat AI Upscale as a plain optional feature rather than something that can crash the rest of
    the app."""
    from app.gpu_pack import activate_gpu_pack

    activate_gpu_pack()
    try:
        import torch  # noqa: F401
        import realesrgan  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name in ("torch", "realesrgan"):
            return False  # simply not installed -- the normal CPU-only case, nothing to report
        _logger.warning("AI Upscale disabled: an optional dependency failed to import", exc_info=True)
        return False
    except Exception:
        # Installed but broken (e.g. a packaged build missing a data file, or a torch/basicsr
        # incompatibility) -- still degrade to "unavailable", but say why, since otherwise the
        # feature just silently disappears with no trace of the cause.
        _logger.warning("AI Upscale disabled: an optional dependency failed to import", exc_info=True)
        return False
    return True


def _resolve_device_and_dtype():
    """Mirrors PixelFlow's backend/nodes/io_nodes.py::_resolve_device_and_dtype(): CUDA+float16
    when available, CPU+float32 otherwise. Imported lazily by callers that already know torch is
    installed (ai_upscale_available() returned True), so this itself does not import torch at
    module scope either."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


_WEIGHTS_URL = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{_MODEL_NAME}.pth"
# SHA-256 of the official RealESRGAN_x4plus.pth release asset (67,040,989 bytes). Checked after
# every download and before every load, so a truncated or substituted file is never fed to torch.
WEIGHTS_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
WEIGHTS_SIZE_BYTES = 67_040_989

# Tiled inference: RRDBNet's activation memory grows with the input's pixel count, so a whole-frame
# pass (tile=0) on a large photo risked CUDA out-of-memory on a 16 GB card. 512 px tiles with a
# 32 px overlap are seam-free in practice; _tile_size() halves the tile when free VRAM is low.
_TILE = 512
_SMALL_TILE = 256
_TILE_PAD = 32
_LOW_VRAM_BYTES = 3 * 1024**3

# One inference at a time: the model object is shared by every request thread of the backend, and
# two concurrent passes would also compete for the same VRAM.
_INFERENCE_LOCK = threading.Lock()


def weights_path():
    return weights_dir() / f"{_MODEL_NAME}.pth"


def weights_present() -> bool:
    """True when the model file is already on disk at its expected size (no network needed to
    upscale). Cheap -- the full hash is checked when the model actually loads."""
    path = weights_path()
    return path.is_file() and path.stat().st_size == WEIGHTS_SIZE_BYTES


def download_weights(progress=None, url: str | None = None, opener=None):
    """Download the Real-ESRGAN weights into weights_dir(), verifying their SHA-256 (app.model_weights),
    and return the path. `progress(done_bytes, total_bytes)` is called as data arrives (the Export
    panel's "Download model" button drives an st.progress bar with it)."""
    from app.model_weights import download_verified

    return download_verified(url or _WEIGHTS_URL, weights_path(), WEIGHTS_SHA256, WEIGHTS_SIZE_BYTES, progress=progress, opener=opener,
                             failure="The downloaded AI Upscale model failed its checksum -- try again (the file was discarded).")


def _verified_weights():
    from app.model_weights import verified_weights

    return verified_weights(_WEIGHTS_URL, weights_path(), WEIGHTS_SHA256, WEIGHTS_SIZE_BYTES)


def _tile_size(device: str) -> int:
    if device != "cuda":
        return _TILE
    try:
        import torch

        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 -- an odd driver must not block upscaling; keep the default
        return _TILE
    return _SMALL_TILE if free_bytes < _LOW_VRAM_BYTES else _TILE


@lru_cache(maxsize=1)
def _load_upsampler():
    """Builds (and caches, module-process-lifetime) the RealESRGANer wrapper around the
    RealESRGAN_x4plus architecture. Only ever called once real work is requested -- neither
    importing this module nor calling ai_upscale_available() triggers a model load or a weights
    download."""
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    device, dtype = _resolve_device_and_dtype()
    half = dtype == torch.float16

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=_NATIVE_SCALE)
    # A verified local file (see _verified_weights) handed to RealESRGANer as a path -- given a
    # URL, RealESRGANer would download into its own package folder, unverified.
    return RealESRGANer(
        scale=_NATIVE_SCALE,
        model_path=str(_verified_weights()),
        model=model,
        tile=_tile_size(device),
        tile_pad=_TILE_PAD,
        pre_pad=0,
        half=half,
        device=device,
    )


def upscale(image: Image.Image, factor: int = 2) -> Image.Image:
    """Upscales `image` by `factor` (2 or 4) using Real-ESRGAN, auto-selecting CUDA when present
    (falls back to CPU inference otherwise -- slower but still functional, same graceful-
    degradation contract as every other optional dependency in this app). Raises RuntimeError if
    the optional deps aren't installed -- callers (app.image_processor.PhotoProcessor.apply_ai_upscale
    / app.backend.process_image) are expected to check ai_upscale_available() first (the frontend
    only shows the control when it's True) so this is a defensive check, not the primary gate.

    `factor` must be 2 or 4. RealESRGANer's `enhance(outscale=...)` handles both directly (it runs
    the native 4x model, then itself resizes to whatever `outscale` was requested), so 2x doesn't
    need a second model -- it's the same architecture, just resized down further on the way out.
    Inference is serialized (_INFERENCE_LOCK), runs under torch.inference_mode(), and frees the
    CUDA cache afterwards."""
    if not ai_upscale_available():
        raise RuntimeError(
            "AI Upscale requires the optional 'torch' and 'realesrgan' packages -- "
            "install requirements-gpu.txt to enable this feature."
        )
    if factor not in (2, 4):
        raise ValueError("AI Upscale factor must be 2 or 4")

    import numpy as np

    with _INFERENCE_LOCK:
        upsampler = _load_upsampler()
        rgb = np.array(image.convert("RGB"))
        # RealESRGANer works in BGR (it's built around cv2 conventions internally).
        bgr = rgb[:, :, ::-1]
        with _inference_mode():
            output, _ = upsampler.enhance(bgr, outscale=factor)
        _empty_cuda_cache()
    result_rgb = output[:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(result_rgb), mode="RGB")


def _inference_mode():
    try:
        import torch

        return torch.inference_mode()
    except ImportError:
        import contextlib

        return contextlib.nullcontext()


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- best effort only
        pass
