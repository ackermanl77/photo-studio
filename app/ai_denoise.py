"""AI Denoise -- the third optional learned step (after AI Upscale and the AI masks), through the
GPU pack's torch. Only torch is needed: the network is defined here, so the GPU pack gains no
package.

The model is SCUNet (Zhang et al., "Practical Blind Image Denoising via Swin-Conv-UNet and Data
Synthesis", 2023; Apache-2.0), its "real" PSNR weights from the KAIR release (~72 MB, SHA-256
checked, downloaded into app.model_weights.weights_dir() the first time it runs). It is a blind
denoiser trained on synthesized camera noise (shot/read noise through a camera pipeline, JPEG), so
it needs no noise level, which suits real photos better than a model trained on Gaussian noise.

How the pipeline uses it (app.pipeline.run_pipeline, right after the ICC conversion and before
Geometry, so it sees the noise as shot): the denoised pixels are cached by their input
(_RESULTS), so every later edit and every change of the amount reuses one inference per photo
and preview size; the amount blends the result with the original. Without torch the pipeline
falls back to the classical Non-Local Means denoise (FALLBACK_DENOISE), so a recipe that asks for
AI Denoise still renders anywhere; the provenance mark (F34) is written only when the model ran.

Like the other optional modules, nothing here imports torch at module scope.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import threading
from collections import OrderedDict
from functools import lru_cache

import numpy as np

_logger = logging.getLogger(__name__)

MODEL_NAME = "scunet_color_real_psnr"
WEIGHTS_URL = f"https://github.com/cszn/KAIR/releases/download/v1.0/{MODEL_NAME}.pth"
WEIGHTS_SHA256 = "fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e"
WEIGHTS_SIZE_BYTES = 71_982_841
WEIGHTS_DOWNLOAD = "72 MB"
# The classical Denoise strength (PhotoProcessor.apply_denoise) used when the model can't run.
FALLBACK_DENOISE = 5

# Tiled inference: SCUNet's memory grows with the pixel count. Each tile is inferred with
# _TILE_PAD px of real context on every side and only its core is kept, so tiles meet seamlessly.
_TILE = 512
_TILE_PAD = 32
_INFERENCE_LOCK = threading.Lock()
# Denoised pixels by input (sha1 of the array): one photo at its preview and full-resolution sizes.
_RESULTS: OrderedDict[str, np.ndarray] = OrderedDict()
_RESULTS_MAX = 3


def ai_denoise_installed() -> bool:
    """torch present (the GPU pack, or this environment). Cheap: no import."""
    from app.gpu_pack import activate_gpu_pack

    activate_gpu_pack()
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def ai_denoise_available() -> bool:
    """torch really imports (a broken install counts as unavailable, with the reason logged)."""
    if not ai_denoise_installed():
        return False
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 -- any import failure means "unavailable"
        _logger.warning("AI Denoise disabled: torch failed to import", exc_info=True)
        return False
    return True


def weights_path():
    from app.model_weights import weights_dir

    return weights_dir() / f"{MODEL_NAME}.pth"


def weights_present() -> bool:
    path = weights_path()
    return path.is_file() and path.stat().st_size == WEIGHTS_SIZE_BYTES


# --- SCUNet ----------------------------------------------------------------------------------------
# The architecture of KAIR's models/network_scunet.py (config [4]*7, dim 64, 8 px windows, 32-dim
# heads), with its einops/timm helpers written out in torch so the parameter names match the
# released weights. Inference only: no drop path.

def _build_model():
    import torch
    from torch import nn

    class WMSA(nn.Module):
        """(Shifted) window multi-head self-attention on a (b, h, w, c) tensor."""

        def __init__(self, dim, head_dim, window, shifted):
            super().__init__()
            self.heads, self.head_dim, self.window, self.shifted = dim // head_dim, head_dim, window, shifted
            self.scale = head_dim ** -0.5
            self.embedding_layer = nn.Linear(dim, 3 * dim, bias=True)
            self.relative_position_params = nn.Parameter(torch.zeros(self.heads, 2 * window - 1, 2 * window - 1))
            self.linear = nn.Linear(dim, dim)
            coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"), -1).reshape(-1, 2)
            relation = coords[:, None, :] - coords[None, :, :] + window - 1
            self.register_buffer("relation", relation, persistent=False)

        def _mask(self, h_windows, w_windows, device):
            p, s = self.window, self.window - self.window // 2
            mask = torch.zeros(h_windows, w_windows, p, p, p, p, dtype=torch.bool, device=device)
            mask[-1, :, :s, :, s:, :] = True
            mask[-1, :, s:, :, :s, :] = True
            mask[:, -1, :, :s, :, s:] = True
            mask[:, -1, :, s:, :, :s] = True
            return mask.reshape(1, 1, h_windows * w_windows, p * p, p * p)

        def forward(self, x):
            p = self.window
            if self.shifted:
                x = torch.roll(x, shifts=(-(p // 2), -(p // 2)), dims=(1, 2))
            b, h, w, c = x.shape
            hw, ww = h // p, w // p
            x = x.reshape(b, hw, p, ww, p, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hw * ww, p * p, c)
            qkv = self.embedding_layer(x).reshape(b, hw * ww, p * p, 3 * self.heads, self.head_dim).permute(3, 0, 1, 2, 4)
            q, k, v = qkv.chunk(3, dim=0)  # each (heads, b, windows, p*p, head_dim)
            sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
            bias = self.relative_position_params[:, self.relation[..., 0], self.relation[..., 1]]
            sim = sim + bias[:, None, None]
            if self.shifted:
                sim = sim.masked_fill(self._mask(hw, ww, x.device), float("-inf"))
            out = torch.einsum("hbwij,hbwjc->hbwic", torch.softmax(sim, dim=-1), v)
            out = out.permute(1, 2, 3, 0, 4).reshape(b, hw * ww, p * p, c)
            out = self.linear(out).reshape(b, hw, ww, p, p, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)
            if self.shifted:
                out = torch.roll(out, shifts=(p // 2, p // 2), dims=(1, 2))
            return out

    class Block(nn.Module):
        def __init__(self, dim, head_dim, window, shifted):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.msa = WMSA(dim, head_dim, window, shifted)
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

        def forward(self, x):
            x = x + self.msa(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class ConvTransBlock(nn.Module):
        def __init__(self, conv_dim, trans_dim, shifted):
            super().__init__()
            self.conv_dim, self.trans_dim = conv_dim, trans_dim
            self.trans_block = Block(trans_dim, 32, 8, shifted)
            self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
            self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
            self.conv_block = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False), nn.ReLU(True), nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False))

        def forward(self, x):
            conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
            conv_x = self.conv_block(conv_x) + conv_x
            trans_x = self.trans_block(trans_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))

    class SCUNet(nn.Module):
        def __init__(self, dim=64, blocks=4):
            super().__init__()

            def stage(width):
                return [ConvTransBlock(width, width, bool(i % 2)) for i in range(blocks)]

            self.m_head = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1, bias=False))
            self.m_down1 = nn.Sequential(*stage(dim // 2), nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False))
            self.m_down2 = nn.Sequential(*stage(dim), nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False))
            self.m_down3 = nn.Sequential(*stage(2 * dim), nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False))
            self.m_body = nn.Sequential(*stage(4 * dim))
            self.m_up3 = nn.Sequential(nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), *stage(2 * dim))
            self.m_up2 = nn.Sequential(nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), *stage(dim))
            self.m_up1 = nn.Sequential(nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), *stage(dim // 2))
            self.m_tail = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1, bias=False))

        def forward(self, x0):
            h, w = x0.shape[-2:]
            x0 = nn.functional.pad(x0, (0, -w % 64, 0, -h % 64), mode="replicate")
            x1 = self.m_head(x0)
            x2 = self.m_down1(x1)
            x3 = self.m_down2(x2)
            x4 = self.m_down3(x3)
            x = self.m_body(x4)
            x = self.m_up3(x + x4)
            x = self.m_up2(x + x3)
            x = self.m_up1(x + x2)
            return self.m_tail(x + x1)[..., :h, :w]

    return SCUNet()


@lru_cache(maxsize=1)
def _load_model():
    import torch

    from app.model_weights import verified_weights

    path = verified_weights(WEIGHTS_URL, weights_path(), WEIGHTS_SHA256, WEIGHTS_SIZE_BYTES)
    model = _build_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.eval().to(device), device


def _windows(length: int) -> list[tuple[int, int, int, int]]:
    """Tiles along one axis as (start, end, keep_from, keep_to): every tile is _TILE long (the last
    one shifted inward rather than cut short -- a thin tile was denoised visibly worse), and
    neighbours split their overlap at its middle."""
    if length <= _TILE:
        return [(0, length, 0, length)]
    starts = list(range(0, length - _TILE, _TILE - 2 * _TILE_PAD)) + [length - _TILE]
    windows = []
    for index, start in enumerate(starts):
        keep_from = 0 if index == 0 else windows[-1][3]
        keep_to = length if index == len(starts) - 1 else (start + _TILE + starts[index + 1]) // 2
        windows.append((start, start + _TILE, keep_from, keep_to))
    return windows


def _infer(rgb: np.ndarray, between_tiles=None) -> np.ndarray:
    """SCUNet over `rgb` (float32 h x w x 3, 0..1) in overlapping tiles; float32 0..1 out. The photo
    is mirrored outward by _TILE_PAD first, so its border pixels have noisy context like the rest
    (the network's own padding repeats the edge, which left a noisy rim)."""
    import torch

    model, device = _load_model()
    height, width = rgb.shape[:2]
    pad = min(_TILE_PAD, height - 1, width - 1)
    padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(padded)
    with torch.inference_mode():
        for y0, y1, keep_top, keep_bottom in _windows(padded.shape[0]):
            for x0, x1, keep_left, keep_right in _windows(padded.shape[1]):
                if between_tiles is not None:
                    between_tiles()
                tile = torch.from_numpy(np.ascontiguousarray(padded[y0:y1, x0:x1].transpose(2, 0, 1))).unsqueeze(0).to(device)
                result = model(tile)[0].clamp_(0, 1).permute(1, 2, 0).float().cpu().numpy()
                out[keep_top:keep_bottom, keep_left:keep_right] = result[keep_top - y0:keep_bottom - y0, keep_left - x0:keep_right - x0]
    if device == "cuda":
        torch.cuda.empty_cache()
    return out[pad:pad + height, pad:pad + width]


def denoise(rgb: np.ndarray, between_tiles=None) -> np.ndarray:
    """The denoised copy of `rgb` (float32 h x w x 3, 0..1), from the cache when this exact input was
    denoised before. `between_tiles` is called before each tile (the pipeline's cancel check)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    key = hashlib.sha1(rgb.tobytes()).hexdigest() + str(rgb.shape)
    with _INFERENCE_LOCK:
        if key in _RESULTS:
            _RESULTS.move_to_end(key)
            return _RESULTS[key]
        result = _infer(rgb, between_tiles)
        _RESULTS[key] = result
        while len(_RESULTS) > _RESULTS_MAX:
            _RESULTS.popitem(last=False)
        return result


def blend(original: np.ndarray, denoised: np.ndarray, amount: float) -> np.ndarray:
    """`amount` 0-100: how much of the denoised result to use (100 = all of it)."""
    weight = float(np.clip(amount, 0, 100)) / 100.0
    return original + (denoised - original) * weight
