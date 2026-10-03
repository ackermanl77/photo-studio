"""Optional CUDA resize and output sharpening for final image exports.

Image file encoding remains on the CPU. PyTorch is imported only when a GPU export is requested,
so the CPU build has no additional dependency or startup cost.
"""
from __future__ import annotations

import math

import numpy as np


def resize_and_sharpen(
    pixels: np.ndarray,
    output_size: tuple[int, int],
    sharpening: tuple[float, int, int] | None,
) -> np.ndarray | None:
    """Resize and optionally unsharp-mask HxW, HxWx1, RGB, or RGBA pixels on CUDA.

    Returns None when PyTorch is missing or CUDA is unavailable; CUDA execution failures propagate
    so the caller can log the reason and retry the same work on the CPU.
    """
    from app.gpu_pack import activate_gpu_pack

    activate_gpu_pack()
    try:
        import torch
        import torch.nn.functional as functional
    except (ImportError, OSError):
        return None

    if not torch.cuda.is_available():
        return None
    if pixels.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
        raise ValueError(f"GPU export pixels must be uint8 or float32, got {pixels.dtype}")

    grayscale = pixels.ndim == 2
    source = pixels[..., None] if grayscale else pixels
    if source.ndim != 3 or source.shape[2] not in (1, 3, 4):
        raise ValueError(f"GPU export needs 1, 3 or 4 channels, got shape {pixels.shape}")

    is_eight_bit = source.dtype == np.uint8
    tensor = torch.from_numpy(np.ascontiguousarray(source)).to(device="cuda", dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    if is_eight_bit:
        tensor = tensor / 255.0

    output_width, output_height = output_size
    if (tensor.shape[-1], tensor.shape[-2]) != output_size:
        tensor = functional.interpolate(
            tensor,
            size=(output_height, output_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    if sharpening:
        radius, percent, threshold = sharpening
        support = max(1, math.ceil(radius * 3))
        coordinates = torch.arange(-support, support + 1, device="cuda", dtype=torch.float32)
        kernel = torch.exp(-0.5 * (coordinates / radius).square())
        kernel = kernel / kernel.sum()
        channels = tensor.shape[1]
        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        padding_mode = "reflect" if min(output_width, output_height) > support else "replicate"
        padded = functional.pad(tensor, (support, support, 0, 0), mode=padding_mode)
        blurred = functional.conv2d(padded, horizontal, groups=channels)
        padded = functional.pad(blurred, (0, 0, support, support), mode=padding_mode)
        blurred = functional.conv2d(padded, vertical, groups=channels)
        detail = tensor - blurred
        detail = torch.where(detail.abs() * 255.0 < threshold, 0.0, detail)
        tensor = (tensor + detail * (percent / 100.0)).clamp(0.0, 1.0)

    result = tensor.squeeze(0).permute(1, 2, 0).contiguous()
    if is_eight_bit:
        result = result.mul(255.0).round().clamp(0, 255).to(torch.uint8)
    torch.cuda.synchronize()
    output = result.cpu().numpy()
    return output[..., 0] if grayscale else output
