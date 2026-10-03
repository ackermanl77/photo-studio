"""The optional GPU pack (K5): AI Upscale's torch/Real-ESRGAN stack shipped apart from the app.

The release is a CPU base package (a few hundred MB) plus an optional "gpu-pack" folder built by
scripts/build_release.ps1 -GpuPack. The pack holds only the extra packages (torch, torchvision,
basicsr, realesrgan and what they need beyond the base) in `gpu-pack/site-packages`, and a
`gpu-pack.json` manifest naming the Python version they were built for. Dropping the folder next
to PhotoStudio.exe (or installing the installer's "GPU pack" component) enables AI Upscale; nothing
else changes.

activate_gpu_pack() is called lazily by app.ai_upscale's probes -- importing this module does
nothing -- and appends the pack's site-packages to sys.path (after the base, so the base's own
numpy/Pillow/OpenCV always win). Where it looks, in order:

1. $PHOTO_STUDIO_GPU_PACK (a pack folder; used by the build's self-test and for development);
2. `gpu-pack` next to the executable (the frozen app's own folder);
3. `gpu-pack` in the per-user data folder (app.config.user_data_dir()).

A source checkout with requirements-gpu.txt installed in its venv needs no pack at all.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from pathlib import Path

ENV_VAR = "PHOTO_STUDIO_GPU_PACK"
PACK_DIR_NAME = "gpu-pack"
MANIFEST_NAME = "gpu-pack.json"
PACK_FORMAT = 1

_lock = threading.Lock()
_state: dict = {"activated": None, "checked": False, "problem": None}


def python_tag() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def candidate_dirs() -> list[Path]:
    candidates = []
    if os.environ.get(ENV_VAR):
        candidates.append(Path(os.environ[ENV_VAR]))
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / PACK_DIR_NAME)
    try:
        from app.config import user_data_dir

        candidates.append(user_data_dir() / PACK_DIR_NAME)
    except Exception:  # noqa: BLE001 -- a missing data folder just means one fewer place to look
        pass
    return candidates


def read_manifest(pack_dir: Path) -> dict | None:
    try:
        manifest = json.loads((pack_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def check_pack(pack_dir: Path) -> str | None:
    """None when `pack_dir` is a usable pack for this interpreter, else why not."""
    manifest = read_manifest(pack_dir)
    if manifest is None:
        return f"{pack_dir} has no readable {MANIFEST_NAME}"
    if manifest.get("format") != PACK_FORMAT:
        return f"{pack_dir} is GPU pack format {manifest.get('format')!r}; this app reads format {PACK_FORMAT}"
    if manifest.get("python") != python_tag():
        return f"{pack_dir} was built for Python {manifest.get('python')}; this app runs Python {python_tag()} -- get the GPU pack made for this version"
    if not (pack_dir / "site-packages").is_dir():
        return f"{pack_dir} has no site-packages folder"
    return None


def find_gpu_pack() -> tuple[Path | None, str | None]:
    """(the first usable pack folder, None) or (None, why the first pack found can't be used)."""
    problem = None
    for candidate in candidate_dirs():
        if not (candidate / MANIFEST_NAME).exists():
            continue
        reason = check_pack(candidate)
        if reason is None:
            return candidate, None
        problem = problem or reason
    return None, problem


def activate_gpu_pack() -> Path | None:
    """Put a found pack's packages on sys.path, once per process. Returns the pack folder, or None
    when there is no usable pack. Never raises."""
    with _lock:
        if _state["checked"]:
            return _state["activated"]
        _state["checked"] = True
        try:
            pack, problem = find_gpu_pack()
        except Exception as error:  # noqa: BLE001
            pack, problem = None, str(error)
        _state["problem"] = problem
        if pack is None:
            return None
        site_packages = str(pack / "site-packages")
        if site_packages not in sys.path:
            sys.path.append(site_packages)
        torch_lib = pack / "site-packages" / "torch" / "lib"
        if os.name == "nt" and torch_lib.is_dir():
            try:
                os.add_dll_directory(str(torch_lib))
            except OSError:
                pass
        importlib.invalidate_caches()
        _state["activated"] = pack
        return pack


def gpu_pack_status() -> dict:
    """For System Health and the build's self-test."""
    pack = activate_gpu_pack()
    manifest = read_manifest(pack) if pack else None
    return {
        "active": pack is not None,
        "path": str(pack) if pack else None,
        "torch": (manifest or {}).get("torch"),
        "problem": _state["problem"],
    }


def _reset_for_tests() -> None:
    with _lock:
        _state.update(activated=None, checked=False, problem=None)
