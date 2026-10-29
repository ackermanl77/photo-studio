"""Third-party Photoshop filter plug-ins (.8bf) as a non-destructive Effects step (PS20).

The plug-ins run in ``PhotoStudio.FilterHost.exe`` (tools/filter_host), a small helper built on
PSFilterHost (Ms-PL, https://github.com/0xC0000054/PSFilterHost), in a process of its own: a
plug-in that crashes or hangs takes the helper down, not the editor. Windows only, like the
plug-ins themselves; scripts/build_filter_host.ps1 builds the helper.

A filter is applied twice over its life:
- once with its own dialog (``configure``): the user picks its settings on the current photo, and
  the helper returns them as an opaque blob (PSFilterHost's ``ParameterData``);
- then on every render and export without a dialog (``apply_filters``), replaying that blob, so the
  step is part of the recipe like any other edit.

Each step in the ``plugin_filters`` option is {"file": "...8bf", "entry": "...", "title": "...",
"params": "<base64>"}. Adobe's own filters aren't offered: they carry a JSON PiPL locked to
Photoshop ("RequiredHost 8BIM"), which PSFilterHost doesn't list.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

from PIL import Image

from app.observability import get_logger

HOST_NAME = "PhotoStudio.FilterHost.exe"
HOST_ENV = "PHOTO_STUDIO_FILTER_HOST"  # a helper path, for tests and source checkouts
MAX_FILTERS = 8  # steps per recipe
MAX_PARAMS_BYTES = 1_000_000  # one filter's saved settings
LIST_TIMEOUT = 60
APPLY_TIMEOUT = 300  # without the dialog
DIALOG_TIMEOUT = 3600  # the user is choosing settings in the plug-in's own window
_EXIT_CANCELLED = 3
_CACHE_ENTRIES = 6

logger = get_logger("filter_plugins")


class FilterPluginError(RuntimeError):
    """A plug-in filter couldn't run (helper missing, plug-in failed, bad settings)."""


class FilterCancelled(FilterPluginError):
    """The user closed the plug-in's dialog without applying it."""


def host_path() -> Path | None:
    """The helper executable: $PHOTO_STUDIO_FILTER_HOST, next to the packaged app (filterhost\\),
    or a source checkout's build\\filter_host\\. None when it isn't built/installed."""
    override = os.environ.get(HOST_ENV, "").strip()
    candidates = [Path(override)] if override else []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / "filterhost" / HOST_NAME)
        candidates.append(Path(getattr(sys, "_MEIPASS", "")) / "filterhost" / HOST_NAME)
    candidates.append(Path(__file__).resolve().parents[1] / "build" / "filter_host" / HOST_NAME)
    return next((path for path in candidates if path.is_file()), None)


def available() -> bool:
    return sys.platform == "win32" and host_path() is not None


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    host = host_path()
    if host is None or sys.platform != "win32":
        raise FilterPluginError("Photoshop plug-in filters need Windows and the filter host (scripts/build_filter_host.ps1).")
    try:
        return subprocess.run([str(host), *args], capture_output=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as error:
        raise FilterPluginError("The plug-in filter took too long and was stopped.") from error
    except OSError as error:
        raise FilterPluginError(f"The filter host couldn't start: {error}") from error


def list_filters(folder: str | Path, recursive: bool = True) -> list[dict]:
    """The filters in `folder` (and its subfolders): [{"file", "entry", "category", "title"}],
    sorted by category and title."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FilterPluginError(f"Not a folder: {folder}")
    args = ["list", str(folder)] + (["--recursive"] if recursive else [])
    result = _run(args, LIST_TIMEOUT)
    if result.returncode != 0:
        raise FilterPluginError(_message(result) or "The filter host couldn't read the plug-in folder.")
    try:
        filters = json.loads(result.stdout.decode("utf-8-sig"))
    except ValueError as error:
        raise FilterPluginError("The filter host returned an unreadable plug-in list.") from error
    cleaned = [
        {"file": item["file"], "entry": item["entry"], "category": item.get("category") or "",
         "title": (item.get("title") or "").rstrip(".").strip() or Path(item["file"]).stem}
        for item in filters if isinstance(item, dict) and item.get("file") and item.get("entry")
    ]
    return sorted(cleaned, key=lambda item: (item["category"].lower(), item["title"].lower()))


def _message(result: subprocess.CompletedProcess) -> str:
    return result.stderr.decode("utf-8", "replace").strip().splitlines()[-1] if result.stderr.strip() else ""


def _run_filter(image: Image.Image, step: dict, show_ui: bool) -> tuple[Image.Image, str | None]:
    alpha = image.getchannel("A") if image.mode == "RGBA" else None
    with tempfile.TemporaryDirectory(prefix="photo-studio-8bf-") as work:
        work = Path(work)
        source, output, params_in, params_out = (work / name for name in ("in.png", "out.png", "in.params", "out.params"))
        image.convert("RGBA").save(source, format="PNG")
        args = ["run", "--plugin", step["file"], "--entry", step["entry"], "--input", str(source),
                "--output", str(output), "--params-out", str(params_out)]
        if step.get("params"):
            params_in.write_bytes(base64.b64decode(step["params"]))
            args += ["--params-in", str(params_in)]
        if show_ui:
            args.append("--ui")
        result = _run(args, DIALOG_TIMEOUT if show_ui else APPLY_TIMEOUT)
        if result.returncode == _EXIT_CANCELLED:
            raise FilterCancelled("The filter was cancelled.")
        if result.returncode != 0 or not output.is_file():
            raise FilterPluginError(_message(result) or f"{step.get('title') or 'The filter'} failed.")
        with Image.open(output) as rendered:
            filtered = rendered.convert("RGBA")
        params = base64.b64encode(params_out.read_bytes()).decode("ascii") if params_out.is_file() else step.get("params")
    if filtered.size != image.size:
        raise FilterPluginError("The filter changed the image's size, which isn't supported.")
    if alpha is None:
        filtered = filtered.convert("RGB")
    else:
        filtered.putalpha(alpha)
    return filtered, params


def configure(image: Image.Image, step: dict) -> dict:
    """Show the plug-in's own dialog on `image` and return the step with the settings chosen there
    ("params"). Raises FilterCancelled when the dialog is closed without applying."""
    _filtered, params = _run_filter(image, {**step, "params": None}, show_ui=True)
    if not params:
        raise FilterPluginError("The filter didn't return its settings, so it can't be replayed.")
    return {**step, "params": params}


_cache: OrderedDict[str, bytes] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(image: Image.Image, step: dict) -> str:
    digest = hashlib.sha256(image.tobytes())
    digest.update(f"{image.mode}{image.size}{step['file']}|{step['entry']}|{step.get('params')}".encode())
    return digest.hexdigest()


def apply_filters(image: Image.Image, steps: list[dict]) -> Image.Image:
    """Replay each step's saved settings, in order, without dialogs. A recent identical input reuses
    its result: plug-ins can take seconds, and a re-render often feeds the same pixels."""
    for step in steps:
        key = _cache_key(image, step)
        with _cache_lock:
            cached = _cache.get(key)
            if cached is not None:
                _cache.move_to_end(key)
        if cached is not None:
            with Image.open(io.BytesIO(cached)) as hit:
                image = hit.convert(image.mode)
            continue
        image, _params = _run_filter(image, step, show_ui=False)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        with _cache_lock:
            _cache[key] = buffer.getvalue()
            while len(_cache) > _CACHE_ENTRIES:
                _cache.popitem(last=False)
    return image


def clean_steps(value) -> list[dict]:
    """Validate the `plugin_filters` option: at most MAX_FILTERS steps, each with a .8bf file, an
    entry point, a title and saved settings (base64, at most MAX_PARAMS_BYTES)."""
    if not isinstance(value, list) or len(value) > MAX_FILTERS:
        raise ValueError(f"plugin_filters must be a list of at most {MAX_FILTERS} filters")
    steps = []
    for index, step in enumerate(value):
        if not isinstance(step, dict) or set(step) - {"file", "entry", "title", "params"}:
            raise ValueError(f'plugin_filters[{index}] must be {{"file", "entry", "title", "params"}}')
        file, entry, title, params = step.get("file"), step.get("entry"), step.get("title", ""), step.get("params")
        if not isinstance(file, str) or not file.lower().endswith(".8bf") or not isinstance(entry, str) or not entry:
            raise ValueError(f"plugin_filters[{index}] needs a .8bf file and an entry point")
        if not isinstance(title, str) or not isinstance(params, str) or not params:
            raise ValueError(f"plugin_filters[{index}] needs a title and its saved settings")
        try:
            raw = base64.b64decode(params, validate=True)
        except ValueError as error:
            raise ValueError(f"plugin_filters[{index}].params isn't base64") from error
        if len(raw) > MAX_PARAMS_BYTES:
            raise ValueError(f"plugin_filters[{index}].params is larger than {MAX_PARAMS_BYTES} bytes")
        steps.append({"file": file, "entry": entry, "title": title[:120], "params": params})
    return steps
