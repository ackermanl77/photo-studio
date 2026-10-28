from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    status: str
    message: str


def check_environment(storage_path: str | Path | None = None) -> list[Check]:
    checks: list[Check] = []
    checks.append(Check("Python", "ok" if sys.version_info >= (3, 10) else "error", sys.version.split()[0]))
    for package in ["PIL", "numpy", "cv2", "streamlit", "flask", "tifffile"]:
        checks.append(Check(package, "ok" if importlib.util.find_spec(package) else "missing", "installed" if importlib.util.find_spec(package) else "not installed"))
    checks.append(Check("RAW support", "ok" if importlib.util.find_spec("rawpy") else "optional", "rawpy installed" if importlib.util.find_spec("rawpy") else "optional rawpy package missing"))
    checks.append(Check("7-Zip", "ok" if shutil.which("7z") else "optional", "available" if shutil.which("7z") else "not found"))
    # The Portrait tab's face count depends on a detector that has already silently vanished once
    # (OpenCV 5 dropped the Haar cascade API) -- an explicit line here makes that visible.
    from app.portrait_tools import face_detection_backend

    face_backend = face_detection_backend()
    checks.append(
        Check(
            "Face detection", "ok" if face_backend != "none" else "error",
            {"yunet": "YuNet (cv2.FaceDetectorYN)", "haar": "Haar cascade (fallback)"}.get(face_backend, "no detector available in this OpenCV build"),
        )
    )
    # AI Upscale's optional torch stack: from the GPU pack in a packaged build (K5), or
    # requirements-gpu.txt in a source checkout.
    from app.ai_upscale import ai_upscale_install_hint, ai_upscale_installed
    from app.gpu_pack import gpu_pack_status

    pack = gpu_pack_status()
    if ai_upscale_installed():
        checks.append(Check("AI Upscale", "ok", f"GPU pack at {pack['path']}" if pack["active"] else "torch / Real-ESRGAN installed"))
    else:
        checks.append(Check("AI Upscale", "warning" if pack["problem"] else "optional", ai_upscale_install_hint()))
    from app.ai_denoise import ai_denoise_installed, weights_present

    if ai_denoise_installed():
        checks.append(Check("AI Denoise", "ok", "torch installed; model " + ("downloaded" if weights_present() else "downloads on first use")))
    else:
        checks.append(Check("AI Denoise", "warning" if pack["problem"] else "optional", "Needs the GPU pack; the classical Denoise stands in."))
    if storage_path is None:
        from app.config import user_data_dir

        storage_path = user_data_dir()
    path = Path(storage_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(Check("Storage", "ok", str(path)))
    except OSError as error:
        checks.append(Check("Storage", "error", str(error)))
    return checks




def logs_folder() -> Path:
    """The folder the rotating logs are written to (app.observability.configure_logging)."""
    from app.config import user_data_dir

    return user_data_dir() / "logs"


def open_folder(path: str | Path) -> None:
    """Show `path` in the OS file manager (created first if missing). The Streamlit server runs on
    the user's own machine -- the desktop app's child process, or a local `streamlit run` -- so this
    opens it there. Raises OSError when no file manager can be started."""
    import os
    import subprocess

    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(folder)  # noqa: S606 -- a fixed local folder, not user input
    else:
        opener = shutil.which("open" if sys.platform == "darwin" else "xdg-open")
        if opener is None:
            raise OSError("no file manager found (xdg-open)")
        subprocess.Popen([opener, str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
