from __future__ import annotations

import importlib.util
import os
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
    path = Path(storage_path or (Path.home() / ".photo-studio"))
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(Check("Storage", "ok", str(path)))
    except OSError as error:
        checks.append(Check("Storage", "error", str(error)))
    return checks


def overall_status(checks: list[Check]) -> str:
    return "error" if any(check.status == "error" for check in checks) else "ok"
