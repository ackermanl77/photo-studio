from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreflightCheck:
    name: str
    status: str
    message: str


REQUIRED_ENTRIES = [
    "app/frontend_unified_app.py",
    "app/desktop.py",
    "PhotoStudio.spec",
    "requirements.txt",
    "requirements-desktop.txt",
]
REQUIRED_MODULES = ["PIL", "numpy", "cv2", "streamlit", "flask", "requests", "tifffile"]


def run_preflight(root: str | Path, run_tests: bool = False) -> list[PreflightCheck]:
    project = Path(root)
    checks: list[PreflightCheck] = []
    for relative in REQUIRED_ENTRIES:
        path = project / relative
        checks.append(PreflightCheck(relative, "ok" if path.exists() else "error", "present" if path.exists() else "missing"))
    for module in REQUIRED_MODULES:
        present = importlib.util.find_spec(module) is not None
        checks.append(PreflightCheck(f"package:{module}", "ok" if present else "error", "importable" if present else "missing"))
    if run_tests:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=project, capture_output=True, text=True)
        checks.append(PreflightCheck("pytest", "ok" if result.returncode == 0 else "error", result.stdout.splitlines()[-1] if result.stdout else result.stderr[-200:]))
    return checks


def preflight_ok(checks: list[PreflightCheck]) -> bool:
    return all(check.status == "ok" for check in checks)
