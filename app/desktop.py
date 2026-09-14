from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import webview

FRONTEND_PORT = 8504
FRONTEND_URL = f"http://127.0.0.1:{FRONTEND_PORT}"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_streamlit_child() -> None:
    from streamlit.web.cli import main

    if getattr(sys, "frozen", False):
        frontend_path = Path(__file__).parent / "app" / "frontend_unified_app.py"
    else:
        frontend_path = Path(__file__).parent / "frontend_unified_app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(frontend_path),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(FRONTEND_PORT),
        "--global.developmentMode",
        "false",
        "--server.fileWatcherType",
        "none",
    ]
    main()


def _run_backend_child() -> None:
    from app.backend import app
    from app.config import BACKEND_HOST, BACKEND_PORT

    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=False, use_reloader=False)


def _wait_for_frontend(process: subprocess.Popen) -> None:
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError("The embedded Streamlit frontend stopped before it was ready")
        try:
            if requests.get(FRONTEND_URL, timeout=0.5).ok:
                return
        except requests.RequestException:
            time.sleep(0.25)
    raise RuntimeError("The embedded Streamlit frontend did not start")


def _start_frontend() -> subprocess.Popen:
    command = [sys.executable, "--streamlit-child"] if getattr(sys, "frozen", False) else [sys.executable, "-m", "streamlit"]
    if not getattr(sys, "frozen", False):
        command.extend(["run", str(_project_root() / "app" / "frontend_unified_app.py")])
    command.extend(
        [
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(FRONTEND_PORT),
            "--global.developmentMode",
            "false",
            "--server.fileWatcherType",
            "none",
        ]
    )
    log_path = Path(sys.executable).resolve().parent / "desktop-child.log"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(_project_root()),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    process._photo_log_file = log_file
    return process


def main() -> None:
    if "--streamlit-child" in sys.argv:
        _run_streamlit_child()
        return
    if "--backend-child" in sys.argv:
        _run_backend_child()
        return

    frontend = _start_frontend()
    try:
        _wait_for_frontend(frontend)
        webview.create_window("Photo Studio", FRONTEND_URL, width=1440, height=960, min_size=(1024, 700))
        webview.start(gui="edgechromium")
    finally:
        if frontend.poll() is None:
            frontend.terminate()
            try:
                frontend.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend.kill()
        log_file = getattr(frontend, "_photo_log_file", None)
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
