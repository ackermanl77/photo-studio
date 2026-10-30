from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

# pywebview is imported lazily inside main() (the only place the native window is created), not at
# module scope: it's a desktop/build-only dependency (requirements-desktop.txt) that CI and a plain
# `pip install -r requirements.txt` never install, and importing this module for its helpers
# (_log_path/_child_cwd/_streamlit_server_flags, the --self-test report) must not require it --
# the module-level import made tests/test_review_regressions.py fail with ModuleNotFoundError in CI.

# A file passed on the command line (the installer's .photoproject association, or "Open with")
# reaches the Streamlit child through this variable; the frontend opens it once per session.
OPEN_PATH_ENV = "PHOTO_STUDIO_OPEN_PATH"
_MUTEX_NAME = "Local\\PhotoStudioDesktopSingleInstance"
_WINDOW_TITLE = "Photo Studio"


def _frontend_port() -> int:
    from app.config import FRONTEND_PORT

    return int(os.getenv("PHOTO_FRONTEND_PORT", str(FRONTEND_PORT)))


def _frontend_url() -> str:
    return f"http://127.0.0.1:{_frontend_port()}"


def _free_port() -> int:
    """A port nothing is listening on right now (bind to port 0 and read back what the OS chose).
    The desktop build used fixed ports 8502/8504, so anything else holding either one -- or a second
    copy of the app -- made a child fail to bind and the window never opened."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _prepare_child_environment(argv: list[str]) -> None:
    """Per launch: free ports for both children, a fresh backend token (app.config.TOKEN_ENV), and
    the file to open, all passed through os.environ, which both children inherit. app.config reads
    the ports at import, so each child sees the values chosen here."""
    os.environ["PHOTO_BACKEND_PORT"] = str(_free_port())
    os.environ["PHOTO_FRONTEND_PORT"] = str(_free_port())
    os.environ["PHOTO_BACKEND_TOKEN"] = secrets.token_urlsafe(32)
    open_path = _open_path_argument(argv)
    if open_path:
        os.environ[OPEN_PATH_ENV] = open_path
    else:
        os.environ.pop(OPEN_PATH_ENV, None)


def _open_path_argument(argv: list[str]) -> str | None:
    """The first argument that names an existing file (flags and the exe itself skipped)."""
    for argument in argv[1:]:
        if not argument.startswith("--") and Path(argument).is_file():
            return str(Path(argument).resolve())
    return None


def _show_error(message: str) -> None:
    """A native message box -- the packaged exe has no console (console=False in
    PhotoStudio.spec), so a startup failure used to end the process with nothing on screen."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, _WINDOW_TITLE, 0x10)  # MB_ICONERROR
            return
        except (AttributeError, OSError):
            pass
    print(message, file=sys.stderr)


def _write_startup_error(path: Path) -> None:
    """Appends the current exception's traceback to `path` (the windowed exe has no console)."""
    import traceback
    from datetime import datetime

    try:
        with path.open("a", encoding="utf-8") as log:
            log.write(f"--- {datetime.now().isoformat(timespec='seconds')} startup failed\n{traceback.format_exc()}\n")
    except OSError:
        pass


def _acquire_single_instance():
    """A named Windows mutex held for the app's lifetime. Returns the handle, or None when another
    instance already holds it -- that one's window is brought to the front instead of starting a
    second pair of servers. Always "acquired" off Windows (the desktop build is Windows-only)."""
    if sys.platform != "win32":
        return True
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    if handle and kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(handle)
        window = ctypes.windll.user32.FindWindowW(None, _WINDOW_TITLE)
        if window:
            ctypes.windll.user32.ShowWindow(window, 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(window)
        return None
    return handle or True


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _webview2_runtime_folder() -> Path:
    """A machine without the Microsoft Edge WebView2 Runtime installed can't open this app's
    window at all (pywebview's edgechromium backend needs it) -- confirmed live on a second PC.
    scripts/build_release.ps1 bundles a pinned "Fixed Version" WebView2 Runtime as a sibling
    "WebView2Runtime" folder next to the packaged .exe specifically so the release doesn't depend
    on the target machine having it. Setting webview.settings['WEBVIEW2_RUNTIME_PATH'] to that
    folder (done in main(), below) makes pywebview use it; pywebview itself already falls back to
    the system-installed runtime with just a debug-log warning if the path doesn't exist (see
    webview/platforms/edgechromium.py), which is exactly what happens in dev mode, where this
    folder is never bundled."""
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else _project_root()
    return base / "WebView2Runtime"


def _log_path(name: str) -> Path:
    """Child-process logs go to the per-user data folder (app.config.user_data_dir()), not next to
    PhotoStudio.exe: an installed build lives under Program Files, which a normal user can't write
    to, so opening a log there raised PermissionError and the app never started."""
    from app.config import user_data_dir

    folder = user_data_dir() / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def _child_cwd() -> Path:
    """Working directory for both child processes. Dev mode needs the project root so
    `python -m app.backend` / the relative frontend path resolve. A frozen build doesn't (its
    children import from the bundle), and must not use the install folder, which under Program
    Files is unwritable for anything a child might write relative to its working directory."""
    if getattr(sys, "frozen", False):
        from app.config import user_data_dir

        return user_data_dir()
    return _project_root()


def _streamlit_server_flags() -> list[str]:
    """Every `streamlit run` flag both child command lines (frozen `--streamlit-child` and dev
    `python -m streamlit run`) pass. The theme/telemetry/upload-size ones are passed explicitly
    rather than left to .streamlit/config.toml: Streamlit reads that file relative to the working
    directory, and a frozen build's children run in user_data_dir() (see _child_cwd()), where no
    config exists -- so the shipped app silently lost the pinned dark theme the header CSS relies
    on and kept Streamlit's usage telemetry on (its default). maxUploadSize matches the backend's
    own byte cap (app.config.MAX_UPLOAD_BYTES), so a file the backend would reject with a 413 is
    refused by the uploader up front instead of after a full upload."""
    from app.config import MAX_UPLOAD_BYTES

    return [
        "--server.headless", "true",
        "--server.address", "127.0.0.1",
        "--server.port", str(_frontend_port()),
        "--global.developmentMode", "false",
        "--server.fileWatcherType", "none",
        "--theme.base", "dark",
        # As in .streamlit/config.toml: stated colours stop the browser-console warnings about
        # empty sidebar colours (UI14).
        "--theme.primaryColor", "#ff4b4b",
        "--theme.textColor", "#fafafa",
        "--theme.sidebar.primaryColor", "#ff4b4b",
        "--theme.sidebar.textColor", "#fafafa",
        "--browser.gatherUsageStats", "false",
        "--server.maxUploadSize", str(max(1, MAX_UPLOAD_BYTES // (1024 * 1024))),
        "--client.toolbarMode", "minimal",
    ]


def _run_streamlit_child() -> None:
    from streamlit.web.cli import main

    if getattr(sys, "frozen", False):
        frontend_path = Path(__file__).parent / "app" / "frontend_unified_app.py"
    else:
        frontend_path = Path(__file__).parent / "frontend_unified_app.py"
    sys.argv = ["streamlit", "run", str(frontend_path), *_streamlit_server_flags()]
    main()


def _run_backend_child() -> None:
    from app.backend import serve

    serve()


def _wait_for_frontend(process: subprocess.Popen) -> None:
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError("The embedded Streamlit frontend stopped before it was ready")
        try:
            if requests.get(_frontend_url(), timeout=0.5).ok:
                return
        except requests.RequestException:
            time.sleep(0.25)
    raise RuntimeError("The embedded Streamlit frontend did not start")


def _start_backend() -> subprocess.Popen:
    command = [sys.executable, "--backend-child"] if getattr(sys, "frozen", False) else [sys.executable, "-m", "app.backend"]
    log_path = _log_path("backend-child.log")
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(_child_cwd()),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    process._photo_log_file = log_file
    return process


def _wait_for_backend(process: subprocess.Popen) -> None:
    from app.config import BACKEND_HOST

    backend_health_url = f"http://{BACKEND_HOST}:{os.environ.get('PHOTO_BACKEND_PORT', '8502')}/health"
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError("The local processing backend stopped before it was ready")
        try:
            if requests.get(backend_health_url, timeout=0.5).ok:
                return
        except requests.RequestException:
            time.sleep(0.25)
    raise RuntimeError("The local processing backend did not start")


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    log_file = getattr(process, "_photo_log_file", None)
    if log_file:
        log_file.close()


def _start_frontend() -> subprocess.Popen:
    command = [sys.executable, "--streamlit-child"] if getattr(sys, "frozen", False) else [sys.executable, "-m", "streamlit"]
    if not getattr(sys, "frozen", False):
        command.extend(["run", str(_project_root() / "app" / "frontend_unified_app.py")])
    command.extend(_streamlit_server_flags())
    log_path = _log_path("desktop-child.log")
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(_child_cwd()),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    process._photo_log_file = log_file
    return process


# Library modules --self-test imports: the packaged build is expected to ship every one. A frozen
# build that silently dropped one (PyInstaller missing a hidden import) fails here, at build time,
# instead of on a user's machine. app.frontend_unified_app is a Streamlit *script* (it renders on
# import), so it is only located, never executed.
_SELF_TEST_MODULES = (
    "app.backend", "app.pipeline", "app.edit_schema", "app.export", "app.image_processor",
    "app.professional_color", "app.lens_corrections", "app.portrait_tools", "app.layer_workspace",
    "app.project_bundle", "app.custom_presets", "app.batch_release", "app.cli",
    # The UI package the Streamlit script composes (F14), and the registry/static theme it reads.
    "app.controls", "app.gpu_pack", "app.ui.canvas", "app.ui.export_panel", "app.ui.tabs.library",
)


def self_test_report() -> dict:
    """What `PhotoStudio.exe --self-test` reports (scripts/build_release.ps1 runs it right after
    PyInstaller): every module in _SELF_TEST_MODULES imports, the Streamlit script is bundled, and
    the capabilities every build profile promises are present -- rawpy (both profiles install
    requirements-raw.txt, so rawpy and lensfun's lens database) and a working face detector for the Portrait tab. Whether AI Upscale's
    torch stack is installed is reported but not required (it ships in the optional GPU pack, or in
    the all-in-one gpu profile)."""
    import importlib
    import importlib.util

    modules = {}
    for name in _SELF_TEST_MODULES:
        try:
            importlib.import_module(name)
            modules[name] = "ok"
        except Exception as error:  # noqa: BLE001 -- report every failure, never crash the report
            modules[name] = f"error: {error}"
    modules["app.frontend_unified_app"] = "ok" if importlib.util.find_spec("app.frontend_unified_app") else "missing"

    from app.ai_upscale import ai_upscale_available, ai_upscale_installed
    from app.export import EXPORT_FORMATS
    from app.gpu_pack import gpu_pack_status
    from app.raw_import import SUPPORTED_EXTENSIONS
    from app.lens_lookup import lens_database_size
    from app.portrait_tools import face_detection_backend
    from app.ui.theme import page_style

    modules["app/ui/static"] = "ok" if "--ps-" in page_style() else "missing"
    installed = ai_upscale_installed()
    capabilities = {
        "rawpy": importlib.util.find_spec("rawpy") is not None,
        # HEIC/AVIF import and AVIF export (F19): the frozen build must bundle pillow-heif's libheif.
        "heif": ".heic" in SUPPORTED_EXTENSIONS,
        "avif": "AVIF" in EXPORT_FORMATS,
        "lensfun_lenses": lens_database_size(),
        "face_detection": face_detection_backend(),
        "ai_upscale_installed": installed,
        # A real import of torch + Real-ESRGAN (seconds) when the packages are present -- a GPU pack
        # can be found on sys.path yet fail to import in the frozen interpreter.
        "ai_upscale_importable": ai_upscale_available() if installed else False,
        # The optional GPU pack (K5): found and on sys.path? (PHOTO_STUDIO_GPU_PACK points the
        # build's second self-test at a freshly built pack.)
        "gpu_pack": gpu_pack_status(),
    }
    ok = (
        all(status == "ok" for status in modules.values())
        and capabilities["rawpy"]
        and capabilities["heif"]
        and capabilities["lensfun_lenses"] > 0
        and capabilities["face_detection"] != "none"
    )
    return {
        "ok": ok, "python": sys.version.split()[0], "frozen": bool(getattr(sys, "frozen", False)),
        "modules": modules, "capabilities": capabilities,
    }


def _run_self_test(argv: list[str]) -> int:
    """`--self-test [--self-test-report PATH]`: prints the report as JSON, also writes it to PATH
    when given (the windowed release exe has no console to print to, so the build script reads the
    file instead), and returns the exit code: 0 when every required check passed, 1 otherwise."""
    import json

    report = self_test_report()
    text = json.dumps(report, indent=2, sort_keys=True)
    if "--self-test-report" in argv:
        index = argv.index("--self-test-report")
        if index + 1 < len(argv):
            Path(argv[index + 1]).write_text(text, encoding="utf-8")
    if sys.stdout is not None:
        print(text)
    return 0 if report["ok"] else 1


def main() -> None:
    if "--self-test" in sys.argv:
        raise SystemExit(_run_self_test(sys.argv))
    if "--streamlit-child" in sys.argv:
        _run_streamlit_child()
        return
    if "--backend-child" in sys.argv:
        _run_backend_child()
        return

    from app.observability import configure_logging

    logger = configure_logging("desktop").getChild("desktop")
    instance = _acquire_single_instance()
    if instance is None:
        return  # another copy is running; its window was brought to the front

    # The backend starts first and is awaited before the frontend: by the time the Streamlit
    # child's frontend_unified_app.py runs, its own ensure_backend_running() health check finds
    # the backend already up and does nothing further (see app/backend_client.py).
    _prepare_child_environment(sys.argv)
    backend = None
    frontend = None
    try:
        backend = _start_backend()
        _wait_for_backend(backend)
        frontend = _start_frontend()
        _wait_for_frontend(frontend)
        import webview

        webview.settings["WEBVIEW2_RUNTIME_PATH"] = str(_webview2_runtime_folder())
        webview.create_window(_WINDOW_TITLE, _frontend_url(), width=1440, height=960, min_size=(1024, 700))
        webview.start(gui="edgechromium")
    except Exception as error:  # noqa: BLE001 -- every startup failure gets a visible message
        logger.exception("Desktop startup failed")
        details = _log_path("desktop-startup-error.log")
        _write_startup_error(details)
        _show_error(f"Photo Studio could not start: {error}\n\nDetails are in the log files in:\n{details.parent}")
        # Exit instead of re-raising: in the windowed exe an uncaught exception also makes
        # PyInstaller show its own "Unhandled exception" box, so the user got two dialogs.
        raise SystemExit(1) from None
    finally:
        if frontend is not None:
            _terminate(frontend)
        if backend is not None:
            _terminate(backend)


if __name__ == "__main__":
    main()
