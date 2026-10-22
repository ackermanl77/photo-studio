"""Opt-in real-browser smoke test: what AppTest can't see.

Runs only with PHOTO_BROWSER_TESTS=1 and a Chrome or Edge on this machine (driven over the
DevTools protocol with the `websockets` package -- no Selenium/Playwright needed):

    $env:PHOTO_BROWSER_TESTS="1"; python -m pytest -q tests/test_browser_smoke.py

It starts the real Streamlit app (which starts its own backend) and a headless browser, uploads a
photo large enough for the proxy preview, then checks:

1. **refine-on-pause** -- with no further interaction, the polling status fragment moves from the
   fast preview to "Full-resolution view" (fragment-only reruns plus the deferred app rerun that
   swaps the canvas image);
2. **an on-canvas editor** -- a real mouse drag on the tone curve's midtone point inside its
   ``components.html`` iframe commits through the hidden text area, and the app's Midtones slider
   (``c-p2``) follows;
3. **an ordinary control** -- moving the Exposure slider with the keyboard re-renders the canvas.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("PHOTO_BROWSER_TESTS") != "1", reason="set PHOTO_BROWSER_TESTS=1")

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]


def _browser() -> str | None:
    for candidate in BROWSERS:
        if os.path.isfile(candidate):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait(predicate, timeout: float, what: str, interval: float = 0.25):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
            if last:
                return last
        except Exception as exc:  # noqa: BLE001 -- keep polling; reported below
            last = exc
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what} (last: {last!r})")


def _kill_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    else:
        process.kill()
    process.wait(10)


class DevTools:
    """A minimal Chrome DevTools Protocol client over one page target."""

    def __init__(self, ws_url: str):
        from websockets.sync.client import connect

        self.ws = connect(ws_url, max_size=None, open_timeout=20)
        self.next_id = 0

    def send(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        self.next_id += 1
        message_id = self.next_id
        self.ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while True:
            reply = json.loads(self.ws.recv(timeout=max(0.1, deadline - time.monotonic())))
            if reply.get("id") == message_id:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply.get("result", {})

    def eval(self, expression: str):
        result = self.send("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise RuntimeError(f"{(details.get('exception') or {}).get('description') or details.get('text')} "
                               f"in: {expression.strip()[:120]}")
        return result["result"].get("value")

    def mouse(self, kind: str, x: float, y: float, buttons: int = 0) -> None:
        self.send("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, "button": "left",
                                               "buttons": buttons, "clickCount": 1 if kind != "mouseMoved" else 0})

    def key(self, key: str, code: str, key_code: int) -> None:
        for kind in ("keyDown", "keyUp"):
            self.send("Input.dispatchKeyEvent", {"type": kind, "key": key, "code": code,
                                                 "windowsVirtualKeyCode": key_code})

    def close(self) -> None:
        self.ws.close()


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    browser = _browser()
    if browser is None:
        pytest.skip("no Chrome/Edge found")
    pytest.importorskip("websockets.sync.client")
    data_dir = tmp_path_factory.mktemp("browser-data")
    app_port, debug_port = _free_port(), _free_port()
    env = dict(os.environ, PHOTO_STUDIO_DATA_DIR=str(data_dir), PHOTO_BACKEND_TOKEN="")
    app = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/frontend_unified_app.py", "--server.headless", "true",
         "--server.port", str(app_port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    profile = tempfile.mkdtemp(prefix="photo_browser_")
    chrome = subprocess.Popen(
        [browser, "--headless=new", f"--remote-debugging-port={debug_port}", f"--user-data-dir={profile}",
         "--window-size=1600,1100", "--no-first-run", "--disable-gpu", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    devtools = None
    try:
        _wait(lambda: urllib.request.urlopen(f"http://127.0.0.1:{app_port}/_stcore/health", timeout=2).status == 200,
              180, "the Streamlit app")
        targets = _wait(lambda: [t for t in json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{debug_port}/json/list", timeout=2).read()) if t.get("type") == "page"], 30, "the browser")
        devtools = DevTools(targets[0]["webSocketDebuggerUrl"])
        devtools.send("Page.enable")
        devtools.send("DOM.enable")
        devtools.send("Page.navigate", {"url": f"http://127.0.0.1:{app_port}/"})
        _wait(lambda: devtools.eval("!!document.querySelector('input[type=file]')"), 90, "the uploader")
        yield devtools, tmp_path_factory.mktemp("browser-upload")
    finally:
        if devtools is not None:
            devtools.close()
        _kill_tree(chrome)
        _kill_tree(app)
        shutil.rmtree(profile, ignore_errors=True)


def _canvas_src(devtools) -> str:
    """The largest image on the page: the canvas."""
    return devtools.eval("""(() => {
        const imgs = [...document.querySelectorAll('[data-testid="stImage"] img, img')];
        imgs.sort((a, b) => b.naturalWidth * b.naturalHeight - a.naturalWidth * a.naturalHeight);
        return imgs.length ? imgs[0].src.slice(-80) + ':' + imgs[0].naturalWidth : '';
    })()""") or ""


def _text(devtools) -> str:
    return devtools.eval("document.body.innerText") or ""


def test_refine_on_pause_editor_drag_and_a_slider_in_a_real_browser(page):
    devtools, folder = page
    from tests.synthetic_assets import photo_like

    photo = folder / "large.jpg"
    photo_like((3200, 2000)).save(photo, quality=90)  # over PROXY_TRIGGER_LONG_EDGE: a proxy preview
    document = devtools.send("DOM.getDocument", {"depth": -1, "pierce": True})
    node = devtools.send("DOM.querySelector", {"nodeId": document["root"]["nodeId"], "selector": "input[type=file]"})
    devtools.send("DOM.setFileInputFiles", {"nodeId": node["nodeId"], "files": [str(photo)]})
    _wait(lambda: _canvas_src(devtools), 120, "the canvas image")

    # 1. Refine-on-pause: nothing touched, the polling fragment brings in the full resolution.
    _wait(lambda: "Full-resolution view" in _text(devtools), 180, "the full-resolution view", interval=0.5)
    refined = _canvas_src(devtools)
    assert refined

    # 2. A real drag on the tone curve's midtone point, inside the editor's iframe.
    devtools.eval("""(() => { const tab = [...document.querySelectorAll('[data-testid=stTab]')]
        .find(b => b.innerText.includes('Curves')); tab.click(); return !!tab; })()""")
    devtools.eval("""(() => { const summary = [...document.querySelectorAll('[data-testid=stExpander] summary')]
        .find(e => e.innerText.includes('Tone Curve')); if (summary && !summary.parentElement.open) summary.click();
        return !!summary; })()""")
    _wait(lambda: devtools.eval("""(() => [...document.querySelectorAll('iframe')]
        .some(f => f.contentDocument && f.contentDocument.getElementById('curve-svg')))()"""), 60, "the curve editor")
    before = devtools.eval("""document.querySelector('.st-key-c-p2 input[type=range]')?.value""")
    point = devtools.eval("""(() => {
        const frame = [...document.querySelectorAll('iframe')].find(f => f.contentDocument &&
            f.contentDocument.getElementById('curve-svg'));
        frame.scrollIntoView({block: 'center'});
        const fr = frame.getBoundingClientRect();
        const pts = frame.contentDocument.querySelectorAll('#curve-svg .pt.fixed');
        const r = pts[2].getBoundingClientRect();
        return [fr.left + r.left + r.width / 2, fr.top + r.top + r.height / 2];
    })()""")
    x, y = point
    devtools.mouse("mouseMoved", x, y)
    devtools.mouse("mousePressed", x, y, buttons=1)
    for step in range(1, 9):
        devtools.mouse("mouseMoved", x, y - 5 * step, buttons=1)
    devtools.mouse("mouseReleased", x, y - 40)
    after = _wait(lambda: (lambda v: v if v not in (None, before) else None)(devtools.eval(
        "document.querySelector('.st-key-c-p2 input[type=range]')?.value")),
        60, "the Midtones slider to follow the drag")
    assert float(after) > float(before or 128)

    # 3. An ordinary slider: Exposure up by keyboard re-renders the canvas.
    devtools.eval("""(() => { const tab = [...document.querySelectorAll('[data-testid=stTab]')]
        .find(b => b.innerText.includes('Basic')); tab.click(); return true; })()""")
    current = _canvas_src(devtools)
    devtools.eval("document.querySelector('.st-key-t-ev input[type=range]').focus()")
    for _ in range(6):
        devtools.key("ArrowRight", "ArrowRight", 39)
    _wait(lambda: _canvas_src(devtools) not in ("", current), 120, "the canvas to re-render after Exposure")
    assert "Traceback" not in _text(devtools)
