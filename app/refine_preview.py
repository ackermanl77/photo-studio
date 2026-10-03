"""Refine-on-pause for the proxy preview (P1 in the improvement report) -- Streamlit-free.

A large photo is previewed from a PROXY_LONG_EDGE proxy so every slider move stays fast. Once the
edit stops changing for PAUSE_SECONDS, the frontend renders the same edit at full resolution in a
background thread and swaps it in for the canvas, histogram and export, until the edit changes again.

Why a background thread rather than rendering in the script: a full-resolution render can take
10-40 s, and a Streamlit script blocked in it can't start the next rerun -- the first slider move
after a pause would freeze the UI until it finished. In a thread, the next rerun starts at once
and renders the proxy as usual; its backend request supersedes the refine (the backend's "refine"
lane yields to any newer interactive request of the same session -- see app.backend), so the
obsolete full-resolution job stops at its next pipeline step instead of competing for the CPU.

RefineTracker is kept in st.session_state (one per browser session). The frontend calls
observe() every full run with the current edit key (None when no proxy is in use), then
maybe_start() from a polling st.fragment; result() returns the finished image for the current key.
"""
from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable

from PIL import Image

# How long the edit must stay unchanged before the full-resolution render starts, and how often the
# canvas fragment checks while one is pending. Module attributes so tests can change them.
PAUSE_SECONDS = 1.2
POLL_SECONDS = 0.75
# False runs the job inline (tests); the app always uses a thread.
RUN_IN_THREAD = True


class RefineJob:
    def __init__(self, key: str):
        self.key = key
        self.status = "running"  # -> "ready" | "superseded" | "failed"
        self.image: Image.Image | None = None
        self.error: str | None = None
        self.done = threading.Event()

    def run(self, render: Callable[[], Image.Image]) -> None:
        try:
            self.image = render()
            self.status = "ready"
        except Exception as error:  # noqa: BLE001 -- any failure just means "keep the proxy"
            self.status = "superseded" if type(error).__name__ == "SupersededError" else "failed"
            self.error = str(error)
        finally:
            self.done.set()


class RefineTracker:
    """Pause detection plus the latest job, for one session."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self.key: str | None = None
        self.changed_at = 0.0
        self.job: RefineJob | None = None
        self.render: Callable[[], Image.Image] | None = None
        self._announced: str | None = None

    def observe(self, key: str | None, render: Callable[[], Image.Image] | None = None) -> None:
        """Every full script run: the current edit's key (None = nothing to refine) and the bound,
        Streamlit-free callable that renders it at full resolution."""
        if key != self.key:
            self.key = key
            self.changed_at = self._clock()
        self.render = render if key is not None else None

    def result(self, key: str | None) -> Image.Image | None:
        job = self.job
        if key is None or job is None or job.key != key or job.status != "ready":
            return None
        self._announced = key  # being shown by a full run already -- no extra rerun needed
        return job.image

    def pending(self) -> bool:
        """True while the canvas should keep polling: waiting for the pause, or a job running."""
        if self.key is None or self.render is None:
            return False
        job = self.job
        if job is not None and job.key == self.key:
            return job.status == "running"
        return True

    def status(self) -> str:
        """"off" (no proxy), "waiting", "running", "ready", "failed" for the current key."""
        if self.key is None:
            return "off"
        job = self.job
        if job is None or job.key != self.key:
            return "waiting"
        return "failed" if job.status == "superseded" else job.status

    def maybe_start(self) -> bool:
        """Start the full-resolution job once the edit has been unchanged for PAUSE_SECONDS.
        Returns True when it started one."""
        if not self.pending() or (self.job is not None and self.job.key == self.key):
            return False
        if self._clock() - self.changed_at < PAUSE_SECONDS:
            return False
        job = RefineJob(self.key)
        self.job = job
        render = self.render
        if RUN_IN_THREAD:
            # copy_context(): the thread inherits the edit session bound by bind_edit_session(), so
            # its request is tagged with it and newer interactive requests can supersede it.
            context = contextvars.copy_context()
            threading.Thread(target=context.run, args=(job.run, render), name="photo-refine", daemon=True).start()
        else:
            job.run(render)
        return True

    def needs_rerun(self) -> bool:
        """True once per finished job for the current key -- the moment the canvas must rerun the
        whole page to show it (or to stop polling after a failure)."""
        job = self.job
        if job is None or job.key != self.key or job.status == "running" or self._announced == job.key:
            return False
        self._announced = job.key
        return True
