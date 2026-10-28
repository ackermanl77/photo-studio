"""Verified downloads of the optional AI models' weights (AI Upscale, AI Denoise).

Each model file lives in the per-user data folder (app.config.user_data_dir() / "weights"), never
inside the installed app, which a normal user can't write to. A download is written to a
temporary file and renamed only after its SHA-256 matches, so an interrupted or substituted
download never leaves a file that looks complete; the hash is checked again before every load.
"""
from __future__ import annotations

import hashlib


def weights_dir():
    from app.config import user_data_dir

    return user_data_dir() / "weights"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, target, sha256: str, size_bytes: int, *, progress=None, opener=None,
                      failure: str = "The downloaded model failed its checksum -- try again (the file was discarded)."):
    """Download `url` to `target`, verifying its SHA-256, and return `target`. `progress(done_bytes,
    total_bytes)` is called as data arrives. `opener` (a urllib-style callable) exists for tests."""
    import os
    import tempfile
    import urllib.request

    if target.is_file() and target.stat().st_size == size_bytes and sha256_file(target) == sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    digest = hashlib.sha256()
    done = 0
    try:
        # A stalled connection must fail (and be retried by the user) rather than hang the request.
        open_url = opener or (lambda address: urllib.request.urlopen(address, timeout=60))
        with os.fdopen(handle, "wb") as out, open_url(url) as response:
            total = int(response.headers.get("Content-Length") or size_bytes)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        if digest.hexdigest() != sha256:
            raise RuntimeError(failure)
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return target


def verified_weights(url: str, target, sha256: str, size_bytes: int):
    """The local weights file, downloaded (verified) on first use when missing. A file on disk whose
    hash doesn't match (e.g. a partial copy) is replaced rather than loaded."""
    if target.is_file() and sha256_file(target) == sha256:
        return target
    if target.is_file():
        target.unlink()
    return download_verified(url, target, sha256, size_bytes)
