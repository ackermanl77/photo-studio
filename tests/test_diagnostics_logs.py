"""R2: System Health's "Open logs folder"."""
from __future__ import annotations

import sys

import pytest

from app import diagnostics
from app.config import user_data_dir


def test_logs_folder_is_under_the_user_data_folder():
    assert diagnostics.logs_folder() == user_data_dir() / "logs"


def test_open_folder_creates_the_folder_and_hands_it_to_the_os(monkeypatch, tmp_path):
    opened = []
    target = tmp_path / "logs"
    if sys.platform == "win32":
        monkeypatch.setattr("os.startfile", lambda path: opened.append(path), raising=False)
    else:
        monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr("subprocess.Popen", lambda args, **_kw: opened.append(args[-1]))
    diagnostics.open_folder(target)
    assert target.is_dir() and str(opened[0]) == str(target)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows always has a file manager")
def test_open_folder_without_a_file_manager_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: None)
    with pytest.raises(OSError):
        diagnostics.open_folder(tmp_path / "logs")
