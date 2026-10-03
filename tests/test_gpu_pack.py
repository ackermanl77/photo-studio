"""The optional GPU pack (K5): detection by the app (app/gpu_pack.py) and the build's finishing step
(scripts/gpu_pack_tools.py). No torch needed -- a fake package stands in for the pack's contents."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app import gpu_pack

ROOT = Path(__file__).resolve().parent.parent


def _load_tools():
    spec = importlib.util.spec_from_file_location("gpu_pack_tools", ROOT / "scripts" / "gpu_pack_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_pack(folder: Path, python: str | None = None, module: str = "fake_gpu_pack_module") -> Path:
    site = folder / "site-packages"
    (site / module).mkdir(parents=True)
    (site / module / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
    (folder / "gpu-pack.json").write_text(json.dumps({"format": 1, "python": python or gpu_pack.python_tag(), "torch": "2.14.0"}), encoding="utf-8")
    return folder


@pytest.fixture
def clean_pack_state(monkeypatch):
    gpu_pack._reset_for_tests()
    saved_path = list(sys.path)
    yield monkeypatch
    sys.path[:] = saved_path
    for name in [name for name in sys.modules if name.startswith("fake_gpu_pack")]:
        del sys.modules[name]
    gpu_pack._reset_for_tests()


def test_a_pack_next_to_the_app_is_put_on_sys_path(tmp_path, clean_pack_state):
    pack = _make_pack(tmp_path / "gpu-pack")
    clean_pack_state.setenv(gpu_pack.ENV_VAR, str(pack))

    assert gpu_pack.activate_gpu_pack() == pack
    assert sys.path[-1] == str(pack / "site-packages")  # after the base: the base's packages win
    assert importlib.import_module("fake_gpu_pack_module").VALUE == 42
    status = gpu_pack.gpu_pack_status()
    assert status["active"] and status["torch"] == "2.14.0" and status["problem"] is None
    assert gpu_pack.activate_gpu_pack() == pack and sys.path.count(str(pack / "site-packages")) == 1


def test_a_pack_for_another_python_is_refused_with_a_reason(tmp_path, clean_pack_state):
    pack = _make_pack(tmp_path / "gpu-pack", python="2.7", module="fake_gpu_pack_old")
    clean_pack_state.setenv(gpu_pack.ENV_VAR, str(pack))

    assert gpu_pack.activate_gpu_pack() is None
    assert str(pack / "site-packages") not in sys.path
    assert "Python 2.7" in gpu_pack.gpu_pack_status()["problem"]


def test_the_frozen_app_looks_next_to_its_executable(tmp_path, clean_pack_state):
    pack = _make_pack(tmp_path / "gpu-pack", module="fake_gpu_pack_frozen")
    clean_pack_state.delenv(gpu_pack.ENV_VAR, raising=False)
    clean_pack_state.setattr(sys, "frozen", True, raising=False)
    clean_pack_state.setattr(sys, "executable", str(tmp_path / "PhotoStudio.exe"))

    assert pack in gpu_pack.candidate_dirs()
    assert gpu_pack.activate_gpu_pack() == pack


def test_without_a_pack_nothing_changes(tmp_path, clean_pack_state):
    clean_pack_state.setenv(gpu_pack.ENV_VAR, str(tmp_path / "missing"))
    before = list(sys.path)
    assert gpu_pack.activate_gpu_pack() is None
    assert sys.path == before and gpu_pack.gpu_pack_status() == {"active": False, "path": None, "torch": None, "problem": None}


def test_the_install_hint_names_the_gpu_pack_in_a_packaged_build(clean_pack_state):
    from app.ai_upscale import ai_upscale_install_hint

    assert "requirements-gpu.txt" in ai_upscale_install_hint()
    clean_pack_state.setattr(sys, "frozen", True, raising=False)
    assert "GPU pack" in ai_upscale_install_hint()


def test_the_self_test_reports_the_gpu_pack(tmp_path, clean_pack_state):
    from app.desktop import self_test_report

    clean_pack_state.setenv(gpu_pack.ENV_VAR, str(_make_pack(tmp_path / "gpu-pack", module="fake_gpu_pack_selftest")))
    report = self_test_report()
    assert report["capabilities"]["gpu_pack"]["active"] is True


def _fake_distribution(site: Path, name: str, version: str, files: dict[str, str]) -> None:
    dist_info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    records = []
    for relative, content in files.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        records.append(f"{relative},,")
    records += [f"{dist_info.name}/METADATA,,", f"{dist_info.name}/RECORD,,", "../../Scripts/evil.exe,,"]
    (dist_info / "RECORD").write_text("\n".join(records) + "\n", encoding="utf-8")


def test_finalize_prunes_what_the_base_ships_and_writes_the_manifest(tmp_path):
    tools = _load_tools()
    pack = tmp_path / "gpu-pack"
    site = pack / "site-packages"
    _fake_distribution(site, "torch", "2.14.0", {"torch/__init__.py": "", "torch/lib/c10.dll": "x"})
    _fake_distribution(site, "numpy", "2.5.2", {"numpy/__init__.py": "", "numpy/core/x.py": ""})
    _fake_distribution(site, "opencv-python", "5.0.0.93", {"cv2/__init__.py": ""})
    outside = tmp_path / "Scripts" / "evil.exe"
    outside.parent.mkdir()
    outside.write_text("keep me", encoding="utf-8")
    constraints = tmp_path / "constraints-cpu.txt"
    constraints.write_text("# base\nnumpy==2.5.2\nPillow==12.3.0\n-c other.txt\n", encoding="utf-8")

    manifest = tools.finalize(pack, constraints, "9.9.9")

    assert not (site / "numpy").exists() and not (site / "cv2").exists()
    assert (site / "torch" / "lib" / "c10.dll").exists()
    assert outside.exists()  # a RECORD entry pointing outside the pack is never followed
    assert manifest["torch"] == "2.14.0" and manifest["python"] == gpu_pack.python_tag() and manifest["format"] == gpu_pack.PACK_FORMAT
    assert json.loads((pack / "gpu-pack.json").read_text(encoding="utf-8"))["packages"] == {"torch": "2.14.0"}
    assert gpu_pack.check_pack(pack) is None  # what the build writes is what the app accepts


def test_the_build_script_and_installer_know_the_gpu_pack():
    script = (ROOT / "scripts" / "build_release.ps1").read_text(encoding="utf-8")
    assert "[switch]$GpuPack" in script and "gpu_pack_tools.py" in script and "PHOTO_STUDIO_GPU_PACK" in script
    installer = (ROOT / "installer" / "PhotoStudio.iss").read_text(encoding="utf-8")
    assert 'DestDir: "{app}\\gpu-pack"' in installer and "Components: gpupack" in installer
    pack_requirements = (ROOT / "requirements-gpu-pack.txt").read_text(encoding="utf-8")
    assert "torch" in pack_requirements and "-r requirements.txt" not in pack_requirements
    assert "-r requirements-gpu-pack.txt" in (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
