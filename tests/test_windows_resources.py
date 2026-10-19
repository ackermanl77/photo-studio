"""The packaged exe's Windows resources (K6/K7): version resource from app.__version__, the manifest,
the .photoproject document icon and the numba/llvmlite exclusion. No PyInstaller needed."""
import importlib.util
import xml.dom.minidom
from pathlib import Path

from PIL import Image

import app

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("windows_resources", ROOT / "scripts" / "windows_resources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluate_version_file(text: str) -> dict:
    """Evaluate the VSVersionInfo text with stand-ins that record the string table."""
    found = {}

    def record(name):
        def make(*args, **kwargs):
            if name == "StringStruct":
                found[args[0]] = args[1]
            if name == "FixedFileInfo":
                found["filevers"] = kwargs["filevers"]
            return (name, args, kwargs)
        return make

    names = ["VSVersionInfo", "FixedFileInfo", "StringFileInfo", "StringTable", "StringStruct", "VarFileInfo", "VarStruct"]
    eval(compile(text, "version", "eval"), {name: record(name) for name in names})
    return found


def test_version_resource_comes_from_app_version(tmp_path):
    resources = _load()
    assert resources.app_version() == app.__version__
    path = resources.write_version_file(str(tmp_path))
    found = _evaluate_version_file(Path(path).read_text(encoding="utf-8"))
    expected = resources.version_tuple(app.__version__)
    assert found["filevers"] == expected
    assert found["FileVersion"] == found["ProductVersion"] == ".".join(map(str, expected))
    assert found["CompanyName"] == "MELEJ"
    assert found["OriginalFilename"] == "PhotoStudio.exe"
    assert resources.version_tuple("1.2") == (1, 2, 0, 0)
    assert resources.version_tuple("0.3.0rc1") == (0, 3, 0, 1)


def test_manifest_declares_utf8_code_page_and_per_monitor_v2(tmp_path):
    resources = _load()
    dom = xml.dom.minidom.parse(resources.write_manifest(str(tmp_path)))
    root = dom.documentElement
    assert root.tagName == "assembly" and root.getAttribute("manifestVersion") == "1.0"

    def text(tag):
        (node,) = dom.getElementsByTagName(tag)
        return node.firstChild.data

    assert text("activeCodePage") == "UTF-8"
    assert text("dpiAwareness") == "PerMonitorV2"
    assert text("longPathAware") == "true"
    assert dom.getElementsByTagName("requestedExecutionLevel")[0].getAttribute("level") == "asInvoker"
    assert len(dom.getElementsByTagName("supportedOS")) == 5


def test_spec_embeds_version_manifest_and_document_icon():
    spec = (ROOT / "PhotoStudio.spec").read_text(encoding="utf-8")
    assert "version=version_path" in spec and "manifest=manifest_path" in spec
    assert 'icon=["app/assets/photostudio.ico", "app/assets/photoproject.ico"]' in spec
    for name in ("numba", "llvmlite", "facexlib"):
        assert f'"{name}"' in spec
    iss = (ROOT / "installer" / "PhotoStudio.iss").read_text(encoding="utf-8")
    assert 'PhotoStudio.exe,1"; Tasks: assocproject' in iss  # the second icon group = the document icon
    assert '#define MyAppPublisher "MELEJ"' in iss


def test_document_icon_has_every_explorer_size():
    with Image.open(ROOT / "app" / "assets" / "photoproject.ico") as icon:
        sizes = {size[0] for size in icon.info["sizes"]}
    assert {16, 20, 24, 32, 48, 256} <= sizes


def test_gpu_pack_prunes_numba_and_llvmlite():
    spec = importlib.util.spec_from_file_location("gpu_pack_tools", ROOT / "scripts" / "gpu_pack_tools.py")
    tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools)
    assert {"numba", "llvmlite", "facexlib", "gfpgan"} <= tools.ALWAYS_PRUNE
