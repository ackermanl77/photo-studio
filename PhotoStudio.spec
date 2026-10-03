# -*- mode: python ; coding: utf-8 -*-

import importlib.util

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas = []
binaries = []
# app.frontend was renamed to app.frontend_unified_app when the specialized Streamlit frontends
# were consolidated into one; streamlit_drawable_canvas was removed as an unused dependency (see
# docs/PROJECT_STATUS.md) -- it was never imported anywhere under app/.
hiddenimports = ["app.backend", "app.frontend_unified_app"]

for package in ["streamlit", "pywebview"]:
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

hiddenimports += collect_submodules("app")

# The optional GPU pack (K5, app/gpu_pack.py) is imported from a folder outside this bundle, so the
# frozen base must already carry every standard-library module torch & co. might import -- the
# app's own analysis only bundles the ones the app uses. Bundle the whole stdlib (a few MB) except
# GUI/test/developer-only parts.
import sys

_STDLIB_SKIP = {"tkinter", "turtle", "turtledemo", "idlelib", "test", "ensurepip", "lib2to3", "pydoc_data", "venv", "__phello__", "antigravity", "this"}
for _name in sorted(sys.stdlib_module_names):
    if _name in _STDLIB_SKIP:
        continue
    try:
        _spec = importlib.util.find_spec(_name)
    except (ImportError, ValueError):
        continue
    if _spec is None:
        continue
    if _spec.submodule_search_locations:
        hiddenimports += collect_submodules(_name, filter=lambda module: ".test" not in module and ".idle_test" not in module)
    else:
        hiddenimports.append(_name)
# ("app", "app") also carries app/assets/ (the sample photo and the YuNet face-detection model
# app/portrait_tools.py loads). The Streamlit config is bundled for standalone runs from the
# release folder; the desktop launcher passes the same settings as command-line flags anyway
# (app/desktop.py::_streamlit_server_flags), since its children run in the per-user data folder.
datas += [("app", "app"), (".streamlit/config.toml", ".streamlit")]

# Optional AI Upscale deps (requirements-gpu.txt) -- the all-in-one gpu profile only; the default
# release ships them as the separate GPU pack instead. Only bundled when the interpreter running
# PyInstaller actually has them -- a plain requirements.txt build is unchanged. basicsr and
# realesrgan both build their architecture/dataset registries by scandir()-ing their own package
# folders for *_arch.py/*_dataset.py files at import time, so their .py sources must exist on disk
# in the frozen app (include_py_files=True), not only as compiled modules in the PYZ archive --
# otherwise `import realesrgan` raises FileNotFoundError in the packaged build.
for package in ["basicsr", "realesrgan"]:
    if importlib.util.find_spec(package) is not None:
        datas += collect_data_files(package, include_py_files=True)
        hiddenimports += collect_submodules(package)

# torchvision (a basicsr dependency) loads its native ops library (_C*.pyd -- e.g. _C_stable.pyd)
# via torch.ops.load_library() by file path at import time, which PyInstaller's import scanner
# can't see -- so without this the frozen app fails `import torchvision` with
# "RuntimeError: operator torchvision::nms does not exist" (confirmed in a packaged build), and AI
# Upscale reports itself unavailable. Bundle its .pyd/.dll files into torchvision/ explicitly.
_torchvision_spec = importlib.util.find_spec("torchvision")
if _torchvision_spec is not None:
    import glob
    import os

    _torchvision_dir = os.path.dirname(_torchvision_spec.origin)
    for _lib in glob.glob(os.path.join(_torchvision_dir, "*.pyd")) + glob.glob(os.path.join(_torchvision_dir, "*.dll")):
        binaries.append((_lib, "torchvision"))


a = Analysis(
    ["app/desktop.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)
# One-folder build (exe + _internal\), not one-file: with the optional CUDA torch stack the bundle
# is several GB, which a one-file exe would re-extract to a temp folder on every launch.
# scripts/build_release.ps1 already copies either layout into the release folder.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PhotoStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="app/assets/photostudio.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PhotoStudio",
)
