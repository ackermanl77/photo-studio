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
datas += [("app", "app")]

# Optional AI Upscale deps (requirements-gpu.txt). Only bundled when the interpreter running
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PhotoStudio",
)
