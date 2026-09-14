# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules

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
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PhotoStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
