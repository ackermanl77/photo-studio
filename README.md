# Photo Professional Tool

A CPU-first Python photo editor with a Streamlit interface, Windows desktop packaging, RAW import, layers, masks, presets, catalog tools, batch export, project sessions, and diagnostics.

## Current Status

- Global Python: `C:/Users/MELEJ/AppData/Local/Python/pythoncore-3.14-64/python.exe`.
- Current packaged release: `PhotoStudio-0.1.0-Windows/`.
- Designed for computers without a dedicated GPU.
- `docs/PROJECT_STATUS.md` is the authoritative reference for entry points, architecture, and known issues — this README only covers day-to-day commands.

## Install

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pip install -r requirements.txt
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pip install -r requirements-raw.txt -r requirements-desktop.txt
```

The `.venv` folder is optional.

## Recommended App

```powershell
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app/frontend_unified_app.py
```

`app/frontend_unified_app.py` is currently the **only** Streamlit entry point in the project. Earlier "specialized workspace" scripts (`frontend_project.py`, `frontend_layers.py`, `frontend_masking.py`, `frontend_batch.py`, `frontend_library_editor.py`, `frontend_portrait.py`, `frontend_release.py`, and others) were consolidated into it during development and no longer exist. If you find them mentioned elsewhere in `docs/workflows/`, that's a historical development note, not a current instruction — see the notice at the top of `docs/workflows/README.md`.

The unified workspace includes Tone, Color, Detail, Layers, Masks, Portrait, and Presets tabs, plus a "More" tab for Library, Batch, and Diagnostics. The layout keeps the preview central with the tool tabs on the left.

## Desktop App

Current packaged app:

```text
PhotoStudio-0.1.0-Windows\PhotoStudio.exe
```

Rebuild:

```powershell
.\scripts\build_release.ps1 -Version "0.1.0"
```

The release script prefers global Python, supports one-file and one-folder PyInstaller output, optional signing, 7-Zip compression, and SHA-256 checksums. Microsoft Edge WebView2 Runtime is required for the native window.

`app/desktop.py` starts both the Streamlit frontend (`frontend_unified_app.py`) and the Flask processing backend (`app/backend.py`, `127.0.0.1:8502`) — the frontend sends its edit pipeline to the backend over HTTP rather than processing in-process. Running the frontend standalone (`streamlit run app/frontend_unified_app.py`, below) also works without `desktop.py`: the frontend starts the backend on demand the first time it's needed. See "Architecture" in `docs/PROJECT_STATUS.md` for the request flow and the latency trade-off that comes with it.

There is currently no dedicated preflight UI. See [docs/workflows/secure-distribution/preflight.md](docs/workflows/secure-distribution/preflight.md) for how to run the preflight checks that exist today before building a release.

## Core Features

- Exposure, contrast, highlights, shadows, temperature, tint, texture, clarity, curves, sharpening, denoise, and a one-click "✨ Auto Enhance" pass, all live in the unified UI. Free-form or named-aspect-ratio crop, positioned right after Tone.
- Object removal / repair: select a blemish or object with the same Luminance/Color/gradient/Radial selection tools as Masks, then it's inpainted away.
- Built-in presets with live thumbnail previews, plus custom presets you can save, and export/import as a shareable `.json` file.
- Undo/Redo and a one-click Reset to defaults, always available in the bar above the canvas regardless of which tab is open.
- Combinable named masks (`app/advanced_masks.py::MaskStack`) — any number of Luminance/Color/gradient/Radial selections, each with opacity/invert/soften, unioned into one target adjustment.
- Layers with opacity, blending (normal/multiply/screen), preserved through Save/Load Project.
- Camera RAW import (`.cr2/.cr3/.nef/.arw/.dng`, optional via `requirements-raw.txt`) directly from the unified UI's uploader.
- A draggable before/after comparison ("Split View"), plus a live per-channel histogram.
- ICC profile preservation and tested 16-bit TIFF export support.
- Local photo library with scanning, thumbnails, ratings, tags, and search.
- Batch profiles, filename templates, error reports, and ZIP delivery.
- Save/Load Project (`.photoproject`, `app/project_bundle.py`) persisting the source image, recipe, layer, and mask selections.
- Preview caching, diagnostics, request IDs, image validation, and structured audit logs.

See `docs/PROJECT_STATUS.md`'s "Known Issues" for the current gaps (e.g. free-form crop not yet in Undo/Redo, no draggable rectangle on the canvas for it).

## Test

```powershell
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m compileall -q app
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest -q
```

Run this yourself before trusting a pass count — different historical snapshots in `docs/workflows/` report different totals as the suite grew.

See [docs/workflows/README.md](docs/workflows/README.md) and [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).
