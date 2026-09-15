# Photo Professional Tool

[![CI](https://github.com/ackermanl77/photo-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/ackermanl77/photo-studio/actions/workflows/ci.yml)

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
PhotoStudio-0.1.0-Windows\WebView2Runtime\   <- bundled, see below
```

Rebuild:

```powershell
.\scripts\build_release.ps1 -Version "0.1.0"
```

The release script prefers global Python, supports one-file and one-folder PyInstaller output, optional signing, 7-Zip compression, and SHA-256 checksums.

**The native window needs Microsoft Edge WebView2 Runtime, and it's bundled — nothing to install on the target machine.** Confirmed missing on a second PC when the app was first tested there, so `build_release.ps1` now downloads a pinned "Fixed Version" copy of the WebView2 Runtime (cached under `vendor/webview2-cache/`, gitignored — first build needs internet access for this one-time ~300MB download) and copies it into the release folder as `WebView2Runtime`, a sibling of `PhotoStudio.exe`. `app/desktop.py` points pywebview at it via `webview.settings["WEBVIEW2_RUNTIME_PATH"]`, so the packaged app never depends on WebView2 being installed system-wide. Copy the whole `PhotoStudio-<version>-Windows` folder (or its `.7z`) to another machine — extract and run, no installer, no Python. The one trade-off: a bundled Fixed Version copy doesn't auto-update with Chromium security patches the way a system install does, so `$WebView2Version` in `build_release.ps1` should be bumped deliberately every so often (see the comment above it for how to get a new version's download URL).

`app/desktop.py` starts both the Streamlit frontend (`frontend_unified_app.py`) and the Flask processing backend (`app/backend.py`, `127.0.0.1:8502`) — the frontend sends its edit pipeline to the backend over HTTP rather than processing in-process. Running the frontend standalone (`streamlit run app/frontend_unified_app.py`, below) also works without `desktop.py`: the frontend starts the backend on demand the first time it's needed. See "Architecture" in `docs/PROJECT_STATUS.md` for the request flow and the latency trade-off that comes with it.

There is currently no dedicated preflight UI. See [docs/workflows/secure-distribution/preflight.md](docs/workflows/secure-distribution/preflight.md) for how to run the preflight checks that exist today before building a release.

## Core Features

- Exposure, contrast, highlights, shadows, temperature, tint, texture, clarity, sharpening, and a one-click "✨ Auto Enhance" pass, all live in the unified UI. Free-form or named-aspect-ratio crop, positioned right after Tone.
- Denoise via OpenCV's Non-Local Means — a real upgrade over a flat median filter (preserves edges/detail while smoothing noise), classical rather than a neural network (see `docs/PROJECT_STATUS.md` for why, given this app's CPU-first, no-GPU design).
- A draggable RGB tone curve (drag points on the curve itself, or use the numeric sliders below it for exact values).
- Per-channel HSL color grading ("Color Mixer") — independent Hue/Saturation/Luminance adjustment for each of 8 named color bands (Red/Orange/Yellow/Green/Aqua/Blue/Purple/Magenta).
- Color Grading (Split Toning) — independent Hue/Intensity tinting for Shadows, Midtones, and Highlights, plus a Balance slider, the same three-range panel every major editor offers.
- Dehaze — classical Dark Channel Prior atmospheric haze reduction, not a neural network (see `docs/PROJECT_STATUS.md` for the same CPU-first reasoning as Denoise, and for its known guided-filter/halo limitation).
- Lens corrections (distortion, vignette, chromatic aberration) — not a calibrated per-lens-model profile database, just typical starting points by lens category (Wide-angle, Standard zoom, Telephoto, Fisheye) plus manual sliders to fine-tune.
- Object removal / repair: select a blemish or object with the same Luminance/Color/gradient/Radial/Painted/Magic Wand selection tools as Masks, then it's inpainted away. Magic Wand selects by clicking a point on the photo — contiguous (flood-fill) by default, or every matching-color pixel anywhere in the frame.
- A "⚠️ Show clipping" toggle next to the histogram highlights blocked shadows and blown highlights directly on the canvas, the same idea as Lightroom's clipping triangles.
- Built-in presets with live thumbnail previews, plus custom presets you can save, load back onto a photo, delete, and export/import as a shareable `.json` file. Saving captures the full set of adjustments (masks included), not just a handful of basic sliders.
- Undo/Redo and a one-click Reset to defaults, always available in the bar above the canvas regardless of which tab is open.
- Combinable named masks (`app/advanced_masks.py::MaskStack`) — any number of Luminance/Color/gradient/Radial/Painted/Magic Wand selections, each with opacity/invert/soften, unioned into one target adjustment. "Painted" selections are made on-canvas — freehand brush (with an eraser), a drag-out radial, or a drag-out gradient — instead of typing in parameters; "Magic Wand" is a single click that samples a color and selects similar pixels around it.
- Layers with opacity, blending (normal/multiply/screen), preserved through Save/Load Project.
- Camera RAW import (`.cr2/.cr3/.nef/.arw/.dng`, optional via `requirements-raw.txt`) directly from the unified UI's uploader.
- A draggable before/after comparison ("Split View"), plus a live per-channel histogram.
- Real ICC color management (via littlecms/`PIL.ImageCms`): an embedded non-sRGB profile (Adobe RGB, ProPhoto RGB, a camera's own profile) is actually converted into the app's sRGB working space, not just tagged-and-preserved, plus a "Soft Proof: Gamut Warning" preview highlighting pixels that get clipped in that conversion. Not a simulation of an external printer/press profile — no profile database is bundled. Tested 16-bit TIFF export support.
- Local photo library with scanning, thumbnails, search, and a Rate & Flag panel (star ratings, reject, tags) with keyboard shortcuts (1-5 rate, 0 clears, X rejects, arrows move Prev/Next).
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

GitHub Actions (`.github/workflows/ci.yml`) runs the same `compileall` + `pytest` on every push and pull request, installing only `requirements.txt` — `requirements-raw.txt` (rawpy) and `requirements-desktop.txt` (pywebview/PyInstaller) are for the local dev/build workflow, not the test suite, and installing pywebview on a plain Linux runner risks needing system GTK/WebKit packages the runner doesn't have, for zero test coverage benefit.

See [docs/workflows/README.md](docs/workflows/README.md) and [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).
