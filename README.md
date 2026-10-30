# Photo Professional Tool

[![CI](https://github.com/ackermanl77/photo-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/ackermanl77/photo-studio/actions/workflows/ci.yml)

A CPU-first Python photo editor with a Streamlit interface, Windows desktop packaging, RAW import, layers, masks, presets, catalog tools, batch export, project sessions, and diagnostics. An optional CUDA path can accelerate final export resizing and sharpening; the edit pipeline and file encoders remain CPU-based.

## Current Status

- Python 3.12-3.14 (the desktop build and CI use 3.14); the commands below assume a virtual environment.
- A release build lands in `release\PhotoStudio.exe` (distributed as `PhotoStudio-<version>-Windows.7z`, optionally with an installer and a separate GPU pack -- see below).
- Designed for computers without a dedicated GPU.
- `docs/PROJECT_STATUS.md` is the authoritative reference for entry points, architecture, and known issues — this README only covers day-to-day commands.
- New to the app itself? [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) walks through every tab and feature in the unified workspace.

## Install

From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt          # app + pytest
python -m pip install -r requirements-raw.txt -r requirements-desktop.txt   # optional: RAW import, desktop build
```

`requirements.txt` carries upper bounds; `constraints-cpu.txt` / `constraints-gpu.txt` pin the exact
versions the release build uses (`-c constraints-cpu.txt`).

## Recommended App

```powershell
python -m streamlit run app/frontend_unified_app.py
```

`app/frontend_unified_app.py` is currently the **only** Streamlit entry point in the project. Earlier "specialized workspace" scripts (`frontend_project.py`, `frontend_layers.py`, `frontend_masking.py`, `frontend_batch.py`, `frontend_library_editor.py`, `frontend_portrait.py`, `frontend_release.py`, and others) were consolidated into it during development and no longer exist. Mentions of them in `docs/CHANGELOG.md` are history, not current instructions.

The unified workspace groups its tools by workflow in nine tabs -- ☼ Basic, 🎨 Curves & Color, 🔬 Detail, ⛶ Geometry, ◉ Local (Masks, Repair, Portrait), ◈ Layers, ✨ Effects, ⬇ Export and ✦ Presets & Library (sub-tabs ✦ Presets and ☰ Library & Batch, which holds the Library Catalog, the Batch Processor and System Health) -- with a "● modified" mark on every section you have changed. Like Photoshop, the window belongs to the photo: the tool tabs live in a sidebar on the left (drag its edge to widen it, or collapse it for the whole window), and the photo, its view controls and the Save button fill the page.

## Desktop App

**The built app always lands in `release\PhotoStudio.exe`** — a fixed-name folder `build_release.ps1` overwrites with the latest build on every run, so you don't need to know or look up the current version number to find it. It's the only unpacked copy: PyInstaller's output is moved (not copied) into `release\`, and `build-release\` / `dist-release\` are deleted after a successful build (pass `-KeepBuildFiles` to keep PyInstaller's cache for a faster rebuild; a failed build always keeps them for debugging). The distributable is `PhotoStudio-<version>-Windows.7z` + `.sha256`, built from `release\` — `-SkipArchive` skips it and removes a stale one from an earlier build.

```text
release\PhotoStudio.exe
release\WebView2Runtime\   <- bundled, see below
```

Rebuild:

```powershell
.\scripts\build_release.ps1 -Version "0.1.0"
```

The release script builds from a fresh virtual environment (`build-venv\<profile>`, pinned by `constraints-<profile>.txt`, `-BuildProfile cpu|gpu`), verifies the bundled WebView2 runtime's SHA-256, runs `PhotoStudio.exe --self-test`, and supports optional signing (`-Sign -CertificateThumbprint <sha1>`, or `-Sign` with the thumbprint in `CODESIGN_THUMBPRINT`: signs `PhotoStudio.exe` and, with `-Installer`, the installer and its uninstaller, timestamped by `CODESIGN_TIMESTAMP_URL`, default DigiCert), 7-Zip compression, SHA-256 checksums, and the Inno Setup installer (`-Installer`, see below).

**CPU base + optional GPU pack.** The default release is the CPU base package (no PyTorch).
`-GpuPack` (with `-ExtraIndexUrl https://download.pytorch.org/whl/cu130`) also builds the
optional **GPU pack** for AI Upscale and export transforms (`release-gpu-pack\gpu-pack\`,
archived as `PhotoStudio-<version>-GPU-pack.7z`), containing torch/Real-ESRGAN and their
dependencies beyond the base; the build also re-runs the base exe's self-test against it. Users
extract it next to `PhotoStudio.exe` and restart; the installer offers it as an optional component
when it was built. In the Export tab, **Use GPU for export
resize/sharpen** opts into CUDA with automatic CPU fallback. The CLI equivalent is
`python -m app --input photo.jpg --output out.jpg --gpu-export`. Image encoding and the rest of the
edit pipeline remain on the CPU. `-BuildProfile gpu` still produces the older all-in-one folder.

**Installer (optional).** `-Installer` also compiles `installer\PhotoStudio.iss` with Inno Setup 6 into `installer-output\PhotoStudio-Setup-<version>.exe`. Setup lets the user install for all users (admin, Program Files) or just for themselves (no admin rights, `%LOCALAPPDATA%\Programs`; `/CURRENTUSER` on the command line), optionally associates `.photoproject` files, and offers the GPU pack as an optional component when it was built. Uninstalling asks whether to also delete the per-user data in `%LOCALAPPDATA%\PhotoStudio` (a silent uninstall keeps it).

**Or let CI build it for you**: pushing a tag like `v0.1.0` triggers `.github/workflows/release.yml`, which first runs the full test matrix and then runs this exact same script on a `windows-latest` runner and attaches `PhotoStudio-<version>-Windows.7z` + its `.sha256` to a new GitHub Release, plus keeps them as a build artifact on the workflow run either way. It can also be run on demand (Actions tab → "Release Build" → "Run workflow", enter a version; tick `gpu_pack` to also build the GPU pack as an artifact) without needing a tag push or a real release at all, which just builds and uploads the artifact.

**The native window needs Microsoft Edge WebView2 Runtime, and it's bundled — nothing to install on the target machine.** Confirmed missing on a second PC when the app was first tested there, so `build_release.ps1` now downloads a pinned "Fixed Version" copy of the WebView2 Runtime (cached under `vendor/webview2-cache/`, gitignored — first build needs internet access for this one-time ~300MB download) and copies it into the release folder as `WebView2Runtime`, a sibling of `PhotoStudio.exe`. `app/desktop.py` points pywebview at it via `webview.settings["WEBVIEW2_RUNTIME_PATH"]`, so the packaged app never depends on WebView2 being installed system-wide. Copy the whole `release` folder (or the `PhotoStudio-<version>-Windows.7z`) to another machine — extract and run, no Python needed (or use the optional installer above). The one trade-off: a bundled Fixed Version copy doesn't auto-update with Chromium security patches the way a system install does, so `$WebView2Version` in `build_release.ps1` should be bumped deliberately every so often (see the comment above it for how to get a new version's download URL).

`app/desktop.py` starts both the Streamlit frontend (`frontend_unified_app.py`) and the Flask processing backend (`app/backend.py`, on free local ports chosen per launch, with a per-launch access token) — the frontend sends its edit pipeline to the backend over HTTP rather than processing in-process. Running the frontend standalone (`streamlit run app/frontend_unified_app.py`, below) also works without `desktop.py`: the frontend starts the backend on demand the first time it's needed. See "Architecture" in `docs/PROJECT_STATUS.md` for the request flow and the latency trade-off that comes with it.

The release script is its own preflight: it builds from a clean, pinned venv and refuses to finish unless the packaged `PhotoStudio.exe --self-test` passes; the release workflow additionally runs the whole CI test matrix first.

## Core Features

- Exposure in EV (linear light, soft highlight roll-off), contrast, edge-aware highlights/shadows, temperature, tint, texture, local-contrast clarity, sharpening, and an image-adaptive "✨ Auto Enhance", all live in the unified UI. Rotate 90°, flip, straighten with auto-level and "constrain crop", plus a draggable free-form crop rectangle with numeric fields, a custom W:H ratio, or a named one.
- Denoise via OpenCV's Non-Local Means — a real upgrade over a flat median filter (preserves edges/detail while smoothing noise), classical rather than a neural network (see `docs/PROJECT_STATUS.md` for why, given this app's CPU-first edit pipeline). An independent "Color Noise" slider pushes chroma (color speckle) smoothing further than luminance/detail smoothing, without needing a second algorithm.
- A draggable RGB tone curve — drag any of the 5 fixed Blacks/Shadows/Midtones/Highlights/Whites points vertically, or click an empty spot on the curve to add a fully free-form point (drag it anywhere, double-click to remove), or use the numeric sliders below it for exact values on the 5 fixed points.
- Per-channel HSL color grading ("Color Mixer") — independent Hue/Saturation/Luminance adjustment for each of 8 named color bands (Red/Orange/Yellow/Green/Aqua/Blue/Purple/Magenta).
- Vibrance — a smarter saturation boost than the plain Saturation slider: protects skin tones and already-saturated colors, pushing washed-out/pastel ones harder instead.
- A "🎯 Pick White Balance" eyedropper — click a spot in the photo that's supposed to be neutral gray or white, and Temperature/Tint are set to whatever would neutralize it. A starting point to fine-tune from, the same convenience darktable's color-calibration picker offers.
- Color Grading (Split Toning) — independent Hue/Intensity tinting for Shadows, Midtones, and Highlights, plus a Balance slider, the same three-range panel every major editor offers.
- Dehaze — classical Dark Channel Prior atmospheric haze reduction, not a neural network (see `docs/PROJECT_STATUS.md` for the same CPU-first reasoning as Denoise, and for its known guided-filter/halo limitation).
- Lens corrections (distortion, vignette, chromatic aberration, perspective/keystone) — "From the photo's lens" looks the EXIF lens up in the lensfun database and fits the sliders to its calibration; otherwise typical starting points by lens category (Wide-angle, Standard zoom, Telephoto, Fisheye), plus manual sliders to fine-tune. Perspective (Vertical/Horizontal Keystone) corrects converging lines from shooting angle rather than lens hardware, so it isn't part of a lens profile.
- Object removal / repair: select a blemish or object with the same Luminance/Color/gradient/Radial/Painted/Magic Wand selection tools as Masks, then it's inpainted away. Magic Wand selects by clicking a point on the photo — contiguous (flood-fill) by default, or every matching-color pixel anywhere in the frame.
- A "⚠️ Show clipping" toggle next to the histogram highlights blocked shadows and blown highlights directly on the canvas, the same idea as Lightroom's clipping triangles.
- Built-in presets with live thumbnail previews, plus custom presets you can save, load back onto a photo, delete, and export/import as a shareable `.json` file — a custom preset also gets its own live thumbnail preview (in "Load a Preset" and in the Batch Processor's preset picker), rendered by running its full recipe through the real pipeline on a small copy of the current photo. Saving captures the full set of adjustments (masks included), not just a handful of basic sliders.
- Undo/Redo and a Reset to defaults (in the History menu), always available in the one-row bar above the canvas regardless of which tab is open, plus per-section "↺ Reset" buttons. Keyboard: Ctrl+Z / Ctrl+Shift+Z (undo/redo), Ctrl+E (save), Z (1:1 view), B (before/after), G (grid).
- Combinable named masks (`app/advanced_masks.py::MaskStack`) — any number of Luminance/Color/gradient/Radial/Painted/Magic Wand selections, each with opacity/invert/soften, combined by add/subtract/intersect into one target adjustment — or each with its own adjustment set (exposure, contrast, saturation, clarity, temperature, tint). "Painted" selections are made on-canvas — freehand brush (with an eraser), a drag-out radial, or a drag-out gradient — instead of typing in parameters, and can be reopened later for a second pass at the canvas ("🎨 Edit Painting") instead of only delete-and-repaint; "Magic Wand" is a single click that samples a color and selects similar pixels around it.
- Creative "✨ Effects" — a post-crop vignette (applied to the final, already-cropped frame, separate from the Lens corrections' corrective one), film grain, and Creative Filters (lens blur with tilt-shift, motion blur, bloom, halftone, glitch).
- 💡 Relight — one movable light plus ambient and warmth over a classical shape-from-shading normal map (`app/relight.py`), and Background — blur, remove (transparent PNG/TIFF) or replace with a colour, from the AI subject mask or an OpenCV GrabCut cut-out (`app/background.py`).
- Intensity (0-100) for Preset Looks and Auto Enhance, and `.cube` 3D/1D LUTs (`app/lut.py`) with their own library.
- Liquify (push, bloat, pinch, twirl) drawn on the canvas, as part of the geometry step (`app/liquify.py`).
- Optional AI masks — Subject and Person through the GPU pack, a classical Sky mask without it — added as Painted selections; AI Denoise (SCUNet, GPU pack; the classical Denoise without it); and AI Upscale (Real-ESRGAN, GPU pack). Exports an AI tool contributed to are marked with IPTC Digital Source Type and an XMP edit history.
- Any number of stacked layers, each with its own opacity, visibility, and blend mode — 26 Photoshop-matching modes (Multiply/Screen/Overlay/Soft Light/Hard Light/Color Dodge/Color Burn/Difference/Hue/Saturation/Color/Luminosity, and more), reorderable and duplicable, preserved through Save/Load Project. A layer can be an uploaded image, a Text layer (content, font size, color, alignment, position), a vector Shape layer (Rectangle/Ellipse/Line/Polygon/Star, with its own fill and outline), or a Pen tool path — click to place anchor points on the photo, drag a point to move it, drag its handle to curve the path through it, double-click to remove a point; open paths can be stroked, closed ones filled and/or stroked. Image layers keep their proportions and can be moved, scaled and rotated; Text/Shape/Path layers stay editable, including after saving and reopening a `.photoproject` file. Adjustment layers apply exposure/contrast/saturation/temperature/tint/clarity to everything below them; shapes take linear or radial gradient fills; lines and open paths can carry arrow heads or be a translucent highlighter; layers can be aligned and distributed, and the vector layers exported as SVG (`app/svg_export.py`).
- Layer Styles — Drop Shadow, Outer Glow, Stroke, and Color Overlay, any combination stackable on any layer (image, text, or shape), each independently computed from that layer's own silhouette so one effect's settings never bleed into another's shape.
- Guides & Grid — a Rule of Thirds / 4x4 / Golden Ratio grid overlay plus any number of custom horizontal/vertical guides, shown on the canvas as a pure composition aid (never sent through the backend or included in an export). The free-form Crop tool's corner handles snap to whichever grid lines/guides are active.
- Camera RAW import (`.cr2/.cr3/.nef/.arw/.dng/.raf/.rw2/.orf/.pef/.srw`, optional via `requirements-raw.txt`) in the editor, the Batch Processor and the Library.
- A true 1:1 zoom (plus 200 %/400 %) with a click-to-center navigator, a draggable before/after comparison ("Split View"), and a live per-channel histogram.
- Real ICC color management (via littlecms/`PIL.ImageCms`): an embedded non-sRGB profile (Adobe RGB, ProPhoto RGB, a camera's own profile) is actually converted into the app's sRGB working space, not just tagged-and-preserved, plus a "Soft Proof: Gamut Warning" preview highlighting pixels that get clipped in that conversion. Not a simulation of an external printer/press profile — no profile database is bundled. Genuine 16-bit PNG/TIFF export (Export settings → Bit depth, CLI `--bit-depth 16`), processed in floating point from the full-precision original (16-bit files, RAW). Layered PSD import (each Photoshop layer to the Layers stack, with masks, clipping and group visibility) and export (Background + one layer per layer), dependency-free (`app/psd.py`); a PSD also opens as a photo. Floating-point OpenEXR (half, ZIP) and PFM export in linear light from the same path (`app/float_export.py`, no new dependency).
- Local photo library with scanning (RAW included; a rescan keeps ratings), a cached thumbnail grid with "✏️ Edit" into the editor, search, and a Rate & Flag panel (star ratings, reject, tags) with keyboard shortcuts (1-5 rate, 0 clears, X rejects, arrows move Prev/Next — spelled out on screen under "Rate & Flag", not just in these docs). Scan Folder shows a live progress bar as it works, ending in a "Done — indexed N photo(s)" summary; an empty search/star filter says so instead of showing a silent blank table.
- Culling scores (focus, clipped highlights/shadows) in the catalog and a duplicate finder that keeps the sharpest copy; HDR merge and panorama stitching (`app/merge.py`, OpenCV) open the result as a 16-bit source.
- Batch profiles, filename templates, error reports, and ZIP delivery — every supported format, processed from memory and streamed into a ZIP on disk, with a Stop button. The Batch Processor can apply either a built-in named look or a saved custom preset's full recipe (tone/color/detail/lens — "sync settings" across a whole shoot, the same idea as Lightroom's Sync) to every file in the batch; only the two photo-specific fields (mask layers, an overlay layer) are left out, since they reference coordinates/pixels tied to one particular photo. Run Batch shows a live progress bar and per-file status as it works, ending in a "Done — N/M succeeded" summary.
- Save/Load Project (`.photoproject`, `app/project_bundle.py`) persisting the source image, recipe, the editable layer stack (image layers as PNG entries), and mask selections — one-click save.
- "Export Multiple Sizes" — bundle the current photo under several named export profiles (Web/Social/Print/Archive) into one ZIP in a single click, distinct from the Batch Processor's "one recipe, many photos."
- An English / Español interface (the 🌐 Language selector, remembered between launches).
- Preview caching, diagnostics, request IDs, image validation, and structured audit logs.

See `docs/PROJECT_STATUS.md`'s "Known Issues" for the current gaps.

## Privacy

Photo Studio edits on your computer; your photos are never uploaded. It connects to the
internet only to download AI model weights the first time you use a feature that needs them:
AI Denoise (SCUNet, from github.com/cszn/KAIR), AI Upscale (Real-ESRGAN, from
github.com/xinntao/Real-ESRGAN) and AI masks (from the Hugging Face Hub). Those downloads
reveal your IP address to the host, nothing else. Once per launch the app checks for a newer release: it downloads a small version file (update.json) from this project's GitHub Releases and sends nothing but the app's name and version. To turn that off, set the environment variable PHOTO_STUDIO_UPDATE_FEED_URL to off. There is no telemetry.

## Test

```powershell
python -m compileall -q app
python -m pytest -q -n auto           # parallel (pytest-xdist, in requirements-dev.txt); plain `pytest -q` works too
python -m pytest -q -m "not slow"     # skip the AppTest-driven frontend tests
```

Run this yourself before trusting a pass count; the last recorded run is in `docs/PROJECT_STATUS.md` ("Verification": 1777 passed, 13 skipped on 2026-10-22). `$env:PHOTO_PERF_TESTS="1"` runs the 24 MP performance budgets and `$env:PHOTO_BROWSER_TESTS="1"` the real-browser smoke test (`tests/test_browser_smoke.py`, a local Chrome or Edge).

GitHub Actions (`.github/workflows/ci.yml`) runs the same `compileall` + `pytest -n auto` on every push and pull request on Windows and Ubuntu with Python 3.12 and 3.14 (installing `requirements-dev.txt`), plus a job that installs `requirements-raw.txt` and decodes a real (generated) DNG. `requirements-desktop.txt` (pywebview/PyInstaller) is build-only. The test images are generated in code (`tests/synthetic_assets.py`); the only binary fixtures are `tests/assets/portrait_public_domain.jpg` and the golden images in `tests/golden/`.

See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md), [docs/USER_GUIDE.md](docs/USER_GUIDE.md) and [docs/CHANGELOG.md](docs/CHANGELOG.md).
