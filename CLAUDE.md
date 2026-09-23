# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CPU-first Python photo editor (no GPU dependency, deliberately avoids ML/neural dependencies — see "Known Issues" in `docs/PROJECT_STATUS.md` for why denoise/dehaze/lens-correction stay classical algorithms). Streamlit UI, Flask processing backend, optional PyInstaller Windows desktop packaging via pywebview.

## Commands

```powershell
# Install
python -m pip install -r requirements.txt
python -m pip install -r requirements-raw.txt -r requirements-desktop.txt   # optional: RAW import, desktop packaging

# Run the app (only Streamlit entry point in the project)
python -m streamlit run app/frontend_unified_app.py

# Test (run this yourself before trusting a pass count -- historical docs report stale totals)
python -m compileall -q app
python -m pytest -q

# Single test file / test
python -m pytest -q tests/test_backend.py
python -m pytest -q tests/test_backend.py::test_process_endpoint_applies_lens_corrections

# CLI (single image or folder batch, no Streamlit)
python -m app --input photo.jpg --output out.jpg --exposure 1.1 --contrast 1.05

# Desktop build (always lands in release\PhotoStudio.exe, overwritten each run)
.\scripts\build_release.ps1 -Version "0.1.0"
```

CI (`.github/workflows/ci.yml`) runs `compileall` + `pytest` on `ubuntu-latest` with only `requirements.txt` installed — `requirements-raw.txt` (rawpy) and `requirements-desktop.txt` (pywebview/PyInstaller) are dev/build-only and are never exercised by the test suite or CI. Pushing a tag like `v0.1.0` (or a manual `workflow_dispatch`) runs `.github/workflows/release.yml`, which builds the same way `build_release.ps1` does on a `windows-latest` runner.

## Architecture

```
Streamlit frontend (app/frontend_unified_app.py)
        |  HTTP (multipart POST /process, JSON options)
        v
Flask processing backend (app/backend.py, 127.0.0.1:8502)
        |
        v
PhotoProcessor + feature modules (app/image_processor.py + app/*.py)
        |
        v
Pillow / NumPy / OpenCV / scikit-image / optional rawpy
```

**The frontend never edits images in-process.** `app/frontend_unified_app.py` (2300+ lines, the *only* Streamlit entry point — earlier per-feature frontends like `frontend_project.py`/`frontend_layers.py`/etc. were consolidated into it and no longer exist, only stale `.pyc` remnants remain in `app/__pycache__/`) only collects widget values into an `options` dict every tab, every rerun. `app/backend_client.py::process_via_backend()` sends the source image (plus combined mask/layer overlay bytes when present) to `/process`; the JSON+multipart response becomes the canvas/export image. `PhotoProcessor` is still used directly and locally for three things that must not go over the network round trip: decoding the uploaded file (including RAW), building mask geometry against the un-edited source, and face detection for the Portrait tab caption.

**Pipeline order is a single source of truth that must stay in sync in three places**, and drifting it apart reproduces real historical bugs (see `docs/PROJECT_STATUS.md` "Recently Fixed"):
1. `app/backend.py::process_image()` — the actual step order: ICC-normalize to sRGB (unconditional, whenever an embedded profile exists) → Lens → Tone → Color → Detail → Repair → Layers → Masks → Portrait → Crop → Effects.
2. The tab order in `app/frontend_unified_app.py`'s `st.tabs()` — kept close to this for UX, but the tab *visual* position and the backend *pipeline* position are explicitly decoupled (e.g. the Crop tab sits early, right after Tone, purely for workflow convenience; the backend still applies crop last).
3. `app/cli.py::main()` and `app/batch_release.py::process_files()` / `app/recipe_options.py::recipe_to_options()` — independent code paths that apply the same operations and must replicate the same relative ordering (e.g. `--preset` applies right after the Tone group in the CLI, matching where the backend applies a preset, not at the end).

**Every tab's code runs on every Streamlit rerun**, regardless of which tab is visually selected — there's no lazy tab rendering in this app. This is why: (a) a mid-script `st.rerun()` before the last tab renders would garbage-collect not-yet-instantiated widgets' `session_state` (fixed generically via `_request_rerun()`/`_rerun_pending`, consumed once at the true end of the script, after `canvas_col`); (b) any panel that conditionally skips rendering a widget (e.g. a dropdown showing only one item's controls) loses that widget's state on the next rerun the same way — the HSL Color Mixer works around this by always rendering all 8 bands unconditionally rather than sharing the `_request_rerun()` fix.

**State mutation into/out of widgets goes through one convention**: normal control values live directly in widget `session_state` keys, but restoring a value *from outside* a widget (Undo/Redo, Reset, Load Project, Apply Preset, Auto Enhance, the curve editor's commit, lens-profile defaults) always writes into `st.session_state["_pending_restore"]` and calls `_request_rerun()`; a block at the very top of the script (before any widget is instantiated) applies `_pending_restore` into each target widget's key. Never write directly into an already-instantiated widget's `session_state` key — Streamlit forbids it.

**`components.html()` custom editors** (the RGB curve, the on-canvas paint/brush/radial/gradient tool, the magic-wand/white-balance click picker) commit their JS-side state back to Python via a hidden `st.text_area` + a synthetic Ctrl+Enter keydown/keyup — confirmed the only reliable cross-iframe commit mechanism in this Streamlit version. A native `<input>` value-setter trick to directly drive a real `st.slider` from JS does **not** reliably reach the server in this Streamlit version (only a real OS-level keypress or a button `.click()`/text_area Ctrl+Enter do) — route any new on-canvas editor through the hidden-text_area pattern, not through faking slider events.

**Backend caching is two-staged, not one call.** `frontend_unified_app.py` wraps the backend call in `_cached_process_through_detail()` (Lens/Tone/Color/Detail options, `st.cache_data`) feeding `_cached_process_from_repair_onward()` (everything else) so a rerun triggered by a control in only one stage doesn't force the other (more expensive, includes Non-Local Means denoise) stage to recompute.

**Mask/selection layers and image layers use stable UUIDs, not list indices, as widget keys** (`_render_selection_stack()` for Masks/Repair; `_render_layer_stack()` for the Layers tab's own stack, `st.session_state.image_layers`) — indexing by list position previously let removing or reordering one layer corrupt a later layer's widget state after Streamlit's index shifted. Any per-item list widget with a Move Up/Down/Duplicate/Remove pattern must `break` out of its `enumerate()` loop immediately after mutating the underlying list (not just call `_request_rerun()`) — otherwise the item that just got swapped/shifted into a later index gets rendered a second time in the same pass, under the same id, raising `StreamlitDuplicateElementKey`.

**Masks/Repair/Crop preview operate against `_lens_corrected_source()`**, a cached, lens-corrected (but otherwise un-edited) version of the uploaded source — not the raw upload and not the fully backend-processed result — because the real backend applies Lens corrections before Masks/Repair/Crop; painting a mask or framing a crop against the raw source would misalign once the backend actually warps the image.

**`st.tabs()` nested inside `st.expander()` is allowed** (unlike nesting a second `st.expander()`, which Streamlit forbids — the reason `_render_layer_styles()` uses checkboxes instead of its own expander). Each layer's own expander in `_render_layer_stack()` uses this to split into "🎨 Content" / "✨ Styles" / "⚙️ Settings" sub-tabs — like every other tab in this file, every sub-tab's body still executes on every rerun regardless of which is visually selected, so none of the widget-instantiation rules above (deferred reruns, break-after-mutation) change for content nested inside one.

## Module map

- `app/image_processor.py` — `PhotoProcessor`: the core per-image object (load incl. RAW via `app/raw_import.py`, and every `apply_*` adjustment as a thin delegating wrapper into the feature modules below).
- `app/professional_color.py` — HSL color mixer, color grading (split toning), vibrance, white-balance estimation, ICC sRGB normalization + gamut-warning soft proofing, clipping-warning overlay, gamma curves.
- `app/lens_corrections.py` — distortion, vignette, chromatic aberration, perspective/keystone; `LENS_PROFILES` starting points by lens category (not a calibrated per-lens-model database).
- `app/advanced_masks.py` — `MaskStack`/`MaskLayer` (Luminance/Color/Gradient/Radial/Painted/Magic Wand selections, unioned).
- `app/dehaze.py` — classical Dark Channel Prior dehaze.
- `app/layer_workspace.py` — `LayerWorkspace`/`WorkspaceLayer` compositing (opacity, visibility, any number of stacked layers); `layers_to_json()`/`layers_from_json()` serialize a plain layer list (the `/process` wire format), sharing per-layer encode/decode helpers with `LayerWorkspace.to_json()`/`from_json()` (the `.photoproject` file format) so the two are never separate schemas. `render_text_layer()` rasterizes a Text layer's parametric fields (content/font/color/alignment/position) into an RGBA image on demand — a layer in `st.session_state.image_layers` is `"image"` (a stored upload), `"text"`, `"shape"`, or `"path"` (the latter two in `app/vector_shapes.py`), the last three rasterized fresh every rerun by `_layer_dict_to_image()` in the frontend, and Save Project always flattens any kind to a plain image, so Text/Shape/Path editability doesn't survive a save/reload.
- `app/vector_shapes.py` — `render_shape_layer()`: parametric Rectangle/Ellipse/Line/Polygon/Star vector shapes (own fill + outline) via `PIL.ImageDraw`. `render_path_layer()`/`path_outline_points()`: a Pen tool path -- arbitrary anchor points connected by cubic Bezier curves (each point's one handle used symmetrically as both its incoming and outgoing tangent; no independent, non-mirrored in/out handle pair, see "Known Issues" in `docs/PROJECT_STATUS.md`), sampled into a polyline since `PIL.ImageDraw` has no native curve primitive. Both kept parametric the same way Text layers are; the frontend's on-canvas Pen editor is `_pen_editor_html()`.
- `app/layer_styles.py` — `apply_layer_styles()`: Drop Shadow/Outer Glow/Stroke/Color Overlay, stackable on any layer kind. Each effect is computed from the layer's original alpha silhouette independently (never chained through another effect's output — a real bug caught before shipping, see "Recently Fixed"), then composited in Photoshop's own back-to-front order. Called by `_layer_dict_to_image()` after rasterizing a layer's base content, before it becomes a `WorkspaceLayer`.
- `app/major_features.py` — also home to `LayerStack` (the float-based compositing math: 23 Photoshop-matching blend modes, dispatched via `BLEND_MODE_LABELS`) and `PhotoCatalog` (SQLite-backed library ratings/tags/search).
- `app/professional_tools.py` — canvas-only display overlays (`apply_mask_overlay`, `draw_guides_and_grid`/`grid_line_fractions`/`GRID_PRESETS` for the Guides & Grid panel), histogram/zoom/preview helpers, `EXPORT_PROFILES`/`export_profile()`. None of these touch `app/backend.py`'s pipeline — they render a modified copy of an already-processed image for the canvas only.
- `app/portrait_tools.py` — skin smoothing, face-detection caption.
- `app/edit_history.py` / `app/edit_stack.py` — Undo/Redo recipe history.
- `app/project_bundle.py` (current `.photoproject` v2 format) / `app/project.py` (older v1 format, still loadable for backward compatibility, no longer written) — Save/Load Project.
- `app/reliability.py` — archive validation (`validate_project_archive()`: real decompressed-byte-count enforcement, path-traversal/drive-letter checks, manifest-reference checks) guarding both project formats against a malicious `.photoproject` zip.
- `app/library.py` — folder scanning/indexing into the catalog.
- `app/presets.py` (built-in named looks) / `app/custom_presets.py` (`PresetStore`, user-saved) / `app/preset_exchange.py` (JSON export/import).
- `app/batch.py` / `app/batch_release.py` — folder batch processing, filename templates, ZIP delivery; `recipe_to_options()` in `app/recipe_options.py` converts a saved recipe into backend `options` shape for "sync settings across a batch." `export_multiple()` is a different axis (one photo, several `EXPORT_PROFILES` variants zipped together via the same `make_zip()`) — "Export As," not batch processing.
- `app/backend.py` — Flask app, `/process` and `/health`; the only place the full pipeline order is authoritative.
- `app/backend_client.py` — HTTP client + on-demand backend auto-start (`ensure_backend_running()`, cached via `st.cache_resource`).
- `app/desktop.py` — pywebview native window; starts the backend as a child process (waits for `/health`) before starting the Streamlit child; points pywebview at a bundled WebView2 Runtime via `webview.settings["WEBVIEW2_RUNTIME_PATH"]`.
- `app/cli.py` (run as `python -m app`) — standalone single-image/folder CLI, independent of the Flask backend.
- `app/reliability.py`, `app/observability.py`, `app/diagnostics.py` — image validation, structured audit logs/request IDs, health diagnostics.
- `app/cached_processing.py`, `app/performance.py` — preview caching.

## Testing conventions

- Tests that touch the photo library or presets (`Path.home()` → `~/.photo-studio/`) `monkeypatch` `Path.home()` to a `tmp_path` so they don't write real files on the machine running the suite.
- `tests/test_frontend_unified_app.py` drives the real app through Streamlit's `AppTest` harness (`streamlit.testing.v1.AppTest`), simulating uploads via `file_uploader.set_value()` — this is the regression net for the "every tab runs on every rerun" and "state must survive an unrelated tab's rerun" bug class described above.
- `AppTest` is headless (no real DOM/JS), so anything living inside a `components.html()` iframe — drag interactions on the curve/paint editors, the Library Catalog's keyboard shortcuts — cannot be exercised by it. Those are verified by driving the hidden commit field directly (the same wiring a real interaction ultimately depends on) plus live manual verification, not by an automated drag/keypress simulation.
