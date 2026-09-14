# Project Status

## Architecture

```text
Streamlit frontend (app/frontend_unified_app.py)
        |
        v
PhotoProcessor + feature modules (in-process, same Python process)
        |
        v
Pillow / NumPy / OpenCV / scikit-image / optional rawpy
```

`app/backend.py` implements a separate Flask processing API (`127.0.0.1:8502`) with its own copy of the edit pipeline. It is **not** currently started by `app/desktop.py` and is **not** called by `app/frontend_unified_app.py` — the unified frontend calls `PhotoProcessor` directly, in-process, and never talks to Flask. The backend is exercised only by its own tests (`tests/test_backend.py`) today. Treat it as a standalone, optional component until it is either wired in for real or removed (see Known Issues).

The desktop wrapper is `app/desktop.py`, and the PyInstaller specification is `PhotoStudio.spec`.

## Primary Entry Points

| Purpose | Entry point |
|---|---|
| Unified application (the only Streamlit UI) | `app/frontend_unified_app.py` |
| Command-line single-image / folder batch processing | `app/cli.py` (run as `python -m app`) |
| Desktop wrapper (packages the unified app in a native window) | `app/desktop.py` |
| Optional, standalone processing API — not currently wired to any frontend | `app/backend.py` |

Earlier development iterations used additional per-feature Streamlit entry points (`frontend_project.py`, `frontend_layers.py`, `frontend_masking.py`, `frontend_batch.py`, `frontend_library_editor.py`, `frontend_portrait.py`, `frontend_diagnostics.py`, `frontend_release.py`, `frontend_complete.py`, `frontend_center.py`, `frontend_integrated.py`, `frontend_unified_v2.py`, `integrated_workbench_next.py`, and plain `frontend.py`). They were consolidated into `frontend_unified_app.py` and **no longer exist** in `app/` (their compiled `.pyc` files are the only trace left, in `app/__pycache__/`). Any documentation that still tells you to run one of them describes a past development snapshot — see the notice at the top of `docs/workflows/README.md`.

## Release

```text
PhotoStudio-0.1.0-Windows/PhotoStudio.exe
```

Generated build and distribution folders are cleaned after release work. Rebuild with `scripts/build_release.ps1`. Older release notes under `docs/workflows/desktop-exe/` (`current-all-features.md`, `latest-release.md`, `all-features-release.md`, `global-release.md`) reference earlier `dist-*` folder names and test counts from previous builds — they are historical and do not describe the current `PhotoStudio-0.1.0-Windows` release.

## Verification

Run `pytest -q` yourself before relying on a pass count. The suite's size and pass count have changed repeatedly across development snapshots (75, 81, and 84 all appear in historical docs under `docs/workflows/`), so no single number here should be trusted as current without re-running it.

## Recently Fixed

- **Layers and Masks were silently discarded from the final image.** In `app/frontend_unified_app.py`, every tab's code runs on every Streamlit rerun regardless of which tab is visible, and each tab reassigned the same `edited_image` variable. Because the Portrait tab ran last in source order (and its skin-smoothing slider defaulted to `0.2`, not `0`), the canvas preview and the exported file were always `smooth_skin(processor.image, p_smooth)` — which never included the Layers composite (its result was only written to a local variable, never back into `processor.image`), and always applied skin smoothing whether or not the user opened that tab. **Fixed**: the Layers tab now writes its composite back into `processor.image`; the Portrait slider now defaults to `0` (no change) and only mutates `processor.image` when the user opts in; and a single `edited_image = processor.image` line right after all tabs (not one per tab) now feeds the canvas and export, so the final image reflects Tone → Color → Detail → Layers → Masks → Portrait cumulatively regardless of tab order. Verified with a standalone script exercising `PhotoProcessor`/`LayerWorkspace`/`smooth_skin` directly.
- **RAW files were not actually usable from the unified UI.** The uploader accepted `.cr2/.cr3/.nef/.arw/.dng`, but the uploaded bytes were loaded with `PIL.Image.open()` directly, which cannot decode camera RAW. **Fixed**: the upload now goes through `PhotoProcessor.from_bytes(uploaded.getvalue(), uploaded.name)`, which routes RAW extensions through `app.raw_import` (`rawpy`) and everything else through the same PIL path as before. If `rawpy` isn't installed, the app now shows a clear `st.error()` message (from the existing `RuntimeError`) instead of crashing. Verified: non-RAW uploads behave identically to before, RAW extensions are correctly routed, and the missing-`rawpy` case raises the expected friendly error.
- **Only one mask could be active at a time in the unified UI.** The Masks tab was a single dropdown (Luminance/Color/Gradient/Radial) with no way to combine selections, even though `app.advanced_masks.MaskStack`/`MaskLayer` (union of several named, enable/disable-able, invertible, opacity-weighted masks) already existed and was tested. **Fixed**: the Masks tab now lets you add any number of named mask layers (each with its own type-specific parameters, invert, opacity, and edge softening), toggle or remove them individually, and preview the combined selection as a red overlay; `MaskStack.combined()` unions the enabled layers into one mask before the Target Exposure/Contrast/Saturation sliders are applied. Verified with a standalone script: two non-overlapping mask layers (a luminance selection and a radial selection) each contribute their own region to the union, a disabled layer's contribution disappears, and the masked adjustment only changes pixels inside the combined selection.
- **`streamlit-drawable-canvas` was a dead dependency.** It was listed in `pyproject.toml` and `requirements.txt`, and pulled into the PyInstaller build in `PhotoStudio.spec` (hidden imports, its frontend `build/` assets bundled, `collect_all("streamlit_drawable_canvas")`), but was never imported anywhere under `app/` — pure install and build-size overhead. **Fixed**: removed it from `pyproject.toml`, `requirements.txt`, and `PhotoStudio.spec` (all the canvas-specific hidden-import/data-collection code). While in `PhotoStudio.spec`, also fixed a second stale reference in the same file: its `hiddenimports` still listed `"app.frontend"`, which hasn't existed since the frontends were consolidated (same issue as the desktop.py bug fixed earlier) — updated to `"app.frontend_unified_app"`. Verified `pyproject.toml` parses and no longer lists the dependency, `requirements.txt` no longer references it, and `PhotoStudio.spec` still has valid Python syntax after the edit.
- **Cropping / aspect-ratio fitting had no control in the unified UI.** `PhotoProcessor.fit_aspect_ratio()`/`.crop()` existed and were CLI-accessible, but the Streamlit app never called them. **Fixed**: added a new "⛶ Crop" tab (after Portrait, before Presets, so it runs after Masks in tab-execution order and never sees a size mismatch against masks built from the pre-crop image) with an aspect-ratio selector (Square, 4:5, 5:4, 3:2, 16:9, 9:16, or Original/no crop) that calls `processor.fit_aspect_ratio()`, center-cropping the cumulative image from every earlier tab. Verified with a standalone script that `fit_aspect_ratio` is idempotent (re-applying the same ratio to an already-cropped image is a no-op) so the reactive (button-less) Streamlit rerun model can't drift the crop on every interaction.
- **`app/frontend_unified_app.py`'s tab composition had no automated regression coverage.** Every fix listed above (layers/masks dropped from the final image, RAW not decoded, single-mask limit, missing crop control) was the kind of bug that only a test driving the actual composed script — not the individual processing functions in isolation — would catch. **Fixed**: `tests/test_frontend_unified_app.py` now drives the real app through Streamlit's `AppTest` harness (`from streamlit.testing.v1 import AppTest`), simulating an uploaded image via `file_uploader.set_value()` and asserting on the rendered output. It covers the no-image `st.stop()` path, all 9 tabs rendering without exceptions, a layer actually changing the composited canvas image, a combined mask's target-exposure adjustment reaching the final image, the Crop tab changing output dimensions, and Undo restoring a prior control value.
- **Undo/Redo and Save/Load Project were implemented as library code but never wired into the UI.** `app.edit_history.EditHistory` and `app.project.PhotoProject` existed and were tested in isolation, but `app/frontend_unified_app.py` never used them. **Fixed**: every control (Tone/Color/Detail/Masks/Portrait/Crop settings plus the mask-layer list) is captured into one JSON-serializable "recipe" dict (`_build_recipe()`); the Presets tab now records a recipe snapshot into a session-scoped `EditHistory` on every rerun (deduped automatically, so only real changes create a step) and exposes Undo/Redo buttons, and a "Save Project" button bundles the recipe, the original source image, and (if present) the uploaded overlay layer into a downloadable `.photoproject` file via `PhotoProject.save()`. A new "Open a saved project" uploader above the image uploader restores a `.photoproject` via `PhotoProject.load()`. Because Streamlit forbids writing to a widget's `session_state` key after that widget has already been created in the current run, both Undo/Redo and Load Project write the target recipe into a `_pending_restore` session_state entry and call `st.rerun()`; a new block at the very top of the script (before the image uploader) applies `_pending_restore` into every widget's session_state key before any widget is instantiated. Verified with a standalone script: a `.photoproject` saved with a recipe (including a base64-encoded layer payload) round-trips byte-for-byte through `PhotoProject.save()`/`.load()`, and `EditHistory.undo()`/`.redo()` cycle correctly across multiple recorded snapshots, including truncating the redo branch after a new edit follows an undo.

## Known Issues

- The Crop tab only offers a centered aspect-ratio fit (`fit_aspect_ratio`); there is no free-form/draggable crop rectangle in the UI (the underlying `PhotoProcessor.crop()` supports an arbitrary box, but nothing in the UI collects one).
- `ProjectBundle` (the more complete, multi-layer v2 project format) is still unused; Save/Load Project uses the simpler single-layer `PhotoProject` format instead, to keep the UI change bounded.
- Full ICC profile conversion and soft proofing are not implemented; embedded profiles are preserved where supported.
- The normal working pipeline is primarily 8-bit RGB; 16-bit TIFF export exists but does not make every adjustment operation 16-bit internally.
- Face detection gracefully degrades when the installed OpenCV build lacks Haar cascade support.
- Microsoft Edge WebView2 Runtime is required for the native desktop window.
