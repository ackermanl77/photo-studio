# Project Status

## Architecture

```text
Streamlit frontend (app/frontend_unified_app.py; editor HTML in app/ui_editors.py)
        |  HTTP (multipart POST /process, JSON options, X-Photo-Token)
        v
Flask processing backend (app/backend.py, 127.0.0.1, waitress when installed)
        |
        v
app/pipeline.py::run_pipeline()  ->  PhotoProcessor + feature modules (app/geometry.py, ...)
        |
        v
Pillow / NumPy / OpenCV / tifffile / optional rawpy / optional torch (AI Upscale only)
```

The frontend never edits images in-process: each tab captures control values into an `options`
dict, and `app/backend_client.py` sends the source (or a 2048 px proxy for a large photo) plus mask,
per-mask and layer payloads to `/process`. The step order lives only in
`app/pipeline.py::run_pipeline()`: ICC -> sRGB, Geometry (rotate/flip, Lens, straighten, constrain
crop -- `app/geometry.py`), Tone (+ preset), Color, Detail | Repair, Layers, Masks (shared mask, then
per-selection adjustment sets), Portrait, Crop, Effects, AI Upscale. The backend, the CLI and both
batch paths call it; options are validated by `app/edit_schema.py::parse_options()`.

The frontend caches the two halves separately (`_cached_process_through_detail()` /
`_cached_process_from_repair_onward()`); the backend keeps recent Stage 1 results so Stage 2 sends a
handle instead of the image. Interactive requests carry the Streamlit session and a generation, and the
backend abandons a superseded render between steps. `PhotoProcessor` is still used locally only to
decode the upload, to build mask geometry against the geometry-corrected source
(`_lens_corrected_source()`, i.e. `app.geometry.apply_geometry()`), and for the Portrait tab's face
count. The tool tabs are created with `on_change="rerun"`, and display-only work in hidden tabs is
skipped (`_TAB_VISIBLE`); their widgets still render every run.

`app/desktop.py` picks free ports and a per-launch backend token, starts the backend and waits for
`/health`, then the Streamlit child, inside a pywebview window; a second launch focuses the first.
Running `streamlit run app/frontend_unified_app.py` alone also works -- the frontend starts (and, if
it dies, restarts) a backend with its own token.

The desktop wrapper is `app/desktop.py`, and the PyInstaller specification is `PhotoStudio.spec`.

## Primary Entry Points

| Purpose | Entry point |
|---|---|
| Unified application (the only Streamlit UI) | `app/frontend_unified_app.py` |
| Command-line single-image / folder batch processing | `app/cli.py` (run as `python -m app`) |
| Desktop wrapper (packages the unified app in a native window; starts the backend too) | `app/desktop.py` |
| Processing backend the frontend talks to over HTTP | `app/backend.py` |
| HTTP client + on-demand backend startup used by the frontend | `app/backend_client.py` |

Earlier development iterations used additional per-feature Streamlit entry points (`frontend_project.py`, `frontend_layers.py`, `frontend_masking.py`, `frontend_batch.py`, `frontend_library_editor.py`, `frontend_portrait.py`, `frontend_diagnostics.py`, `frontend_release.py`, `frontend_complete.py`, `frontend_center.py`, `frontend_integrated.py`, `frontend_unified_v2.py`, `integrated_workbench_next.py`, and plain `frontend.py`). They were consolidated into `frontend_unified_app.py` and **no longer exist** in `app/` (their compiled `.pyc` files are the only trace left, in `app/__pycache__/`). Any documentation that still tells you to run one of them describes a past development snapshot — see the notice at the top of `docs/workflows/README.md`.

## Release

```text
release/PhotoStudio.exe
release/WebView2Runtime/   -- bundled Fixed Version WebView2 Runtime, see docs/CHANGELOG.md
PhotoStudio-0.1.0-Windows.7z (+ .sha256)   -- the distributable, built from release/
```

`scripts/build_release.ps1` cleans up after itself: PyInstaller's output is moved into `release/` (no second version-named copy anymore), and `build-release/`/`dist-release/` are deleted after a successful build (`-KeepBuildFiles` keeps them). Rebuild with `scripts/build_release.ps1`. Older release notes under `docs/workflows/desktop-exe/` (`current-all-features.md`, `latest-release.md`, `all-features-release.md`, `global-release.md`) reference earlier `dist-*` folder names and test counts from previous builds — they are historical and do not describe the current release.

## Verification

Run `python -m compileall -q app` and `python -m pytest -q` yourself before relying on a pass count (`-m "not slow"` skips the AppTest-driven tests; `PHOTO_PERF_TESTS=1 python -m pytest -m perf` runs the 24 MP budgets). The suite's size and pass count have changed repeatedly across development snapshots (75, 81, and 84 all appear in historical docs under `docs/workflows/`), so no single number here should be trusted as current without re-running it.

## History

The fix history (docs/CHANGELOG.md) is in [`docs/CHANGELOG.md`](CHANGELOG.md).

## Known Issues

- Every interactive edit still costs an HTTP round trip to `app/backend.py` (with a proxy and a
  Stage 1 handle it's small, but the first render after an edit is not instant), and nothing renders
  a sharper version automatically when the user pauses. Non-Local Means denoise is not rescaled for
  the proxy: judge it at 100 % (1:1), which renders at full resolution.
- The working pipeline is 8-bit RGB except Exposure (linear light via a LUT); a float32 linear-light
  pipeline and genuine 16-bit RAW/TIFF output are not implemented (B4).
- The mask overlay previews draw the selection over the geometry-corrected source, not the live
  Tone/Color result (that intermediate image only exists on the backend).
- Film grain is unseeded, so two exports of the same settings don't carry an identical grain pattern
  (deliberate); it is generated on a fixed-size grid so it looks the same in the preview and the export.
- The Export panel is an expander, not an `st.dialog` (a dialog's widgets only exist while it's open,
  which would drop their state between reruns), and offers sRGB output only -- Display P3 would need a
  bundled P3 ICC profile.
- On-canvas editors (crop, curve, paint, pen, pickers) are pointer-driven; the crop has numeric fields
  and the zoom a click navigator, but there's no arrow-key nudging inside the iframes.
- Keyboard shortcuts use Streamlit's native `shortcut=`; their behaviour while a text field has focus
  is Streamlit's, and AppTest can't press keys, so they're covered by source checks and manual testing.
- The repository still contains the third-party test images under `tests/assets/` (no test reads them
  any more); delete them, and purge them from git history if the repository is public.

- Text layers have no font picker or custom font upload -- `app.layer_workspace._load_font()` checks a short list of common TrueType font paths per OS and falls back to Pillow's built-in bitmap font when none exist, so the actual typeface rendered depends on what's installed on the machine running the app (see docs/CHANGELOG.md), not a choice the user makes.
- The free-form Crop tool's corner-drag handles snap to active grid lines/guides; dragging the whole rectangle to move it does not (see docs/CHANGELOG.md for why that scope cut was made).
- Guides can only be added at a typed-in percentage position via a slider, not dragged out from an on-screen ruler the way Photoshop's own guides are -- consistent with this app's established preference (see the free-form crop rectangle's own history) for a plain-widget alternative over adding a new interactive canvas component where a simpler one is workable.
- The Pen tool's curve handles are symmetric only (one handle offset per point, mirrored to act as both that point's incoming and outgoing tangent) -- there's no independent, non-mirrored in/out handle pair (Photoshop's Alt-drag "corner point with two different curve directions"). A zero-length handle still gives a plain straight corner, so corner-only paths remain fully expressible; only a smooth point whose curve direction changes abruptly on each side is out of reach. See docs/CHANGELOG.md for the tool itself.
- A Pen path layer's own vector outline (`stroke_color`/`stroke_width`) is a straight-line-segment polyline approximation of its Bezier curves (24 sample points per segment, see `app.vector_shapes._cubic_bezier_points`), not a mathematically exact curve render -- invisible at any normal zoom level, but a very tight zoom on a large, gently-curving segment could in principle show faceting.
- Ellipse shape layers ignore their own `rotation_deg` -- PIL has no rotated-ellipse drawing primitive without approximating one as a many-sided polygon, judged not worth it for how rarely a rotated oval looks meaningfully different from an unrotated one. Rectangle/Polygon/Star/Line all rotate correctly (their vertices are computed and rotated directly).
- Layer Styles (Drop Shadow/Outer Glow/Stroke/Color Overlay) produce no visible effect on a layer that's fully opaque across the entire canvas (e.g. an uploaded photo used as a same-size overlay with no transparency) -- there's no "outside the shape" area within the canvas for a shadow/glow/stroke to occupy, and the layer's own opaque content covers whatever's drawn behind it regardless. This matches Photoshop's own behavior for the same underlying reason, not a defect specific to this implementation; it's meaningful on Text and Shape layers (which have real transparent padding) and on an Image layer with actual transparency (e.g. a PNG cutout).
- Clicking "➕ Add Path Layer" doesn't reset the Closed Path checkbox back to unchecked for the next new path being drafted (see docs/CHANGELOG.md for why -- it's the checkbox's own widget key, which can't be written to directly). A minor carry-over, not a correctness issue: it only affects the *next* path's starting checkbox state, not any already-added layer.
- ICC soft-proofing targets this app's own sRGB working space, not a real printer/press profile -- there's no profile database or bundled `.icc` file (see docs/CHANGELOG.md), the same limitation already true of Lens Profiles above.
- The denoise algorithm (Non-Local Means, via OpenCV) is a real, meaningful upgrade over the flat median filter it replaced, but it is classical/non-learned, not a neural network -- see docs/CHANGELOG.md for why, given this app's CPU-first, no-GPU-dependency, no-heavy-new-dependency constraints.
- Lens corrections aren't calibrated to any specific camera/lens model — there's no real profile database (see docs/CHANGELOG.md); `LENS_PROFILES`'s entries are generic starting points by lens category, not anything matched against a photo's actual EXIF lens metadata (which isn't read at all currently).
- The HSL Color Mixer's "always render all 8 bands" fix for Streamlit's widget-garbage-collection-on-non-instantiation bug is architecturally different from `_request_rerun()`/`_rerun_pending` and the hidden-textarea commit pattern (both now sharing `_consume_new_commit()`, see docs/CHANGELOG.md) — an unconditional-rendering strategy rather than a guarded session_state write, so it isn't a natural fit for that same helper. Unifying it would mean redesigning how that panel renders, which is deliberately left as-is rather than forced.
- The normal working pipeline is primarily 8-bit RGB; 16-bit TIFF export exists but does not make every adjustment operation 16-bit internally.
- Face detection uses YuNet via `cv2.FaceDetectorYN`; an OpenCV build with neither it nor the Haar cascade reports "no detector" in System Health and the Portrait tab instead of a misleading 0. Skin smoothing is still a global skin-hue mask, not limited to detected faces.
- Microsoft Edge WebView2 Runtime is required for the native desktop window.
- The Library Catalog's keyboard shortcuts (rate/reject/navigate) only work while a real browser tab has the page focused — they reach into `window.parent.document` from a `components.html()` iframe, which `streamlit.testing.v1.AppTest` (a headless harness with no real DOM/JS) can't exercise, so that part is only covered by live manual verification, not the automated suite. The same limitation now also applies to the draggable free-form crop rectangle and the tone curve's click-to-add/drag/double-click-to-remove extra points — their JS interactions are covered by live verification and by driving the hidden commit field directly (the same wiring a real interaction ultimately depends on), not by an automated test that simulates an actual drag or double-click.
- The on-canvas paint editor (including re-editing an existing "Painted" selection's "🎨 Edit Painting") draws at a downscaled preview resolution (capped at 480px on the long edge via `preview_image()`), not the source's full resolution — fine for the bilinear upscale `MaskLayer.normalized()` already does for every mask kind, but a very precise, high-resolution brush edge will look softer than a pixel-perfect selection would.
- Dehaze's transmission map is smoothed with a plain Gaussian blur, not a real edge-aware guided filter (see docs/CHANGELOG.md) — expect a faint halo around strong, hard edges at high strength. There's also no sky segmentation, so a hazy sky can receive more correction than the rest of the scene.
- Magic Wand samples a single pixel's color at the click point with no averaging over a small neighborhood (unlike some tools' "sample size" option) — a click that happens to land on a noisy/anti-aliased pixel can select a narrower or wider region than expected; adjusting Tolerance compensates.
- The histogram clipping-warning overlay uses fixed thresholds (a channel at exactly 0 or 255) with no adjustable margin, and — like the Masks/Repair preview overlays and the ICC gamut warning — is a locally-computed debug preview over the already backend-processed `edited_image`, not a separate network round trip.
- Vibrance's skin-tone protection is a fixed hue-range heuristic (~25 degrees, no adjustable center/width), not real face or skin detection — an unrelated orange/tan subject (wood, terracotta, autumn leaves) gets the same reduced boost a genuine skin tone would.
- `estimate_white_balance()`'s closed-form correction is derived from a linearized (cross-term-dropped) version of `apply_temperature_tint()`'s own formula — exact for a pixel already close to neutral (the intended use: click something that's *supposed* to be gray or white), but only an approximate starting point for a pixel with a severe color cast, the same caveat darktable's own picker carries ("often it works really well," not "always exactly").
- `correct_perspective()`'s two axes (Vertical/Horizontal Keystone) are applied as independent sequential warps rather than one combined homography solve — visually indistinguishable from a combined solve in practice, but means the "untouched" edge of a single-axis correction isn't perfectly pixel-identical to the source (a homography that fixes only 2 of a quadrilateral's 4 corners doesn't leave the line between them pointwise-fixed), just far less changed than the edge actually being corrected.
- Applying a saved custom preset across a Batch Processor run intentionally skips that preset's `mask_layers`/`removal_mask_layers`/overlay layer — see `app/recipe_options.py::recipe_to_options()`'s docstring for why (they're tied to one specific photo's pixel coordinates) — so a preset built around a local mask adjustment will only carry its global tone/color/detail/lens settings into the batch, not the local one. A Free-form crop box is excluded from batch application for the same reason, even though — unlike masks/layers — it's stored size-independently (as fractions) and so could in principle be translated per-photo if a future caller had each batch photo's own dimensions on hand at conversion time.
- The draggable free-form crop rectangle only offers two corner handles (top-left, top-right's and bottom-left's positions follow implicitly) plus drag-the-body-to-move — there's no independent handle on every corner/edge, and a resize is clamped to a minimum 2% of the frame in each dimension so the box can't be dragged down to zero-size.
- The tone curve's "extra" points can be added and dragged freely in both X and Y, but there's no visual warning while dragging one directly onto another point's exact X — the collision is only caught and silently resolved (the earlier-sorted point wins, the later one is dropped) when the curve is actually applied, in `app.recipe_options.merge_curve_points()`.
- Post-Crop Vignette and the Lens tab's corrective vignette are independent controls that both run in the same pipeline (Lens's before Crop, Effects' after) — using both at once stacks their darkening/brightening rather than one overriding the other, which is the expected behavior for two conceptually different tools but can look like "double vignetting" if not intentional.
- "Color Noise" has no effect while Denoise is 0 (the slider is disabled to make this visible) — it's a modifier to Non-Local Means' existing `hColor` parameter, not an independent chroma-only denoise pass, so it can't run without Denoise also being nonzero.
- **AI Upscale (`app/ai_upscale.py`) is a deliberate, explicitly-scoped exception to this app's otherwise CPU-first, no-ML-dependency design** — every other "AI-ish" feature in this app (denoise, dehaze, lens corrections) was kept classical specifically because a real learned model needs a heavy new dependency (PyTorch) and runs meaningfully slower without a GPU (see the denoise entry above and docs/CHANGELOG.md for that reasoning in full). Real-ESRGAN was chosen anyway for this one feature because upscaling — unlike denoise/dehaze/lens-correction, which run on every edit of every photo — is an opt-in, occasional, expensive operation the user explicitly asks for once per export, so the CPU-first constraint's main justification (every edit round-trips through the backend on every slider nudge, per "Architecture" above) doesn't apply to it the same way. It is entirely optional: `torch`/`realesrgan` are never installed by `requirements.txt`/CI, only by the new `requirements-gpu.txt`; `app.ai_upscale.ai_upscale_available()` reports `False` (never raises) when they aren't importable, and the Export settings panel hides the control (showing "Install requirements-gpu.txt to enable AI Upscale on export" instead) in that case, so the rest of the app is completely unaffected on a machine without these packages. When available, it auto-detects CUDA (`torch.cuda.is_available()`) and uses float16 on GPU / float32 on CPU, the same device-selection pattern the sibling PixelFlow project already uses — confirmed against `torch==2.14.0+cu130` on an RTX 5080. Applied last in `app/backend.py::process_image()`'s pipeline (after Effects), since it upscales the final composited/edited result rather than being redone at the original resolution by a later step.
