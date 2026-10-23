# Project Status

The current-state reference: architecture, entry points, backend API and limits, release and
installer, verification and known issues. Using the app: [`USER_GUIDE.md`](USER_GUIDE.md). Fix
history: [`CHANGELOG.md`](CHANGELOG.md). Contributor notes: [`../CLAUDE.md`](../CLAUDE.md).

## Architecture

```text
Streamlit frontend (app/frontend_unified_app.py + the app/ui/ package; editor HTML in app/ui_editors.py)
        |  HTTP (multipart POST /process, JSON options, X-Photo-Token)
        v
Flask processing backend (app/backend.py, 127.0.0.1, waitress when installed)
        |
        v
app/pipeline.py::run_pipeline()  ->  PhotoProcessor + feature modules (app/geometry.py, ...)
        |
        v
Pillow / NumPy / OpenCV / tifffile / optional rawpy / optional torch (GPU pack: AI Upscale and opt-in export resize/sharpen)
```

The frontend never edits images in-process: `app/frontend_unified_app.py` (about 240 lines, the
only Streamlit entry point) composes the panels in `app/ui/` (`tabs/`, `canvas.py`, `export_panel.py`,
`selections.py`, `layers.py`, `processing.py`, ...). Each panel records its widgets' values in an
`EditorContext`, and the options are built from them through the one control registry,
`app/controls.py::CONTROLS` (which also generates the widgets, the recipe sanitizer, the
control-backed option specs and the stage split). Through `app/backend_client.py` the frontend sends
the source (or a 2048 px proxy for a photo whose long edge exceeds 2560 px) plus mask, per-mask and
layer payloads to `/process`. The step order lives only in
`app/pipeline.py::run_pipeline()`: ICC -> sRGB (or Display P3 for a P3 export), Geometry
(rotate/flip, Lens, straighten, constrain crop, Liquify -- `app/geometry.py`), Tone (+ preset), Color,
Detail | Repair, Layers (incl. adjustment layers), Masks (shared mask, then per-selection adjustment
sets), Portrait, Relight, Crop, Background, Effects (creative filters, vignette, grain), AI Upscale; a
Background "Remove" becomes the result's alpha channel last. The backend, the CLI and both
batch paths call it; options are validated by `app/edit_schema.py::parse_options()`.

The frontend caches the two halves separately (`app.ui.processing.cached_process_through_detail()` /
`cached_process_from_repair_onward()`); the backend keeps recent Stage 1 results so Stage 2 sends a
handle instead of the image. Interactive requests carry the Streamlit session and a generation, and the
backend abandons a superseded render between steps. `PhotoProcessor` is still used locally only to
decode the upload, to build mask geometry against the geometry-corrected source
(`lens_corrected_source()`, i.e. `app.geometry.apply_geometry()`), and for the Portrait face count.
The nine tool tabs (grouped by workflow: Basic, Curves & Color, Detail, Geometry, Local, Layers,
Effects, Export, Presets & Library) are created with `on_change="rerun"`, and display-only work in
hidden tabs is skipped (`app.ui.state.tab_visible()`); their widgets still render every run. The
canvas view and the scopes are `st.fragment`s, so panning or toggling a scope reruns only them.
Logging is configured only by the entry points (backend `serve()`, desktop, CLI, the Streamlit
script) into `logs/<component>.log` in the user data folder.

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

Earlier development iterations used additional per-feature Streamlit entry points (`frontend_project.py`, `frontend_layers.py`, `frontend_masking.py`, `frontend_batch.py`, `frontend_library_editor.py`, `frontend_portrait.py`, `frontend_diagnostics.py`, `frontend_release.py`, `frontend_complete.py`, `frontend_center.py`, `frontend_integrated.py`, `frontend_unified_v2.py`, `integrated_workbench_next.py`, and plain `frontend.py`). They were consolidated into `frontend_unified_app.py` and **no longer exist**; neither do `app/cached_processing.py`, `app/performance.py`, `app/edit_stack.py` and `app/release_preflight.py` (removed as dead code, B12) -- stale `.pyc` files of removed modules may linger in `app/__pycache__/`. Old mentions of any of them in `docs/CHANGELOG.md` describe a past development snapshot.

### Backend API and limits

`app/backend.py` (Flask; served by waitress when installed) binds to `127.0.0.1` (`PHOTO_BACKEND_HOST`/`PHOTO_BACKEND_PORT`, default port 8502; the desktop launcher picks free ports per launch) and, whenever `PHOTO_BACKEND_TOKEN` is set (always under `app/desktop.py`, and for a backend the frontend starts itself), answers every request except `/health` with 401 unless it carries that token in `X-Photo-Token`:

- `GET /health` -- status.
- `POST /ai_mask` -- an AI Subject/Person mask (`app/ai_masks.py`, needs the GPU pack) or the classical Sky mask for the uploaded image, returned as a grayscale PNG; the frontend adds it as a Painted selection.
- `POST /process` -- multipart: the source `image`, optional mask / per-selection / layer payloads, and an `options` JSON field validated by `app/edit_schema.py::parse_options()` (unknown keys are rejected). The response is JSON + multipart; Stage 1 results come back with an `X-Photo-Stage-Handle` that Stage 2 sends instead of the image (410 when it has expired). A superseded interactive request ends with 409; every response carries an `X-Request-ID`.

Limits (environment variables, read at start-up by `app/config.py`): `PHOTO_MAX_UPLOAD_BYTES` (default 50 MB), `PHOTO_MAX_IMAGE_PIXELS` (50,000,000), `PHOTO_MAX_UPSCALE_OUTPUT_PIXELS` (200,000,000, the AI Upscale result), `PHOTO_STAGE_CACHE_ENTRIES` (4 Stage 1 handles), `PHOTO_MAX_HEAVY_JOBS` (2 concurrent Non-Local Means / dehaze steps). Per-user files (settings, presets, catalog, logs, an optional GPU pack) live in `%LOCALAPPDATA%\PhotoStudio` on Windows, `~/.photo-studio` elsewhere (`app.config.user_data_dir()`; `PHOTO_STUDIO_DATA_DIR` overrides).

## Release

```text
release/PhotoStudio.exe
release/WebView2Runtime/   -- bundled Fixed Version WebView2 Runtime, see docs/CHANGELOG.md
PhotoStudio-0.1.0-Windows.7z (+ .sha256)   -- the distributable CPU base, built from release/
release-gpu-pack/gpu-pack/ -- the optional GPU pack (-GpuPack): torch for AI Upscale/export transforms, Real-ESRGAN
PhotoStudio-0.1.0-GPU-pack.7z (+ .sha256)  -- extracts as gpu-pack/, to put next to PhotoStudio.exe
```

The release is split (K5): the CPU base package carries the app without PyTorch, and the optional
GPU pack carries torch (for AI Upscale and opt-in CUDA export resize/sharpen) plus Real-ESRGAN's
packages and a `gpu-pack.json` manifest (Python version, torch version). `app/gpu_pack.py` looks for
it in `$PHOTO_STUDIO_GPU_PACK`, next to the exe, then in the
user data folder, checks the manifest and appends its `site-packages` to `sys.path` (the base's own
packages stay first). The base bundles the whole standard library so the pack's packages can import
what they need. The installer offers the pack as an optional component when it was built;
`-BuildProfile gpu` still builds the older all-in-one folder.

**Exe resources (K6/K7).** `PhotoStudio.spec` embeds three things from `scripts/windows_resources.py`:
- a version resource rendered from `app.__version__` (publisher "MELEJ", as in the installer);
- a manifest with `activeCodePage` UTF-8 and `dpiAwareness` PerMonitorV2;
- `app/assets/photoproject.ico` as the second icon group, which the `.photoproject` `DefaultIcon` uses as `PhotoStudio.exe,1`.

The spec excludes numba/llvmlite/facexlib/gfpgan/filterpy (realesrgan's metadata pulls them in; nothing imports them), and the GPU pack prunes the same names.

`scripts/build_release.ps1` cleans up after itself: PyInstaller's output is moved into `release/` (no second version-named copy anymore), and `build-release/`/`dist-release/` are deleted after a successful build (`-KeepBuildFiles` keeps them). Rebuild with `scripts/build_release.ps1`.

**Installer.** `build_release.ps1 -Installer` runs Inno Setup 6 (`ISCC.exe` on `PATH`, in Program Files,
or a per-user install under `%LOCALAPPDATA%\Programs\Inno Setup 6`) on `installer/PhotoStudio.iss`,
passing `-Version` through, and writes `installer-output\PhotoStudio-Setup-<version>.exe` (gitignored).
It packages the finished `release\` folder (including `WebView2Runtime\`). Setup asks "all users"
(admin, `Program Files`) or "just me" (no admin, `%LOCALAPPDATA%\Programs`; `/CURRENTUSER` on the
command line), offers a `.photoproject` file association (written per machine or per user to match)
and desktop/Start-menu shortcuts, and adds a "GPU pack" component when `-GpuPack` built one. The
uninstaller asks whether to delete `%LOCALAPPDATA%\PhotoStudio` (presets, catalog, logs, a GPU pack
kept there); a silent uninstall (`/SUPPRESSMSGBOXES`) keeps it.

**Code signing.** `build_release.ps1 -Sign` (thumbprint from `-CertificateThumbprint` or
`CODESIGN_THUMBPRINT`) checks the certificate and finds signtool (PATH, else the newest Windows Kits 10)
before the build, signs `PhotoStudio.exe` (SHA-256, RFC 3161 timestamp from `CODESIGN_TIMESTAMP_URL`,
default DigiCert), and with `-Installer` passes Inno Setup a `codesign` tool: `PhotoStudio.iss` turns on
`SignTool`/`SignedUninstaller` under `#ifdef Sign`, so the installer and its uninstaller are signed too.
Checked on 2026-10-07 with a self-signed test certificate (all three signed and timestamped). Releases
aren't signed with a trusted certificate yet, so SmartScreen still warns; `release.yml` doesn't sign.

## Verification

Run `python -m compileall -q app` and `python -m pytest -q -n auto` (or without `-n auto`) yourself
before relying on a pass count (`-m "not slow"` skips the AppTest-driven tests; `PHOTO_PERF_TESTS=1
python -m pytest -m perf` runs the 24 MP budgets). CI (`.github/workflows/ci.yml`) runs the same on
Windows and Ubuntu with Python 3.12 and 3.14, plus a `raw-decode` job with `requirements-raw.txt`.

Last run 2026-10-22 on Windows 11 / Python 3.14 with the dev `.venv`'s packages: `compileall` clean;
`pytest -q -n auto` **1777 passed, 13 skipped** in about 27 s (716/12 on 2026-10-07, 639/11 on 2026-10-03). The 13 skips are by
design: nine photo-specific or UI-only controls in `tests/test_parity_golden.py`, the two opt-in
`PHOTO_PERF_TESTS` budgets, the opt-in real-browser smoke test (`PHOTO_BROWSER_TESTS=1`, see below), and the no-file-manager fallback of "Open logs folder" (Windows always has one).

`tests/test_browser_smoke.py` (opt-in) starts the real app and a headless Chrome/Edge over the DevTools
protocol and checks refine-on-pause, a real mouse drag on the tone curve committing through the hidden
field, and a keyboard-moved slider re-rendering the canvas. It passed three runs in a row on 2026-10-22. That venv has CUDA torch 2.14 and Real-ESRGAN, so the GPU-export and
AI Upscale tests ran; without torch the ones that need it skip (CI).

Release checks last run on 2026-10-01 (Windows 11, RTX 5080): `build_release.ps1` built both profiles
from a fresh `build-venv\<profile>` and `--self-test` passed (cpu: about 1.0 GB, 2,795 files; gpu with
`-ExtraIndexUrl https://download.pytorch.org/whl/cu130`: about 4.2 GB, 5,945 files). The gpu profile
now patches basicsr itself (see `requirements-gpu.txt`). The pinned WebView2 cab matches Microsoft's
download (same SHA-256, Microsoft-signed). The packaged exe was launched headlessly: free per-launch
ports bound to 127.0.0.1, `/process` returns 401 without the token or with a wrong one and 200 with it,
the window uses the bundled `WebView2Runtime`, and a second launch exits in under a second without
starting servers. The startup-failure dialog was checked by reading the code only, not by triggering it.
Real CUDA Real-ESRGAN with 12 forced tiles matched an untiled run (PSNR 59 dB, no seams, no NaNs). CI (`ci.yml`) is green on `master` (2026-10-07); the tag-triggered `release.yml` hasn't run since
`v0.1.0`, and `v0.2.0` waits until GitHub Actions artifact uploads work again (the account's free
monthly artifact storage ran out on 2026-10-07 and resets on 1 November).

2026-10-08: a full `-GpuPack -ExtraIndexUrl https://download.pytorch.org/whl/cu130 -Installer` build. The pack
(`release-gpu-pack\gpu-pack`, 3.5 GB: torch/torchvision 2.14.1/0.29.1+cu130, Real-ESRGAN 0.3.0, the patched
basicsr 1.4.2) passed the base exe's self-test (GPU pack active, AI Upscale importable), and the installer
(2.16 GB with the pack) installed per user with the GPU-pack component selected (31,181 files, 4.35 GB); the
installed app found `{app}\gpu-pack` by itself and its self-test passed. A real 4x AI Upscale through
`PhotoStudio.exe --backend-child` /process, from the release folder and from the installed copy: 512x384 ->
2048x1536 in 0.4 s warm (4-5 s the first time, including the weights download and CUDA start-up) on the RTX
5080, about 0.9 GB of VRAM. The silent uninstall removed the folder, shortcuts and uninstall entry and kept
the user data folder.

2026-10-07: a `-SkipArchive -Installer -Sign` build with the lensfun database bundled passed `--self-test`
and produced a signed exe, uninstaller and installer (self-signed test certificate).

2026-10-02: the installer was built (`-Installer`) and installed/uninstalled silently, which found and
fixed the per-user install, the silent-uninstall prompt and a leftover registry key (`c78612b`); its
GPU-pack component was not exercised. The CUDA export path was checked on an RTX 5080: a 4500x3000 ->
2048 px Screen-sharpened export matched the CPU path to a mean of 0.35 levels (max 4), 202 ms vs
305 ms once warm, with ~8 s of CUDA start-up on the first call.

Clean-up round, 2026-10-02: a scratch PyInstaller build of the CPU base (from a fresh venv, outside
`release/`; 331 MB / 2,585 files without WebView2) passed `--self-test`, and with
`PHOTO_STUDIO_GPU_PACK` pointing at a pack folder whose `site-packages` was the dev venv's (CUDA torch,
Real-ESRGAN) it found the pack and really imported torch + Real-ESRGAN (`ai_upscale_importable`). (The
full `-GpuPack` build and the installer's GPU-pack component were checked on 2026-10-08, see above.)

### Streamlit upgrades

`requirements.txt` pins `streamlit>=1.64.0,<2` and `constraints-cpu.txt` the exact version. Raising
it means checking, in a real browser (`streamlit run app/frontend_unified_app.py`, desktop window
too), the rules in `app/ui/static/theme.css` marked "Streamlit DOM" and the behaviours the app relies
on:

1. The default footer stays hidden, the header is transparent and takes no clicks, and the page
   padding is unchanged; collapsing the tools sidebar leaves its expand button
   (`[data-testid="stExpandSidebarButton"]`, inside the header's toolbar) visible and working.
2. The nine tool tabs show as one row of icons, with the name on hover and under the row for the
   open tab, while the sub-tabs keep their words (`icon_tabs_style()` in `app/ui/theme.py`:
   `.st-key-tool-tab > div > [role="tablist"] > [data-testid="stTab"][data-key][aria-selected]`).
   On a 390 px-wide window the controls above the photo wrap into a compact grid
   (`.st-key-canvas-controls [data-testid="stHorizontalBlock"/"stColumn"]`).
3. The tools sidebar is at least 420 px wide while open (`[data-testid="stSidebar"][aria-expanded="true"]`),
   can be dragged wider and collapsed, and the photo is capped to the window's height
   (`.st-key-canvas-stage [data-testid="stImage"] img`).
4. The editors' hidden commit text_areas stay invisible (`.st-key-*-hidden`, `[class*="-paint-hidden"]`)
   and a commit still lands: drag the curve, paint and add a selection, drag the crop box, click the
   white-balance picker, add Pen points, drag-pan at 200 %.
5. Tab switching still reports `.open` (hidden tabs skip thumbnails/face detection), the Local and
   Presets & Library sub-tabs too; keyed expanders keep their open state when a "●" mark appears.
6. Fragments: panning at 200 % and toggling the waveform/clipping scopes rerun only the canvas; the
   refine-on-pause status line still swaps in the full-resolution render.
7. `shortcut=` buttons (Ctrl+Z, Ctrl+Shift+Z, Ctrl+E, Z, B, G) and the Library's 1-5/0/X/arrow keys.
   Find a tool still switches tabs through `st.session_state["tool-tab"]` (tabs with `on_change="rerun"`),
   and its one-shot script still finds the sidebar (`[data-testid="stSidebar"] .st-key-<key>`), opens an
   expander by clicking its `<summary>` and flashes the place; the ⌨️ button's icon isn't clipped.
8. The `--ps-*` colours: header, canvas background, export footer, and the editors' panels/handles.
9. Then run the whole suite (AppTest catches API changes, not layout ones).

Last upgrade: 1.64.0 -> 1.65.0 (2026-10-06, Dependabot #9). Checked headless in Chromium with a photo
loaded: every `data-testid` hook the stylesheet targets, the hidden header, the tab-row wrapping,
the narrow-window column order, the 0-px commit fields and the `--ps-*` variables measured identical
to 1.64 at 1400 and 800 px (items 1-3, 8); the suite passes (item 9). Items 4-7 (editor commits,
tab `.open`, fragments, shortcuts) still need a real browser.

## History

The fix history, newest first, is in [`docs/CHANGELOG.md`](CHANGELOG.md).

## Known Issues

- Every interactive edit still costs an HTTP round trip to `app/backend.py` (with a proxy and a
  Stage 1 handle it's small, but the first render after an edit is not instant). After a 1.2 s pause
  a proxy preview is replaced by a full-resolution render (`app/refine_preview.py`, a background
  thread; a newer edit supersedes it on the backend's "refine" lane). Until that lands, Non-Local
  Means denoise is not rescaled for the proxy: judge it at 100 % (1:1) or after the pause. The
  refine's polling fragment and its swap-in are covered by unit tests and AppTest with the job run
  inline, and the real thread + `run_every` polling by the opt-in browser smoke test.
- The interactive preview and the default export stay 8-bit. A 16-bit PNG/TIFF export (Export
  settings > Bit depth, CLI `--bit-depth 16`, API `bit_depth`/`export.bit_depth`) runs the pipeline
  in float32 (`app/high_bit.py`, PhotoProcessor's high-precision mode) from a full-precision decode
  (16-bit PNG/TIFF, RAW at 16 bits): Geometry/Lens, Exposure, Brightness, Contrast, Highlights/
  Shadows, Temperature/Tint, Saturation, Curve, Clarity, HSL, Vibrance, Color Grading, Texture,
  Sharpen, mask blending (shared and per-selection), Crop, the Effects vignette and film grain compute
  in float (Texture and Sharpen blur with OpenCV, since Pillow has no float blur -- within a level of
  the 8-bit preview). Since 2026-10-18 so do Dehaze (`dehaze_float`), Portrait skin smoothing
  (`smooth_skin_float`, mask from the 8-bit view) and the ICC conversion of matrix/TRC profiles --
  Adobe RGB, ProPhoto, Display P3 -- (`app/icc_float.py`: the profile's curves and colorants parsed
  and applied in float, sRGB's taken from the profile the output is tagged with; within 0.2 levels
  of LittleCMS on average). Denoise (OpenCV's Non-Local Means is 8-bit only), Repair (inpaint /
  seamlessClone are 8-bit), Layers (their content is 8-bit anyway) and LUT-based ICC profiles still
  run in 8 bits and only their *change* is added to the float data, which keeps the sub-8-bit detail
  around them; AI Upscale restarts from its 8-bit result. The working space stays sRGB-encoded
  (Exposure alone works in linear light, as before) by design: a linear-light working space for every
  step would change the look of every slider and make the 16-bit file differ from the 8-bit preview. The 8-bit steps truncate after each operation, so a heavy 8-bit edit is ~1 % darker than
  its 16-bit export. A 16-bit TIFF keeps the ICC profile, the IFD0 text tags and the EXIF and GPS
  sub-IFDs (`app.export._tiff16`: tifffile won't write the pointer tags, so they're written as
  placeholders and linked to sub-IFDs appended with Pillow's TIFF tag writer); a 16-bit PNG keeps
  the whole EXIF block.
- The mask overlay previews draw the selection over the geometry-corrected source, not the live
  Tone/Color result (that intermediate image only exists on the backend).
- Film grain is unseeded, so two exports of the same settings don't carry an identical grain pattern
  (deliberate); it is generated on a fixed-size grid so it looks the same in the preview and the export.
- Export settings live in the ⬇ Export tab, not an `st.dialog` (a dialog's widgets only exist while
  it's open, which would drop their state between reruns). Color space: sRGB or Display P3
  (`app/color_profiles.py`), with an ICC v4 Display P3 profile generated in code (no bundled file);
  littlecms, using that profile, agrees with the app's own sRGB -> P3 matrix within one 8-bit level.
  Since 2026-10-17 (B14) a P3 export renders in a **Display P3 working space** (`working_space`
  option, set by the backend from `export.color_space`, by the CLI's `--color-space` and by the folder
  batch): camera RAW is decoded by LibRaw straight to P3 (`rawpy.ColorSpace.P3D65`), an embedded
  profile is converted to P3 instead of to sRGB (so Adobe RGB/ProPhoto/P3 sources keep what fits in
  P3), an untagged source is taken as sRGB and converted numerically (in float at 16 bits), and
  `encode_export()` doesn't convert again pixels already tagged with our P3 profile. Before that, the
  pipeline clipped every source to sRGB first, so a P3 file could not hold a colour outside sRGB.
  Display P3 shares sRGB's transfer curve, so the steps see values on the same scale; hue-based tools
  (HSL bands, the skin mask) see P3 hues, a little different from sRGB's for strongly saturated
  colours. The interactive preview stays sRGB, and a P3 export always takes the full-resolution
  backend render (it needs the original file). No Adobe RGB / ProPhoto output.
- Fragment-only reruns (panning, scopes) can't be exercised by AppTest, which always reruns the whole
  script; they are covered by full-run tests plus a manual browser check. Streamlit's `persist_state=`
  was evaluated as a way to stop rendering hidden widgets (e.g. one HSL band at a time) and not
  adopted: under AppTest a persisted value was not restored when the widget reappeared and was lost
  after an unrelated rerun.
- The on-canvas editors are static components (F14, 2026-10-11): `app/ui/static/editors/<name>.html`
  (markup and CSS with `{{field}}` slots) and `<name>.js`, plus the shared `commit.js`; `app/ui_editors.py`
  only fills the slots (escaped, translated through `app.i18n.t()`) and passes one JSON `CONFIG`, and
  `_render()` inlines them into one page because components.html() can't load sibling files. They
  still commit through the hidden text_area channel: Streamlit's bidirectional component API was
  evaluated and not adopted (it needs a JS build step and a second commit path, while this one is
  verified live and under Node).
- On-canvas editors take the keyboard (F11): arrow keys nudge the focused crop corner/box, Pen
  anchor (Alt: its handle; Delete removes it), curve point and the last radial/gradient; [ / ] size
  the brush; the zoomed canvas pans by drag or arrows (F3). Their JavaScript is exercised under Node
  with a small fake DOM (`tests/js_fake_dom.js`; skipped without Node), not in a real browser --
  focus handling, iframe key routing and the drag feel still need a manual check. The magic-wand/
  white-balance pickers remain click-only. Split View and Original keep the slider/navigator pan.
- Keyboard shortcuts use Streamlit's native `shortcut=`; their behaviour while a text field has focus
  is Streamlit's, and AppTest can't press keys, so they're covered by source checks and manual testing.
- The third-party test images were deleted from `tests/assets/`, but they remain in git history; purge
  them from history (a force-push) if the repository is public.
- Text layers and the watermark store their font by display name (`app/fonts.py`: the fonts installed on this machine plus ones uploaded under Layers > Fonts, kept in `user_data_dir()/fonts`). A project or preset opened on a machine without that font shows it as "(not installed)" and renders with the automatic default (a short list of common fonts, then Pillow's built-in font); the font file itself isn't embedded in projects. The installed-font list is scanned once per process, so a font installed while the app runs appears after a restart (or upload it).
- The free-form Crop tool's corner-drag handles snap to active grid lines/guides; dragging the whole rectangle to move it does not (see docs/CHANGELOG.md for why that scope cut was made).
- Guides can only be added at a typed-in percentage position via a slider, not dragged out from an on-screen ruler the way Photoshop's own guides are -- consistent with this app's established preference (see the free-form crop rectangle's own history) for a plain-widget alternative over adding a new interactive canvas component where a simpler one is workable.
- The Pen tool's curve handles are symmetric only (one handle offset per point, mirrored to act as both that point's incoming and outgoing tangent) -- there's no independent, non-mirrored in/out handle pair (Photoshop's Alt-drag "corner point with two different curve directions"). A zero-length handle still gives a plain straight corner, so corner-only paths remain fully expressible; only a smooth point whose curve direction changes abruptly on each side is out of reach. See docs/CHANGELOG.md for the tool itself.
- A Pen path layer's own vector outline (`stroke_color`/`stroke_width`) is a straight-line-segment polyline approximation of its Bezier curves (24 sample points per segment, see `app.vector_shapes._cubic_bezier_points`), not a mathematically exact curve render -- invisible at any normal zoom level, but a very tight zoom on a large, gently-curving segment could in principle show faceting.
- Layer Styles (Drop Shadow/Outer Glow/Stroke/Color Overlay) produce no visible effect on a layer that's fully opaque across the entire canvas (e.g. an uploaded photo used as a same-size overlay with no transparency) -- there's no "outside the shape" area within the canvas for a shadow/glow/stroke to occupy, and the layer's own opaque content covers whatever's drawn behind it regardless. This matches Photoshop's own behavior for the same underlying reason, not a defect specific to this implementation; it's meaningful on Text and Shape layers (which have real transparent padding) and on an Image layer with actual transparency (e.g. a PNG cutout).
- ICC soft-proofing targets this app's own sRGB working space, not a real printer/press profile -- there's no profile database or bundled `.icc` file (see docs/CHANGELOG.md), the same limitation already true of Lens Profiles above.
- The denoise algorithm (Non-Local Means, via OpenCV) is a real, meaningful upgrade over the flat median filter it replaced, but it is classical/non-learned, not a neural network -- see docs/CHANGELOG.md for why, given this app's CPU-first edit pipeline and no-heavy-new-dependency constraints. CUDA export transforms are a separate, optional final-output acceleration and do not change how denoise or other edits run.
- Lens corrections from the photo's lens (`app/lens_lookup.py`, lensfun via the optional `lensfunpy` in requirements-raw.txt) fit the app's simple slider models (one radial k1 term, a linear vignette ramp, a radial red/blue scale) to lensfun's calibration -- about three quarters of a typical zoom's distortion field; higher-order ("mustache") terms aren't representable without a full lensfun warp. RAW files are decoded without their EXIF, so they can't be looked up. `LENS_PROFILES` remain generic starting points by lens category.
- The HSL Color Mixer still renders all 8 bands (collapsed, keyed expanders) on every run: a widget that isn't rendered loses its state, and `persist_state=` didn't hold up as the fix (see above). `request_rerun()` and the hidden-textarea commit guard (`consume_new_commit()`) are the other two state helpers, all in `app/ui/state.py`.
- Face detection uses YuNet via `cv2.FaceDetectorYN`; an OpenCV build with neither it nor the Haar cascade reports "no detector" in System Health and the Portrait tab instead of a misleading 0. Skin smoothing multiplies its skin-hue mask by feathered ellipses around the faces found on the image being edited (Portrait > "Only around detected faces", on by default); with no face found it falls back to the global mask, so skin-coloured backgrounds in a photo without faces are still smoothed.
- The Library reads XMP sidecars (`app/xmp_sidecar.py`: `xmp:Rating`, -1 = rejected, and `dc:subject` keywords) when it catalogues a photo for the first time, and writes rating/tag changes back only with "Also write ratings and tags to XMP sidecars" on; other fields of an existing sidecar are kept. The catalog remembers each sidecar's modification time, so a rescan imports a sidecar another program changed since (its rating/tags then replace the catalogued ones); a catalog from before this was tracked only records the times on its first rescan.
- Microsoft Edge WebView2 Runtime is required for the native desktop window; the release bundles a pinned Fixed Version copy (`release\WebView2Runtime\`), which does not auto-update -- `.github/workflows/webview2-age.yml` opens an issue when it is more than 90 days old.
- The Library Catalog's keyboard shortcuts (rate/reject/navigate) only work while a real browser tab has the page focused — they reach into `window.parent.document` from a `components.html()` iframe, which `streamlit.testing.v1.AppTest` (a headless harness with no real DOM/JS) can't exercise, so that part is only covered by live manual verification, not the automated suite. The same limitation now also applies to the draggable free-form crop rectangle and the tone curve's click-to-add/drag/double-click-to-remove extra points — their JS interactions are covered by live verification and by driving the hidden commit field directly (the same wiring a real interaction ultimately depends on), not by an automated test that simulates an actual drag or double-click.
- The on-canvas paint editor (including re-editing an existing "Painted" selection's "🎨 Edit Painting") draws at a downscaled preview resolution (capped at 480px on the long edge via `preview_image()`), not the source's full resolution — fine for the bilinear upscale `MaskLayer.normalized()` already does for every mask kind, but a very precise, high-resolution brush edge will look softer than a pixel-perfect selection would.
- Dehaze's transmission map is smoothed with a plain Gaussian blur, not a real edge-aware guided filter (see docs/CHANGELOG.md) — expect a faint halo around strong, hard edges at high strength. There's also no sky segmentation, so a hazy sky can receive more correction than the rest of the scene.
- Magic Wand's Sample size (Point, 3×3, 5×5, 11×11) is in pixels of the image the mask is built on, so on a proxy preview a 5×5 sample covers a larger area of the photo than at full resolution. Point keeps the original single-pixel, neighbour-to-neighbour fill; the averaged sizes compare every pixel with the sampled average (OpenCV's fixed-range fill).
- The histogram clipping-warning overlay flags a channel at 0/255, or within the "Clipping margin" of them, and — like the Masks/Repair preview overlays and the ICC gamut warning — is a locally-computed debug preview over the already backend-processed `edited_image`, not a separate network round trip.
- Vibrance's skin-tone protection is a fixed hue-range heuristic (~25 degrees, no adjustable center/width), not real face or skin detection — an unrelated orange/tan subject (wood, terracotta, autumn leaves) gets the same reduced boost a genuine skin tone would.
- `estimate_white_balance()`'s closed-form correction is derived from a linearized (cross-term-dropped) version of `apply_temperature_tint()`'s own formula — exact for a pixel already close to neutral (the intended use: click something that's *supposed* to be gray or white), but only an approximate starting point for a pixel with a severe color cast, the same caveat darktable's own picker carries ("often it works really well," not "always exactly").
- `correct_perspective()`'s two axes (Vertical/Horizontal Keystone) are applied as independent sequential warps rather than one combined homography solve — visually indistinguishable from a combined solve in practice, but means the "untouched" edge of a single-axis correction isn't perfectly pixel-identical to the source (a homography that fixes only 2 of a quadrilateral's 4 corners doesn't leave the line between them pointwise-fixed), just far less changed than the edge actually being corrected.
- Applying a saved custom preset across a Batch Processor run intentionally skips that preset's `mask_layers`/`removal_mask_layers`/overlay layer — see `app/recipe_options.py::recipe_to_options()`'s docstring for why (they're tied to one specific photo's pixel coordinates) — so a preset built around a local mask adjustment will only carry its global tone/color/detail/lens settings into the batch, not the local one. A Free-form crop box is excluded from batch application for the same reason, even though — unlike masks/layers — it's stored size-independently (as fractions) and so could in principle be translated per-photo if a future caller had each batch photo's own dimensions on hand at conversion time.
- The draggable free-form crop rectangle only offers two corner handles (top-left, top-right's and bottom-left's positions follow implicitly) plus drag-the-body-to-move — there's no independent handle on every corner/edge, and a resize is clamped to a minimum 2% of the frame in each dimension so the box can't be dragged down to zero-size.
- The tone curve editor moves an extra point that's added or dragged onto another point's X to the nearest free X, so two points never share one; `app.recipe_options.merge_curve_points()` still drops a duplicate X server-side for curves from other sources (a hand-edited preset).
- Post-Crop Vignette and the Lens tab's corrective vignette are independent controls that both run in the same pipeline (Lens's before Crop, Effects' after) — using both at once stacks their darkening/brightening rather than one overriding the other, which is the expected behavior for two conceptually different tools but can look like "double vignetting" if not intentional.
- "Color Noise" has no effect while Denoise is 0 (the slider is disabled to make this visible) — it's a modifier to Non-Local Means' existing `hColor` parameter, not an independent chroma-only denoise pass, so it can't run without Denoise also being nonzero.
- **AI Upscale (`app/ai_upscale.py`) is a deliberate, explicitly-scoped learned-processing exception to this app's otherwise CPU-first, no-ML-dependency design** — denoise, dehaze and lens corrections remain classical, and the edit pipeline still runs on CPU. Real-ESRGAN was chosen for upscaling because it is opt-in and occasional. It is entirely optional: `torch`/`realesrgan` are never installed by `requirements.txt`/CI, only by `requirements-gpu.txt`; `app.ai_upscale.ai_upscale_available()` reports `False` when they are not importable, and the Export tab explains how to get them. Separately, `app.gpu_export` optionally uses CUDA via PyTorch for final resize/output sharpening; it is opt-in, lazy-loaded, keeps encoding on CPU, and logs/falls back to the CPU transform if CUDA is unavailable or fails. When available, AI Upscale auto-detects CUDA and uses float16 on GPU / float32 on CPU, and is applied last in `app/pipeline.py::run_pipeline()`, after Effects (on export, not on every preview).
