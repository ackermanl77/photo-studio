# CLAUDE.md

Guidance for Claude Code in this repository. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
(how the app is put together, and why) and [docs/MODULE_MAP.md](docs/MODULE_MAP.md) (what each module does).

## What this is

A CPU-first Python photo editor: Streamlit UI (`app/frontend_unified_app.py` + `app/ui/`), Flask
processing backend (`app/backend.py`), optional pywebview/PyInstaller desktop build. No ML in the edit
pipeline; the only optional exceptions need PyTorch from the GPU pack: AI Upscale
(`app/ai_upscale.py`), AI masks (`app/ai_masks.py`) and the opt-in CUDA export resize/sharpen
(`app/gpu_export.py`, CPU fallback).

## Commands

```powershell
python -m pip install -r requirements-dev.txt          # + requirements-raw.txt / -desktop.txt (optional)
python -m streamlit run app/frontend_unified_app.py    # the only Streamlit entry point
python -m compileall -q app
python -m pytest -q -n auto                            # run it yourself; docs report stale totals
python -m pytest -q -m "not slow"                      # skip the AppTest frontend tests
$env:PHOTO_PERF_TESTS="1"; python -m pytest -q -m perf
$env:PHOTO_BROWSER_TESTS="1"; python -m pytest -q tests/test_browser_smoke.py
$env:PHOTO_REGENERATE_GOLDEN="1"; python -m pytest -q tests/test_parity_golden.py   # after an intended algorithm change
python -m app --input photo.jpg --output out.jpg --exposure-ev 0.5                 # CLI over run_pipeline()
.\scripts\build_release.ps1 -Version "0.1.0" [-GpuPack -ExtraIndexUrl ...] [-Installer] [-Sign]
```

The dev venv is `.venv` (never pip install into it ad hoc). CI and release workflows: see
docs/ARCHITECTURE.md ("CI and releases") and docs/PROJECT_STATUS.md ("Verification", "Release").

## Rules most changes need

- **The frontend never edits images in-process.** Edits go through `/process` to `app/backend.py`;
  locally `PhotoProcessor` only decodes the upload, builds mask geometry and counts faces.
- **One pipeline order**, in `app/pipeline.py::run_pipeline()`; the backend, CLI and batch paths all
  call it. Never re-implement the steps elsewhere.
- **One control registry**, `app/controls.py::CONTROLS`. To add a control: declare it there, render it
  with `render_control()` in a section function, record it in `ctx.values`. Any other new `/process`
  option goes in `edit_schema._OTHER_OPTION_SPECS` with its stage, or the backend rejects it.
- **Widget keys are saved state** (Undo, presets, projects, tests): never rename one
  (`tests/test_ui_structure.py` pins them).
- **Every tab runs on every rerun.** Never skip rendering a widget (its state is lost); skip only
  display work with `app.ui.state.tab_visible()`. Use `request_rerun()`, never a mid-script `st.rerun()`.
- **Setting a widget from outside it** (Undo, Reset, presets, editor commits) goes through
  `st.session_state["_pending_restore"]` + `request_rerun()`; never write an instantiated widget's key.
- **On-canvas editors** (`components.html()`, static files in `app/ui/static/editors/`) commit through a
  hidden `st.text_area` + `consume_new_commit()`; a JS-driven `st.slider` does not reach the server. An
  editor that works on the photo passes `canvas_label`/`slot_id` to `show_component()` and fits its
  background to `editor_box()`, so it is drawn over the photo; its commit field stays in the sidebar.
- **List widgets** (masks, layers) use stable UUID keys, and `break` out of the loop right after a
  Move/Duplicate/Remove.
- **Every `st.cache_data` has `max_entries`**; large inputs are underscore-prefixed (unhashed) args.
- **Text**: wrap user-visible strings in `app.i18n.t()` and add the Spanish entry in `app/i18n_es.py`;
  keys, option values and tab values stay English. Colours only as `var(--ps-...)` from `app/ui/theme.py`.
- **Logging**: modules use `app.observability.get_logger()`; only entry points call `configure_logging()`.

## Testing conventions

- Per-user files live under `app.config.user_data_dir()`; `tests/conftest.py` points it at a temp folder
  and blanks `PHOTO_BACKEND_TOKEN`. The suite is xdist-safe: no fixed ports, no writes to the source tree.
- Test images are generated (`tests/synthetic_assets.py`); don't add third-party photos.
- `tests/test_frontend_unified_app.py` drives the app with AppTest (`open_tab()` opens a tab first).
  AppTest can't run iframe JS or fragment-only reruns: drive the hidden commit field, use the Node tests
  (`tests/js_fake_dom.js`) or the browser smoke test.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost), then `graphify export wiki` -- the git hooks rebuild graph.json after each commit/checkout but never the wiki.
- graphify-out/ is gitignored: every clone builds its own graph (`/graphify`), and the git hooks (`graphify hook install`) keep it current. Do not commit it or suggest skipping the rebuild or its hooks.
- Community names live in graphify-out/.graphify_labels.json (curated); a community a rebuild creates or reshapes gets a hub-name fallback in `.graphify_labels.pending.json` (e.g. a bare file or test name). When the report shows such names, rename them there (2-5 plain words) and run `graphify cluster-only .`.
