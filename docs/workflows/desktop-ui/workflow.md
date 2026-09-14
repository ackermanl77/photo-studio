# Desktop UI Workflow

## Goal

Provide a local desktop editor that previews edits without requiring a GPU or cloud service.

## Steps

1. Keep image processing in `app/` and keep UI code separate.
2. Load a preview-sized copy for responsive controls.
3. Keep the original image available for final export.
4. Apply controls through the same `PhotoProcessor` methods used by the CLI.
5. Show before/after previews and export status.
6. Add undo/redo only after the non-destructive pipeline is stable.

## Recommended next slice

Build a small Tkinter or PySide interface with open, preset selection, preview, and export controls. Keep the first UI CPU-friendly and avoid introducing a GPU dependency.

## Extension guide

1. Choose the UI toolkit and add it as an optional dependency.
2. Add an application entry point separate from `app/__main__.py`.
3. Test processing logic independently from widget events.
4. Add a manual smoke-test checklist for common image formats.
