# Frontend Batch Workflow

> **Histórico:** los pasos de abajo no coinciden con la UI actual. No hay `st.sidebar` en la app (`frontend_unified_app.py` fija `initial_sidebar_state="collapsed"` y nunca lo usa) — Batch Processor es una opción dentro de la pestaña "☰ Batch & Library", con un botón "Run Batch" y luego "Download ZIP" (no "Download edited batch"), y no tiene control de dimensiones máximas. Ver `docs/PROJECT_STATUS.md` para el flujo vigente.

## Goal

Apply the selected preset to multiple photos and download one ZIP archive from the visual frontend.

## Steps

1. Start Photo Studio normally.
2. Open the `Batch` section in the sidebar.
3. Upload multiple supported images.
4. Choose a preset and delivery format.
5. Set maximum dimensions if the batch should be resized.
6. Click `Download edited batch`.
7. Review representative files from the ZIP before publishing.

## Notes

Batch processing uses the same Flask `/process` endpoint as single-image editing. Originals are never overwritten. The current batch flow applies presets and export sizing; per-image brush masks are intentionally handled through the single-image workflow.
