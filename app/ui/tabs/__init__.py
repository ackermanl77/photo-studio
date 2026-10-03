"""One module per tool tab (F9 grouping): basic, color (curves & colour), detail, geometry, local
(masks, repair, portrait), layers, effects, export (app/ui/export_panel.py) and presets & library.

Each exposes section functions taking the run's EditorContext; app/frontend_unified_app.py places
them into the st.tabs() containers. Section functions only render widgets and record values in
`ctx` -- the pipeline order lives in app.pipeline.run_pipeline() alone, and the tab order is
deliberately independent of it.
"""
