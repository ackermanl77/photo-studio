"""The Streamlit UI, split out of app/frontend_unified_app.py (F14).

app/frontend_unified_app.py stays the only Streamlit entry point: it sets the page up, opens the
photo and composes the panels defined here. Every module in this package is imported by that
script (so `st.cache_data` functions are defined once per process, not once per rerun); none of
them runs anything on import.

- state.py      session_state conventions: recipe capture/restore, deferred reruns, commit guards,
                tab visibility, section reset buttons and "modified" indicators
- widgets.py    widgets generated from the app.controls registry (F15)
- context.py    EditorContext, the values one script run passes between panels
- selections.py the Masks/Repair selection stack and its cached mask builders
- layers.py     the Layers stack and its rasterization
- processing.py the two-stage backend calls, proxy preview, full-resolution producers
- project_io.py Save/Open Project, opening a Library photo or a launch file
- theme.py      the page stylesheet (static/theme.css) and its CSS variables (F12)
- canvas.py     the canvas column (view, scopes, guides)
- export_panel.py the Export tab and the Save footer
- tabs/         one module per tool tab
"""
