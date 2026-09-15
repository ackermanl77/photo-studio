# Unified Application Workflow

> **Histórico:** la sección "Workspaces" de abajo describe un selector de workspace (Edit/Layers/Masks/Library/Batch/Portrait/Export) de una iteración anterior de la UI, y la frase "Specialized frontends remain available" ya no es correcta — esos frontends especializados fueron consolidados y **ya no existen** (ver la nota al inicio de [`docs/workflows/README.md`](../README.md)). La app actual usa pestañas (`st.tabs()`), no un selector de workspace; la lista y el orden vigente de pestañas está en `docs/PROJECT_STATUS.md`.

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app/frontend_unified_app.py
```

## Workspaces (histórico, ver nota arriba)

The Workspace selector provided:

- Edit
- Layers
- Masks
- Library
- Batch
- Portrait
- Export

This was the central entry point for the major application workflows at the time.
