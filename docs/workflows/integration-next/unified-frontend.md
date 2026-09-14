# Unified Frontend Workflow

## Launch

```powershell
streamlit run app/frontend_unified_v2.py
```

This entry point imports the mature frontend first, so it retains upload, adjustments, presets, curves, masks, recipes, export, and backend processing. It then adds the professional workspace panel.

## Added integration

- Add an image layer over the mature processed result.
- Choose layer opacity.
- Choose normal, multiply, or screen blend mode.
- Apply a CPU healing patch using source and target boxes.
- Save ratings and comma-separated tags to the local SQLite catalog.
- Search catalog records.
- Compare mature result and layered result.
- Download the layered PNG.

The existing `app/frontend.py` remains the stable default. Use this entry point while the unified panel is being exercised against real workflows.
