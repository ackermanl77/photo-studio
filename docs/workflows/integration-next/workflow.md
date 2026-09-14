# Integrated Feature Workbench

## Launch

```powershell
streamlit run app/integrated_workbench_next.py
```

## Current integrated features

- Base image preview
- Added image layer
- Layer opacity
- Normal, multiply, and screen blending
- CPU healing patch with source and target boxes
- Local SQLite catalog metadata
- Ratings from 0 to 5
- Comma-separated tags
- Catalog text search
- Bounded preview rendering
- Composite PNG download

## Workflow

1. Upload a base image.
2. Add an optional overlay layer and choose opacity/blend mode.
3. Use source/target boxes for a small healing repair.
4. Save rating and tags to the local catalog.
5. Search catalog metadata from the sidebar.
6. Download the composite.

## Integration boundary

This workbench is intentionally separate from the mature `app/frontend.py` while the larger Layers API is stabilized. The existing frontend and Flask backend remain unchanged and continue to pass their regression suite.

Next steps are to merge these controls into the main frontend, send layer descriptors through Flask, and persist layers inside `.photoproject` archives.
