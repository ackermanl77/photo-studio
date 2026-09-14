# All Controls Below Preview Workflow

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\python.exe -m streamlit run app/frontend_center.py
```

## Layout

After a photo is loaded, the page is arranged as:

1. Original image and edited image at the top.
2. A full-width editing workspace below the images.
3. Tone, Color, Detail, Finish, Professional tools, and Export tabs below the previews.

There are no post-upload controls in the right sidebar. Layers, healing, catalog, presets, curves, tone, color, detail, and export controls all appear below the original and edited images.
