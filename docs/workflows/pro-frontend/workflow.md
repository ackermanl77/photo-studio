# Photo Studio Pro Frontend

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\python.exe -m streamlit run app/frontend_pro.py
```

The current live development instance runs at `http://localhost:8507`.

## New improvements

- Uses the Flask backend for every image-processing request.
- Keeps all controls below the original and edited previews.
- Adds a split before/after inspection slider.
- Adds source and edited RGB histograms.
- Shows clipped-shadow and clipped-highlight percentages.
- Adds maximum export width and height.
- Adds optional metadata stripping.

## Test

1. Upload a real image.
2. Adjust exposure, contrast, color, detail, or curves.
3. Move the Preview split slider.
4. Compare source and edited histograms.
5. Check clipping percentages before export.
6. Download JPEG or PNG output.
