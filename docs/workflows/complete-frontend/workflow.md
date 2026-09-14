# Complete Frontend Workflow

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\python.exe -m streamlit run app/frontend_complete.py
```

## Added features

- All editing controls below the image previews
- Preview zoom from 25% to 400%
- Horizontal and vertical preview panning
- Uploadable grayscale masks
- Mask inversion
- Mask feathering
- Optional mask overlay preview
- Image layers
- Layer opacity
- Normal, multiply, and screen blending
- Web, Social, Print, and Archive export profiles
- Maximum-size export profiles
- JPEG, PNG, and TIFF delivery support

## Workflow

1. Upload a photo.
2. Tune Look, Tone, Color, Detail, and Curves.
3. Add a mask and inspect its overlay if local selection is needed.
4. Add a layer and choose its opacity and blend mode.
5. Use Inspect to zoom and pan through fine detail.
6. Select an export profile.
7. Download the delivery file.

## Current limits

The application remains CPU-first. Full 16-bit/float processing, multi-layer project persistence, and ICC profile conversion are still separate future migrations.
