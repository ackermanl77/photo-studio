# Advanced Masking Frontend Workflow

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\python.exe -m streamlit run app/frontend_masking.py
```

## Steps

1. Upload a photo.
2. Open `Mask layers`.
3. Choose Luminance, Color, Horizontal gradient, Vertical gradient, or Radial.
4. Name the mask and tune opacity, inversion, and feathering.
5. Review the grayscale mask preview.
6. Open `Local adjustment` and tune exposure, contrast, and saturation.
7. Toggle the mask overlay to inspect the selected region.
8. Download the masked result.

The mask engine is CPU-friendly and uses grayscale selection values so gray pixels receive partial-strength edits.
