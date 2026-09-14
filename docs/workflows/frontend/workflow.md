# End-User Frontend Workflow

## Goal

Edit one photo visually in a local browser interface without requiring a dedicated GPU.

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\Activate.ps1
streamlit run app/frontend.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

The frontend automatically starts the Flask processing backend on `http://127.0.0.1:8502`; no second command is needed for normal use.

## Editing steps

1. Upload a JPG, PNG, WEBP, or TIFF in the Source panel.
2. Choose a preset if you want a starting look.
3. Tune exposure, contrast, saturation, highlights, shadows, temperature, and tint.
4. Open Detail for texture, clarity, denoise, and sharpening.
5. Open Curves for a five-point RGB tone curve.
6. Use Crop and framing for a centered delivery ratio.
7. Add a grayscale Object removal mask to repair a small unwanted object.
8. Add a grayscale Local adjustment mask to limit exposure, contrast, or saturation changes to selected pixels.
9. Or choose a brush mask target and paint directly on the black canvas.
10. Use Undo and Redo while refining the recipe.
11. Save a JSON edit recipe when you want to reuse the settings.
12. Save a `.photoproject` to preserve source, recipe, and masks together.
13. Compare Original and Edited preview.
14. Choose JPEG or PNG and download the delivery copy.

## Mask rules

- Black protects pixels.
- White applies the full operation.
- Gray applies a partial operation.
- Masks are resized to the uploaded image dimensions with nearest-neighbor sampling when needed.
- Brush masks use white strokes over a black protected background.

## Design note

The frontend starts each rerun from the original upload and reapplies the current controls. This keeps slider changes reversible and prevents edits from accumulating accidentally.