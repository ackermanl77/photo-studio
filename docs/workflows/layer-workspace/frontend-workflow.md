# Layer Workspace Frontend Workflow

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\python.exe -m streamlit run app/frontend_layers.py
```

## Steps

1. Upload a background image.
2. Add a PNG, JPG, or WEBP layer.
3. Give the layer a name.
4. Choose opacity and blend mode.
5. Add an optional grayscale layer mask.
6. Use Up, Down, Duplicate, or Delete to manage the stack.
7. Compare Background and Composite previews.
8. Download the layer workspace JSON or flattened composite.

The Background layer is protected from deletion. Layer images are resized to the background dimensions, and all compositing remains CPU-friendly.
