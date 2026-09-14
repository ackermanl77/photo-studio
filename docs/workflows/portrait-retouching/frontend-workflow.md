# Portrait Retouching Frontend Workflow

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\python.exe -m streamlit run app/frontend_portrait.py
```

Use the global interpreter if `.venv` is not being used:

```powershell
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app/frontend_portrait.py
```

## Features

- CPU skin-tone mask generation
- Conservative texture-preserving smoothing
- Optional face detection overlay
- Optional skin-mask overlay
- Adjustable retouching strength
- PNG result export

Face detection gracefully returns no detections when the installed OpenCV build does not expose Haar cascade support; skin masking and smoothing remain available.
