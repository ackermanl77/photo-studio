# Application Diagnostics Workflow

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app/frontend_diagnostics.py
```

## Checks

The diagnostics page checks:

- Python version
- Pillow
- NumPy
- OpenCV
- Streamlit
- Flask
- TIFF support
- Optional RAW support
- Optional 7-Zip availability
- Writable Photo Studio storage

Use it before editing on a new computer or before creating a desktop release.
