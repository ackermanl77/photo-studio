# Batch Processing Frontend Workflow

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app/frontend_batch.py
```

## Steps

1. Upload multiple photos.
2. Choose a preset.
3. Choose Web, Social, Print, or Archive export.
4. Set a filename template.
5. Click `Process batch`.
6. Review successful and failed files in the report.
7. Download the successful results as a ZIP.

A failed file does not stop the rest of the batch.
