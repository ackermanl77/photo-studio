# Cached Frontend Workflow

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app/frontend_cached.py
```

The development instance runs at `http://localhost:8517`.

## Behavior

- The first recipe/source combination reports a cache miss.
- Repeating the same settings reports a cache hit.
- Changing any adjustment or preset creates a new cache key.
- The preview is bounded and stored under `%USERPROFILE%\\.photo-studio\\previews`.
- The download path still processes a fresh full-resolution export.
