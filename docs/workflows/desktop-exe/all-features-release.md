# All-Features Desktop Release

> **Histórico:** este archivo documenta una compilación pasada (carpeta `dist-all-features`, 81 tests). La release actual vive en `PhotoStudio-0.1.0-Windows/` — ver [`docs/PROJECT_STATUS.md`](../../PROJECT_STATUS.md) para el estado vigente y el conteo de tests, que debe verificarse corriendo `pytest -q`.

## Executable

```text
C:\Users\MELEJ\OneDrive\photo_professional_tool\dist-all-features\PhotoStudio.exe
```

This build uses the global Python environment, targets `frontend_unified_app.py`, and includes the current application modules through the PyInstaller app package.

Copy the complete `dist-all-features` folder when distributing. Microsoft Edge WebView2 Runtime remains required for the native window.

Verification before build:

```text
81 passed
```
