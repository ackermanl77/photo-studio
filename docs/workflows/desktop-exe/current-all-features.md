# Current All-Features Desktop Package

> **Histórico:** este archivo documenta una compilación pasada (carpeta `dist-final`, 84 tests). La release actual vive en `PhotoStudio-0.1.0-Windows/` — ver [`docs/PROJECT_STATUS.md`](../../PROJECT_STATUS.md) para el estado vigente y el conteo de tests, que debe verificarse corriendo `pytest -q`.

## Executable

```text
C:\Users\MELEJ\OneDrive\photo_professional_tool\dist-final\PhotoStudio.exe
```

This package was rebuilt from the current source after all implemented feature batches and the latest UI layout changes.

## Verification

- Application modules compiled successfully.
- Full global test suite: 84 passed.
- PyInstaller created the executable successfully.

Copy the complete `dist-final` folder when distributing. Microsoft Edge WebView2 Runtime is required for the native window.
