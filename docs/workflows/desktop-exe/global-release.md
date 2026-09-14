# Global Python Release Workflow

The release script now prefers the configured global Python interpreter and falls back to `.venv` only if the global interpreter is unavailable. This part still reflects `scripts/build_release.ps1` as it is today. For the current release output folder name and verified state, see [`docs/PROJECT_STATUS.md`](../../PROJECT_STATUS.md) rather than assuming a specific `dist-*` path.

## Build

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\scripts\build_release.ps1 -Version "0.1.0"
```

The preferred interpreter is:

```text
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

The resulting package is created under the release output directory and includes the current unified desktop launcher.
