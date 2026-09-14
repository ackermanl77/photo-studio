# Release Preflight Workflow

## Current status

There is no dedicated preflight UI or CLI command today. `app/frontend_release.py`, referenced by earlier drafts of this doc, does not exist. The checks themselves are implemented and tested in `app/release_preflight.py` (`run_preflight`, `preflight_ok`) — only the UI wrapper around them was never built.

## Run the checks

Until a CLI/UI wrapper exists, run the checks from the project root with a one-line Python call:

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -c "from app.release_preflight import run_preflight, preflight_ok; checks = run_preflight('.', run_tests=True); [print(c.name, c.status, c.message) for c in checks]; print('READY' if preflight_ok(checks) else 'NOT READY')"
```

## Checks

The checks verify:

- Unified frontend exists (`app/frontend_unified_app.py`)
- Desktop launcher exists (`app/desktop.py`)
- PyInstaller spec exists (`PhotoStudio.spec`)
- Requirements files exist
- Required Python packages are importable
- The full test suite passes, when `run_tests=True`

Build the desktop package only after the status reports `READY`.
