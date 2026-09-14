# Windows Desktop EXE Workflow

## Goal

Launch Photo Studio in a native desktop window instead of asking the end user to open a localhost URL.

## Build prerequisites

Use Windows with the project virtual environment activated. PyWebView's `edgechromium` backend requires Microsoft Edge WebView2 Runtime, which is present on most current Windows installations.

Install desktop build dependencies:

```powershell
python -m pip install -r requirements-desktop.txt
```

## Build

```powershell
.\scripts\build_release.ps1 -Version "0.1.0"
```

The script runs PyInstaller into a scratch `dist-release\` folder and then copies the result into a versioned release folder at the project root. The executable to run is:

```text
PhotoStudio-0.1.0-Windows\PhotoStudio.exe
```

(substitute whatever `-Version` you passed for `0.1.0`). Copy the entire `PhotoStudio-<version>-Windows` folder when distributing it, not just the `.exe` — it needs the files packaged alongside it.

The release script disables UPX, creates a compressed archive (`PhotoStudio-<version>-Windows.7z`, plus a `.sha256` checksum file) when 7-Zip is installed, and leaves the unpacked folder either way.

## Run

Double-click `PhotoStudio.exe` inside the `PhotoStudio-<version>-Windows` folder produced above. It opens a native Photo Studio window and starts the embedded Streamlit frontend (`app/frontend_unified_app.py`) on a loopback port. It does **not** start the Flask backend (`app/backend.py`) — the unified frontend processes images in-process by calling `PhotoProcessor` directly, and never calls the Flask API. See "Architecture" in `docs/PROJECT_STATUS.md`.

The loopback frontend service is an implementation detail and is not shown as a browser tab to the user.

Since the executable is unsigned by default, Windows SmartScreen may show "Windows protected your PC" the first time it runs — click "More info" then "Run anyway". If Defender flags it (a common PyInstaller false positive), add an exclusion or unblock the file via its file Properties.

To run the app from source without building an `.exe` at all (useful while iterating), run `python app/desktop.py` from the project root instead — it launches the same embedded-Streamlit + native-window flow directly from the source tree.

## Distribution checklist

1. Test on a clean Windows account.
2. Confirm WebView2 Runtime is installed.
3. Copy the entire `PhotoStudio-<version>-Windows` folder, not only the executable.
4. Keep the original source images unchanged.
5. Add a code-signing certificate before distributing publicly (removes the SmartScreen warning above).