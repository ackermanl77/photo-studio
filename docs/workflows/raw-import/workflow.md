# RAW Import Workflow

## Goal

Support camera RAW files without changing the existing JPEG/PNG workflow.

## Optional installation

RAW support uses LibRaw through `rawpy` and is optional:

```powershell
python -m pip install -r requirements-raw.txt
```

The normal `requirements.txt` installation remains sufficient for JPEG, PNG, WEBP, and TIFF editing.

## Supported extensions

The adapter recognizes `.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.rw2`, `.orf`, `.dng`, `.pef`, and `.srw`.

## Steps

1. Identify the camera formats that matter first.
2. Add a RAW decoder as an optional dependency because it can be platform-sensitive.
3. Decode RAW data with camera white balance into an RGB working image.
4. Preserve camera metadata where the decoder supports it.
5. Run the normal `PhotoProcessor` pipeline on the decoded image.
6. Validate output dimensions, color mode, and export format with fixtures.

## Current checkpoint

RAW decoding is implemented as an optional path. The existing processor still uses an 8-bit RGB working buffer after decoding; a future 16-bit/float pipeline is required for maximum highlight recovery and print-critical work.

## Extension guide

1. Test `rawpy` and its LibRaw runtime requirements on the target Windows machines.
2. Add representative camera fixtures before documenting camera-specific support.
3. Upgrade the working buffer to 16-bit or float before promising print-critical RAW output.
