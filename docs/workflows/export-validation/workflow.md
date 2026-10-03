# Export Validation Workflow

## Goal

Produce delivery files with predictable format, quality, and destination behavior.

## Steps

1. Finish the edit pipeline.
2. Select a destination extension that matches the delivery requirement.
3. Call `app.export.write_export(image, path, {"quality": 92})` (format follows the extension; see `encode_export()` for every setting).
4. Confirm the destination exists.
5. Re-open the file and verify its format and dimensions.
6. Keep the source file unchanged.

## Current rules

- JPEG and JPG outputs are converted to RGB and written with configurable quality.
- Other supported formats retain the current image mode and use optimized output where available.
- Destination folders are created automatically.
- Quality values are clamped to the safe range 1-100.

## Extension guide

1. Add format-specific options in `app/export.py`.
2. Add a test that re-opens the exported file.
3. Validate metadata behavior before adding metadata preservation.
4. Add color-profile handling before promising print-critical output.
