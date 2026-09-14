# Batch Release Workflow

## Goal

Process a group of source images with a named preset and delivery profile while preserving a report for failures.

## Steps

1. Select input image paths.
2. Choose a preset or `None`.
3. Choose `Web`, `Social`, `Print`, or `Archive` output.
4. Choose a filename template such as `{stem}-edited{suffix}`.
5. Run `process_files()`.
6. Review the `BatchResult` report for successful and failed files.
7. Build a ZIP with `make_zip()`.
8. Publish only after checking representative exports.

## Guarantees

- One corrupt or missing file is reported without stopping the remaining batch.
- Originals are never overwritten.
- Output names are generated from explicit templates.
- The ZIP contains only successful outputs.
