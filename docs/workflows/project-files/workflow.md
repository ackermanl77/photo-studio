# Photo Project Files Workflow

## Goal

Save a complete editing session as one portable `.photoproject` file.

## What it contains

- Original source image
- Current edit recipe
- Optional object-removal mask
- Optional local-adjustment mask
- Project manifest and format version

## Steps

1. Upload a source image.
2. Adjust the image and paint or upload masks if needed.
3. Click `Save Photo Studio project`.
4. Keep the `.photoproject` file with the source workflow.
5. Reopen it through `Open Photo Studio project`.
6. Review the restored recipe, masks, and preview before exporting.

## Design notes

The archive is a ZIP container with a JSON manifest. Source pixels remain unchanged. The format is versioned so future project migrations can be added without breaking existing sessions.
