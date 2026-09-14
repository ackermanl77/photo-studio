# Frontend Batch Workflow

## Goal

Apply the selected preset to multiple photos and download one ZIP archive from the visual frontend.

## Steps

1. Start Photo Studio normally.
2. Open the `Batch` section in the sidebar.
3. Upload multiple supported images.
4. Choose a preset and delivery format.
5. Set maximum dimensions if the batch should be resized.
6. Click `Download edited batch`.
7. Review representative files from the ZIP before publishing.

## Notes

Batch processing uses the same Flask `/process` endpoint as single-image editing. Originals are never overwritten. The current batch flow applies presets and export sizing; per-image brush masks are intentionally handled through the single-image workflow.
