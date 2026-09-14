# Batch Processing Workflow

## Goal

Apply one preset consistently to every supported image in a folder.

## Steps

1. Create an input folder containing `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, or `.webp` files.
2. Create an empty output folder.
3. Choose a named preset.
4. Run the batch command.
5. Review the output count and open representative images.
6. Keep the original folder unchanged so edits remain reversible.

## CLI checkpoint

```powershell
python -m app --input-dir .\photos --output-dir .\edited --preset travel-clean
```

## Extension guide

1. Keep extension filtering in `app/batch.py`.
2. Sort input paths for reproducible processing order.
3. Preserve source filenames unless a collision policy is explicitly added.
4. Add progress reporting only after the basic output contract is tested.
5. Add a failure report for corrupt files before enabling unattended jobs.
