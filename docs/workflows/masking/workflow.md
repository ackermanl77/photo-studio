# Masking Workflow

## Goal

Apply an adjustment to selected pixels while preserving the unmasked area.

## Steps

1. Create a grayscale mask with black for protected areas and white for edited areas.
2. Keep the mask dimensions identical to the image.
3. Choose local exposure, contrast, and saturation values.
4. Review the mask edge for halos or abrupt transitions.
5. Export only after checking the original and masked regions.

## CLI

```powershell
python -m app --input input.jpg --output local-edit.jpg --mask subject-mask.png --mask-exposure 1.15 --mask-contrast 1.08
```

Soft gray mask values create partial-strength edits.