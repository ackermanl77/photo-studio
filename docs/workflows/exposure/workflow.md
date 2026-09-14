# Exposure Workflow

## Goal

Correct overall brightness before applying contrast, color, or detail edits.

## Steps

1. Start with the original image.
2. Use `--exposure 1.0` as the neutral value.
3. Increase above `1.0` to brighten or use a value below `1.0` to darken.
4. Inspect highlights after the adjustment.
5. Follow with highlights/shadows control when only part of the tonal range needs correction.

## CLI

```powershell
python -m app --input input.jpg --output output.jpg --exposure 1.1
```

## Validation

The output keeps the original dimensions and remains an 8-bit RGB image. Values are clipped to the valid 0-255 channel range.