# Contrast Workflow

## Goal

Control separation between light and dark areas without changing the image dimensions.

## Steps

1. Correct exposure first.
2. Use `--contrast 1.0` as neutral.
3. Increase contrast above `1.0` for stronger separation.
4. Use values below `1.0` for a softer result.
5. Check skin tones and shadow detail at 100% zoom.

## CLI

```powershell
python -m app --input input.jpg --output output.jpg --exposure 1.05 --contrast 1.15
```

## Validation

Add a fixture test whenever contrast behavior changes. Keep contrast independent from the highlights/shadows operation so both can be tuned separately.