# Highlights and Shadows Workflow

## Goal

Recover or deepen selected tonal regions while protecting the opposite range.

## Steps

1. Set exposure globally.
2. Use `--highlights` with a negative value to reduce bright areas.
3. Use `--shadows` with a positive value to lift dark areas.
4. Use small adjustments first, such as `-15` and `20`.
5. Review bright edges and dark texture before export.

## CLI

```powershell
python -m app --input input.jpg --output output.jpg --highlights -15 --shadows 20
```

## Validation

The processor uses luminance-weighted masks and clips channel values to 0-255. Extreme values should be tested before being used in batch jobs.