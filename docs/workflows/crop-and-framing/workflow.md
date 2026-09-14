# Crop and Framing Workflow

## Goal

Create deliberate compositions and aspect-ratio deliverables without changing the source file.

## Steps

1. Use explicit pixel coordinates for a precise crop.
2. Or use a centered aspect-ratio crop for common formats.
3. Confirm the resulting dimensions before export.
4. Apply sharpening after the final crop when needed.
5. Save to a separate delivery path.

## CLI

Explicit crop:

```powershell
python -m app --input input.jpg --output crop.jpg --crop 100 50 1800 1200
```

Centered square crop:

```powershell
python -m app --input input.jpg --output square.jpg --aspect 1 1
```

## Validation

Crop coordinates must form a non-empty region. Aspect-ratio values must be positive. Both operations preserve RGB output and are tested independently.