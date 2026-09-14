# Texture and Clarity Workflow

## Goal

Improve perceived detail while keeping the edit CPU-friendly and reviewable.

## Steps

1. Denoise before adding detail when the source is noisy.
2. Use `--texture` for fine local detail.
3. Use `--clarity` for broader midtone contrast.
4. Start around `texture=20` and `clarity=10`.
5. Inspect hair, fabric, and skin at 100% zoom.

## CLI

```powershell
python -m app --input input.jpg --output output.jpg --denoise 3 --texture 25 --clarity 10
```

## Validation

Both controls are clamped to -100 through 100. Texture uses a blur/detail difference and clarity uses contrast, with no GPU dependency.