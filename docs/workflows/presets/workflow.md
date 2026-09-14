# Presets Workflow

## Goal

Create repeatable looks without duplicating low-level edit calls.

## Current implementation

Presets live in `app/presets.py` and are named pipelines:

- `portrait-natural`: restrained exposure, contrast, saturation, denoise, and sharpening
- `travel-clean`: brighter contrast and color for outdoor images
- `product-neutral`: controlled color with crisp detail
- `cinematic-soft`: warm/cool balance, a gentle curve, restrained contrast, and reduced saturation

## Steps

1. Start with a `PhotoProcessor` for one image.
2. Call `apply_preset(processor, "portrait-natural")`.
3. Inspect the result before saving.
4. Save with `PhotoProcessor.save()` or `save_for_delivery()`.
5. Add a test using a small fixture image when changing a preset.

## CLI checkpoint

Use the preset option after the CLI wiring is complete:

```powershell
python -m app --input input.jpg --output output.jpg --preset portrait-natural
```

Presets can be combined with the other controls, but edits are cumulative, so inspect the result before exporting.

## Extension guide

1. Add a private function that receives and returns `PhotoProcessor`.
2. Register it in `PRESETS`.
3. Add the name to the CLI choices.
4. Add a test that verifies the output changes and remains valid.
5. Keep presets deterministic and CPU-friendly.
