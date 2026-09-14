# Preset Exchange Workflow

## Goal

Move custom editing looks between machines as portable JSON without copying image files.

## Steps

1. Open the preset exchange frontend or use `PresetStore` and `export_presets()`.
2. Select the presets to share.
3. Download `photo-studio-presets.json`.
4. Import the JSON on another machine.
5. Existing presets are preserved by default.
6. Enable overwrite only when intentionally replacing a local look.

Preset files contain adjustment recipes, not source images.
