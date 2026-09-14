# Color Grading Workflow

## Goal

Add deliberate color looks while preserving natural skin tones and avoiding destructive source edits.

## Steps

1. Start from an unchanged source image.
2. Apply exposure and white balance first.
3. Adjust saturation conservatively.
4. Add a named preset only after the neutral correction is acceptable.
5. Compare the result at 100% zoom and in a reduced preview.
6. Export a delivery copy and keep the source unchanged.

## Current checkpoint

The current implementation supports exposure, white balance, saturation, contrast, reusable presets, and preservation of embedded ICC profiles during backend processing and export. Dedicated shadow/highlight curves, profile conversion, and LUT support are planned next.

## Extension guide

1. Add one deterministic operation to `PhotoProcessor`.
2. Add a focused test with known pixel expectations.
3. Add the operation to a preset only after the standalone behavior is verified.
4. Document the intended look and safe parameter range here.
