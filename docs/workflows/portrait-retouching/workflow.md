# Portrait Retouching Workflow

## Goal

Improve portrait images carefully while retaining realistic texture and identity.

## Steps

1. Work from a duplicate or preview copy.
2. Correct exposure and white balance before retouching.
3. Apply light denoise only when visible noise requires it.
4. Use restrained sharpening after resizing or denoising.
5. Inspect eyes, hair, and skin at 100% before export.
6. Preserve the original and record the chosen preset or parameters.

## Current checkpoint

The project has CPU-friendly denoise, sharpening, exposure, white balance, and the `portrait-natural` preset. Face-aware masking and local retouching are not implemented yet.

## Extension guide

1. Add face detection only after selecting a CPU-compatible OpenCV approach.
2. Keep masks explicit and test that edits stay inside the intended region.
3. Add before/after fixtures that cover different skin tones and lighting.
4. Avoid fully automatic smoothing without a reviewable preview.
