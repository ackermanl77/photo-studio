# Advanced Masking Workflow

## Goal

Create usable masks directly in the frontend for local adjustments or object removal.

## Steps

1. Open the `Masks` section.
2. Choose `Object removal` or `Local adjustment`.
3. Choose `Paint` to add white mask areas or `Erase` to restore black protected areas.
4. Set brush size for broad or precise work.
5. Reduce brush opacity for gradual masks.
6. Increase feather to soften the mask edge.
7. Inspect the edited preview before exporting.
8. Save a `.photoproject` to preserve the source, recipe, and masks.

## Mask behavior

- White selects the full operation.
- Black protects the image.
- Gray applies a partial operation.
- The brush canvas is scaled for display and the backend resizes the resulting mask to the source dimensions.
