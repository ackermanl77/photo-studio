# Histogram and Clipping Inspection Workflow

## Goal

Evaluate tonal changes while editing instead of relying only on visual preview.

## Steps

1. Upload a photo and open the Session panel.
2. Expand `Histogram and clipping`.
3. Compare the source and edited RGB distributions.
4. Watch the shadow clipping percentage after lifting or deepening dark areas.
5. Watch the highlight clipping percentage after increasing exposure or contrast.
6. Reduce exposure, highlights, or contrast if important detail is being clipped.
7. Recheck the edited preview at 100% before export.

## Interpretation

The histogram is displayed as 32 tone bins for red, green, and blue. Clipping percentages count channel values at the near-black and near-white thresholds. They are warnings for review, not automatic failures; specular highlights and intentional silhouettes may be valid.
