# Histogram and Clipping Inspection Workflow

> **Histórico:** este archivo describe un panel de sesión expandible con 32 bins y porcentajes de clipping que ya no existe. El histograma actual (`app.professional_tools.compute_histogram()`) se muestra siempre visible debajo del canvas en `frontend_unified_app.py`, con 64 bins por canal y el eje X en valores de píxel (0-255) en vez de índice de bin — ver la entrada "processing… indicator, one-click reset, histogram" en `docs/PROJECT_STATUS.md`'s "Recently Fixed" para el estado vigente. No incluye porcentajes de clipping.

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
