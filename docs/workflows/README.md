# Feature Workflows

> **Nota (vigencia de esta carpeta):** estas guías se fueron escribiendo de forma incremental a medida que se construía el proyecto. Varias documentan Streamlit entry points que existieron solo temporalmente durante el desarrollo (`app/frontend.py`, `frontend_complete.py`, `frontend_center.py`, `frontend_integrated.py`, `frontend_unified_v2.py`, `integrated_workbench_next.py`, `frontend_release.py`, entre otros) y que **ya no existen** — todo se consolidó en `app/frontend_unified_app.py`. Esto aplica en particular a las guías "Flask backend", "End-user frontend", "Complete frontend", los cuatro archivos bajo "Integration next" (no listados abajo, ver `integration-next/`), y los registros de release bajo `desktop-exe/` distintos de `workflow.md` (`current-all-features.md`, `latest-release.md`, `all-features-release.md`, `global-release.md`). Se conservan como registro histórico del desarrollo, no como instrucciones vigentes.
>
> Para la arquitectura, los entry points y los problemas conocidos **actuales**, la fuente de verdad es [`docs/PROJECT_STATUS.md`](../PROJECT_STATUS.md).

These guides document the current application behavior, launch commands, validation steps, and known limits.

## Core Editing

- [Exposure](exposure/workflow.md)
- [Contrast](contrast/workflow.md)
- [Highlights and shadows](highlights-shadows/workflow.md)
- [Texture and clarity](texture-clarity/workflow.md)
- [Temperature and tint](temperature-tint/workflow.md)
- [Curves](curves/workflow.md)
- [Crop and framing](crop-and-framing/workflow.md)
- [Color grading](color-grading/workflow.md)
- [Edit stack](edit-stack/workflow.md)
- [Edit stack frontend](edit-stack/frontend-workflow.md)
- [Presets](presets/workflow.md)
- [Preset exchange](preset-exchange/workflow.md)

## Layers and Masks

- [Layer workspace](layer-workspace/frontend-workflow.md)
- [Masking](masking/workflow.md)
- [Advanced masking](advanced-masking/workflow.md)
- [Advanced masking frontend](advanced-masking/frontend-workflow.md)
- [Object removal](object-removal/workflow.md)
- [Portrait retouching](portrait-retouching/workflow.md)
- [Portrait retouching frontend](portrait-retouching/frontend-workflow.md)

## Library and Delivery

- [Photo library](photo-library/workflow.md)
- [Library-to-editor workflow](photo-library/editor-workflow.md)
- [Batch processing](batch-processing/workflow.md)
- [Batch release](batch-release/workflow.md)
- [Batch frontend](batch-release/frontend-workflow.md)
- [Export validation](export-validation/workflow.md)
- [Complete frontend](complete-frontend/workflow.md)

## Projects and Runtime

- [Project files](project-files/workflow.md)
- [Project persistence](project-persistence/workflow.md)
- [Project session frontend](project-persistence/frontend-workflow.md)
- [Unified application](unified-application/workflow.md)
- [End-user frontend](frontend/workflow.md)
- [Flask backend](backend/workflow.md)
- [Global Python](global-python/workflow.md)
- [Diagnostics](diagnostics/workflow.md)

## Performance, Security, and Release

- [Preview performance](performance/workflow.md)
- [Preview cache integration](performance/cache-integration.md)
- [Reliability and security](reliability-security/workflow.md)
- [Backend hardening](reliability-security/backend-integration.md)
- [Observability](observability/workflow.md)
- [Safety and export](safety-and-export/workflow.md)
- [Secure Windows distribution](secure-distribution/workflow.md)
- [Windows desktop EXE](desktop-exe/workflow.md)
- [Latest desktop release](desktop-exe/latest-release.md)
- [Global release](desktop-exe/global-release.md)
- [Histogram and clipping inspection](inspection-tools/workflow.md)

## RAW

- [RAW import](raw-import/workflow.md)

## Project Status

See [docs/PROJECT_STATUS.md](../PROJECT_STATUS.md) for architecture, primary entry points, release location, verification, and known limitations.
