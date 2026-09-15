# Edit History and Recipes Workflow

> **Histórico:** no existen botones "Save edit recipe" / "Load edit recipe" como descarga/carga independiente de JSON. `Undo`/`Redo` sí existen (siempre visibles en la barra sobre el canvas). Persistir el recipe hoy se hace con "💾 Save Project", que empaqueta el recipe junto con la imagen fuente y la capa/layer en un archivo `.photoproject` (`app.project_bundle.ProjectBundle`) — no un JSON de recipe suelto. Ver `docs/PROJECT_STATUS.md` para el flujo vigente.

## Goal

Keep edits reversible and reusable without changing the original upload.

## Steps

1. Upload a photo and adjust the controls.
2. Use `Undo` and `Redo` to move through adjustment snapshots.
3. Click `Save edit recipe` to download a JSON recipe.
4. Upload that recipe beside a new source image with `Load edit recipe`.
5. Review the preview before exporting because the same values can look different on another image.

## What is saved

The recipe stores the declarative processing options: tonal controls, color controls, detail settings, curves, preset, crop ratio, masks settings, and delivery preferences. Source pixels and brush strokes are not embedded in the recipe.

## Design rule

Every preview is rebuilt from the original image and the current recipe. This makes undo, redo, and recipe loading non-destructive.