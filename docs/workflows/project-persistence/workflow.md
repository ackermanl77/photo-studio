# Project Persistence Integration Workflow

## Goal

Save the source image, edit recipe, layers, masks, and project metadata together in one validated `.photoproject` archive.

## Steps

1. Create a layer workspace and edit stack.
2. Collect any local or object-removal masks.
3. Create a `ProjectBundle` with source bytes and recipe.
4. Save it as `.photoproject`.
5. Load it later through `ProjectBundle.load()`.
6. Restore the layer workspace with `layer_workspace_object()`.
7. Review the restored project before exporting.

## Safety

Loading uses the project archive validator, checks the manifest, rejects unsafe paths, and enforces archive limits before reading project content.

## Current integration boundary

The bundle API is ready for the complete frontend to use. The existing simple `PhotoProject` format remains version 1 compatible; the new bundle format is version 2 and adds layers and masks.
