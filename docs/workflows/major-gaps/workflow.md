# Major Feature Foundations

This feature slice adds reusable foundations for the remaining professional-editor gaps.

## Layers and adjustment layers

`app/major_features.py` contains `LayerStack` and `Layer` with float compositing, opacity, visibility, masks, and normal/multiply/screen blend modes.

Next integration step: expose a Layers panel in the Streamlit frontend and send layer descriptors through the Flask API.

## Healing and clone repair

Not implemented. An unused `heal_patch()` prototype (OpenCV seamless cloning) was removed as dead code (B12); a spot heal/clone tool would be rebuilt with a painted source/target UI (F13). Repair (inpainting) covers object removal today.

## Preview performance

`preview_image()` creates a bounded preview before sending a large image through interactive controls. Full-resolution processing should remain reserved for final export.

Next integration step: cache previews by source hash plus recipe hash.

## Photo catalog

`PhotoCatalog` stores local paths, ratings, and tags in SQLite and supports text/rating search without a remote service.

Next integration step: add a browser grid with thumbnails, rating controls, and tag filters.

## 16-bit limitation

The current `PhotoProcessor` still converts normal images to RGB working buffers. RAW decoding and these foundations are ready, but print-critical 16-bit/float processing remains a separate migration because every adjustment method must stop assuming 8-bit channel ranges.
