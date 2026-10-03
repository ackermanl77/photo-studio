"""EditorContext: what one script run's panels hand to each other.

The tabs used to share ~60 script-level variables; now each panel reads the inputs it needs from
the context and writes its results back into it, and the main script turns `values` (recipe key ->
widget value, filled by every registry widget) plus the composite pieces into the pipeline options.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


@dataclass
class EditorContext:
    source: Image.Image
    source_digest: str
    upload_name: str
    upload_bytes: bytes
    source_exif: bytes | None = None
    # Recipe key -> this run's widget value, for every registry control rendered (app.controls).
    values: dict = field(default_factory=dict)
    # Geometry (rotate/flip/straighten + Lens), and the frame Masks/Repair/Crop work against.
    geometry_options: dict = field(default_factory=dict)
    mask_source: Image.Image | None = None
    mask_source_key: str = ""
    auto_level_requested: bool = False
    # Composite options the tabs assemble.
    hsl_bands: dict = field(default_factory=dict)
    aspect_code: str = "Original"
    crop_box: tuple | None = None
    # Masks / Repair / Layers.
    mask_signature: str = "[]"
    shared_mask_layers: list = field(default_factory=list)
    own_mask_layers: list = field(default_factory=list)
    own_signatures: list = field(default_factory=list)
    removal_signature: str = "[]"
    removal_active: bool = False
    layers_signature: str | None = None
    # Presets (the Batch Processor offers the saved ones) and Undo/Redo history.
    preset_names: list = field(default_factory=list)
    preset_store: object = None
    history: object = None
    # Filled by app.ui.processing.render_edit().
    options: dict = field(default_factory=dict)
    edited_image: Image.Image | None = None
    output_size: tuple = (0, 0)
    use_proxy: bool = False
    refined_image: Image.Image | None = None
    refine_tracker: object = None
    full_canvas: tuple = (0, 0)
    layers_key: str | None = None
    # Export tab settings the Save footer under the canvas uses.
    export: dict = field(default_factory=dict)
