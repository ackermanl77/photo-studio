"""Layers tab."""
from __future__ import annotations

import streamlit as st

from app.ui.layers import render_layer_stack
from app.ui.state import section_header
from app.i18n import t


def layers_section(ctx) -> None:
    section_header("Layers", extra_modified=bool(st.session_state.get("image_layers")))
    st.caption(t("Add any number of image, text, shape or pen-path layers, each with its own opacity, blend mode and styles -- composited on the processing backend when the image renders."))
    ctx.layers_signature = render_layer_stack(ctx.source)
