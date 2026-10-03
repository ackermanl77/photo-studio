"""Painted masks stored once per content (P5 in the improvement report).

A "Painted" selection used to carry its whole base64 PNG in `params["mask_b64"]`, and every Undo
step, named snapshot and selection signature copied it -- a few painted selections multiplied by
up to 200 history steps. Inside a session a painted selection now holds only a content reference,
`params["mask_ref"]` ("sha256:<hex>"), and the PNG lives once in a per-session table keyed by that
hash (st.session_state["_painted_masks"]); identical paintings share one entry.

Files stay self-contained and in the same format: presets and projects are saved with the masks
inlined again (inline_painted_masks), and anything restored from a file is interned on the way in
(intern_recipe_painted_masks). A selection may therefore carry either key; resolve_painted_mask()
accepts both.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io

SESSION_KEY = "_painted_masks"
SELECTION_LIST_KEYS = ("mask_layers", "removal_mask_layers")


def painted_mask_ref(mask_b64: str) -> str:
    return "sha256:" + hashlib.sha256(mask_b64.encode("ascii")).hexdigest()


def _session_table() -> dict:
    import streamlit as st

    return st.session_state.setdefault(SESSION_KEY, {})


def session_painted_masks() -> dict:
    """A snapshot of this session's table, for work bound up front to run on another thread."""
    return dict(_session_table())


def intern_painted_mask(mask_b64: str, table: dict | None = None) -> str:
    """Store `mask_b64` once (in `table`, default this session's) and return its reference."""
    ref = painted_mask_ref(mask_b64)
    (table if table is not None else _session_table()).setdefault(ref, mask_b64)
    return ref


def _empty_mask_b64() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def resolve_painted_mask(params: dict, table: dict | None = None) -> str:
    """The base64 PNG of a painted selection, inline or by reference. A reference this session
    doesn't hold (it can't come from a file -- those are always inlined) resolves to an empty mask
    rather than failing the whole render."""
    if params.get("mask_b64"):
        return params["mask_b64"]
    source = table if table is not None else _session_table()
    return source.get(params.get("mask_ref", ""), "") or _empty_mask_b64()


def _map_painted(recipe: dict, convert) -> dict:
    result = copy.deepcopy(recipe)
    for key in SELECTION_LIST_KEYS:
        for selection in result.get(key) or []:
            if isinstance(selection, dict) and selection.get("kind") == "Painted" and isinstance(selection.get("params"), dict):
                selection["params"] = convert(selection["params"])
    return result


def inline_painted_masks(recipe: dict, table: dict | None = None) -> dict:
    """A copy of `recipe` whose painted selections carry their PNG inline (`mask_b64`) -- the form
    written to presets and project files."""
    return _map_painted(recipe, lambda params: {key: value for key, value in params.items() if key != "mask_ref"} | {"mask_b64": resolve_painted_mask(params, table)})


def intern_recipe_painted_masks(recipe: dict, table: dict | None = None) -> dict:
    """A copy of `recipe` whose inline painted masks moved into the table, referenced by hash."""
    def convert(params):
        if not params.get("mask_b64"):
            return params
        rest = {key: value for key, value in params.items() if key != "mask_b64"}
        return rest | {"mask_ref": intern_painted_mask(params["mask_b64"], table)}

    return _map_painted(recipe, convert)
