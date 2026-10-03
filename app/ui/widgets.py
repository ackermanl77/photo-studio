"""Widgets generated from the app.controls registry (F15).

render_control("contrast_amount") renders the Contrast slider with the label, range, step, default,
help and session_state key declared once in app.controls.CONTROLS, and returns its value. Callers
only add what is about the layout (a column to render into, a collapsed label, disabled-ness).
"""
from __future__ import annotations

import streamlit as st

from app.controls import CONTROLS_BY_KEY

_UNSET = object()


def render_control(key: str, container=None, *, label: str | None = None, disabled: bool = False,
                   label_visibility: str = "visible", format_func=None, help=_UNSET, on_change=None):
    """Render recipe control `key` into `container` (default: the current Streamlit context) and
    return the widget's value. The widget keeps the control's own session_state key, which saved
    state, presets and projects depend on."""
    control = CONTROLS_BY_KEY[key]
    target = container if container is not None else st
    kwargs = {"key": control.widget_key, "disabled": disabled, "label_visibility": label_visibility}
    help_text = control.help if help is _UNSET else help
    if help_text:
        kwargs["help"] = help_text
    if on_change is not None:
        kwargs["on_change"] = on_change
    text = label if label is not None else control.label
    widget = control.widget_type
    if widget == "slider":
        args = (text, control.minimum, control.maximum, control.default)
        if control.step is not None:
            kwargs["step"] = control.step
        return target.slider(*args, **kwargs)
    if widget == "checkbox":
        return target.checkbox(text, value=control.default, **kwargs)
    if widget == "selectbox":
        choices = list(control.choices)
        if format_func is not None:
            kwargs["format_func"] = format_func
        return target.selectbox(text, choices, index=choices.index(control.default), **kwargs)
    if widget == "number_input":
        return target.number_input(text, control.minimum, control.maximum, control.default, **kwargs)
    raise ValueError(f"control {key!r} has no widget of its own ({widget})")


def render_controls(keys, container=None, **overrides) -> dict:
    """render_control() for several keys in order; returns {recipe key: value}."""
    return {key: render_control(key, container, **overrides) for key in keys}
