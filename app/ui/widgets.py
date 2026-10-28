"""Widgets generated from the app.controls registry (F15).

render_control("contrast_amount") renders the Contrast slider with the label, range, step, default,
help and session_state key declared once in app.controls.CONTROLS, and returns its value. Callers
only add what is about the layout (a column to render into, a collapsed label, disabled-ness).
"""
from __future__ import annotations

import streamlit as st

from app.controls import CONTROLS_BY_KEY
from app.i18n import t, t_format, t_option

_UNSET = object()
# Container key prefix of a slider and its number box (static/theme.css styles these rows).
VALUE_ROW_PREFIX = "value-row-"


def _number_format(step: float) -> str:
    text = f"{step:g}"
    return f"%.{len(text.split('.')[1])}f" if "." in text else "%.0f"


def _value_typed(key: str, on_change=None) -> None:
    """on_change of a slider's number box: move the slider to the typed value, snapped to its step.
    A callback runs before the script, so the slider isn't instantiated yet and its key may be set."""
    control = CONTROLS_BY_KEY[key]
    typed = st.session_state.get(f"{control.widget_key}-value")
    if typed is None:
        return
    step = control.step or (0.01 if control.kind == "float" else 1)
    snapped = control.minimum + round((float(typed) - control.minimum) / step) * step
    snapped = min(max(snapped, control.minimum), control.maximum)
    st.session_state[control.widget_key] = round(snapped, 6) if control.kind == "float" else int(round(snapped))
    if on_change is not None:
        on_change()


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
        kwargs["help"] = t(help_text)
    if on_change is not None:
        kwargs["on_change"] = on_change
    text = t(label if label is not None else control.label)
    widget = control.widget_type
    if widget == "slider":
        args = (text, control.minimum, control.maximum, control.default)
        if control.step is not None:
            kwargs["step"] = control.step
        if label_visibility != "visible":
            return target.slider(*args, **kwargs)
        # UI4: a number box in the slider's label row (static/theme.css places it), for typing an
        # exact value as in Photoshop's Adjustments. Set from the slider before it renders, so it
        # always shows the slider's value; typing calls _value_typed().
        row = target.container(key=f"{VALUE_ROW_PREFIX}{control.widget_key}")
        value = row.slider(*args, **kwargs)
        value_key = f"{control.widget_key}-value"
        st.session_state[value_key] = value
        number = float if control.kind == "float" else int
        step = number(control.step or (0.01 if number is float else 1))
        row.number_input(text, min_value=number(control.minimum), max_value=number(control.maximum), step=step,
                         format=_number_format(step) if number is float else "%d",
                         key=value_key, disabled=disabled, label_visibility="collapsed", on_change=_value_typed, args=(key, on_change))
        return value
    if widget == "checkbox":
        return target.checkbox(text, value=control.default, **kwargs)
    if widget == "selectbox" and control.kind == "lut":
        # The LUT library (app.lut, F17): "" = none, then every imported LUT; a name a restored
        # recipe or preset refers to but this computer doesn't have stays selectable, marked missing.
        from app.lut import list_luts

        names = list_luts()
        current = st.session_state.get(control.widget_key, "")
        choices = ["", *names, *([current] if current and current not in names else [])]
        kwargs["format_func"] = lambda name: t("None") if not name else (name if name in names else t("{0} (missing)").format(name))
        return target.selectbox(text, choices, index=0, **kwargs)
    if widget == "selectbox":
        choices = list(control.choices)
        kwargs["format_func"] = t_format(format_func) if format_func is not None else t_option
        return target.selectbox(text, choices, index=choices.index(control.default), **kwargs)
    if widget == "color_picker":
        return target.color_picker(text, control.default, **kwargs)
    if widget == "number_input":
        return target.number_input(text, control.minimum, control.maximum, control.default, **kwargs)
    raise ValueError(f"control {key!r} has no widget of its own ({widget})")


def render_controls(keys, container=None, **overrides) -> dict:
    """render_control() for several keys in order; returns {recipe key: value}."""
    return {key: render_control(key, container, **overrides) for key in keys}
