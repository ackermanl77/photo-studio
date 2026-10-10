"""The interface language (app/i18n.py) and its Spanish catalog (app/i18n_es.py)."""
from __future__ import annotations

import string

import pytest

from app import i18n
from app.i18n_es import ES
from app.controls import CONTROLS


@pytest.fixture(autouse=True)
def _english_afterwards():
    yield
    i18n.set_language("en")


def test_english_returns_the_source_text_and_spanish_translates():
    i18n.set_language("en")
    assert i18n.t("Contrast") == "Contrast"
    i18n.set_language("es")
    assert i18n.t("Contrast") == "Contraste"
    assert i18n.t("text with no catalog entry") == "text with no catalog entry"
    assert i18n.t(3.5) == 3.5  # values interpolated into templates pass through untouched
    assert i18n.t_option(90) == "90"


def test_unknown_language_falls_back_to_english():
    i18n.set_language("xx")
    assert i18n.current_language() == "en"


def test_the_language_preference_is_saved_and_env_overrides_it(monkeypatch):
    monkeypatch.delenv("PHOTO_STUDIO_LANGUAGE", raising=False)
    assert i18n.saved_language() == "en"
    i18n.save_language("es")
    assert i18n.saved_language() == "es"
    monkeypatch.setenv("PHOTO_STUDIO_LANGUAGE", "en")
    assert i18n.saved_language() == "en"


def _fields(text: str) -> list:
    return sorted(field for _, field, spec, _ in string.Formatter().parse(text) if field is not None)


@pytest.mark.parametrize("source", sorted(ES))
def test_every_translation_keeps_its_placeholders(source):
    # "{0}" templates are filled with .format(); a lost or extra field would raise or drop a value.
    # "{stem}"-style fields in help text are literal (never formatted) and must survive verbatim.
    assert _fields(ES[source]) == _fields(source)


def test_every_control_label_and_help_has_a_spanish_entry():
    missing = [text for control in CONTROLS if control.widget != "state"
               for text in (control.label, control.help) if text and text not in ES]
    assert missing == []


@pytest.mark.slow
def test_the_app_renders_every_tab_in_spanish(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from tests import test_frontend_unified_app as frontend
    from tests.test_frontend_unified_app import APP_PATH, TOOL_TABS, image_bytes

    monkeypatch.setattr(frontend.backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(frontend.backend_client, "process_via_backend", frontend._fake_process_via_backend)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.selectbox(key="ui-language").set_value("es")
    at.run(timeout=30)
    # AppTest calls the widgets' format_func on this (the test's) thread to match values to the
    # rendered options; the app's own runs set the language on the script thread.
    i18n.set_language("es")
    assert "Abrir imagen" in [uploader.label for uploader in at.file_uploader]
    at.file_uploader[0].set_value(("photo.jpg", image_bytes(), "image/jpeg"))
    at.run(timeout=30)
    assert not at.exception
    labels = [tab.label for tab in at.tabs]
    assert [label for label in labels if label in {ES[tab] for tab in TOOL_TABS}] == [ES[tab] for tab in TOOL_TABS]
    for tab in TOOL_TABS:
        at.session_state["tool-tab"] = ES[tab]
        at.run(timeout=30)
        assert not at.exception, tab
    for parent, key, subs in (("◉ Local", "local-tab", ("◉ Masks", "🩹 Repair", "◌ Portrait")),
                              ("✦ Presets & Library", "library-tab", ("✦ Presets", "☰ Library & Batch"))):
        for sub in subs:
            at.session_state["tool-tab"] = ES[parent]
            at.session_state[key] = ES[sub]
            at.run(timeout=30)
            assert not at.exception, sub
    assert at.slider(key="t-contrast").label == "Contraste"
    # Values stay English: saved state is language-independent.
    assert at.selectbox(key="crop-aspect").value == "Original"
    # And back to English mid-session.
    at.selectbox(key="ui-language").set_value("en")
    at.run(timeout=30)
    i18n.set_language("en")
    assert not at.exception
    assert at.slider(key="t-contrast").label == "Contrast"



def test_the_on_canvas_editors_follow_the_interface_language():
    from PIL import Image

    from app.ui_editors import _crop_editor_html, _paint_editor_html, _pen_editor_html

    background = Image.new("RGB", (40, 30))
    i18n.set_language("es")
    paint = _paint_editor_html(background, "m")[0]
    assert "Pincel" in paint and "Añadir a las selecciones" in paint
    assert '"Recorte:"' in _crop_editor_html(background, "crop-editor", (0.1, 0.1, 0.9, 0.9), (40, 30))[0]
    assert "Haz clic en la foto para añadir un punto" in _pen_editor_html(background, "pen", [], False)[0]
    i18n.set_language("en")
    assert "🖌 Brush" in _paint_editor_html(background, "m")[0]
