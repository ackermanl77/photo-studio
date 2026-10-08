"""Interface language (English / Español).

The UI's text is written in English in the source; `t(text)` looks it up in the active language's
catalog (app/i18n_es.py for Spanish) and returns the text unchanged when there is no entry, so
English needs no catalog and an untranslated string still shows (in English). Only displayed
text goes through here: widget keys, option values, recipe keys and tab values passed to the
pipeline stay English, so saved state (Undo/Redo, presets, projects) is language-independent.
Choice widgets keep their English values and show them translated through `format_func=t_option`.

Strings with values in them are translated as templates: f"Saved {name}" in the UI source is
written t("Saved {0}").format(t(name)), and the catalog entry translates "Saved {0}".

The language is per Streamlit session: the entry script calls set_language() at the start of
every run (a thread-local, like app.ui.state's tab visibility); anything else falls back to the
saved preference. The preference lives in user_data_dir()/settings.json (`"language"`), and
PHOTO_STUDIO_LANGUAGE ("en"/"es") overrides it. Streamlit-free.
"""
from __future__ import annotations

import json
import os
import threading

LANGUAGES = {"en": "English", "es": "Español"}
DEFAULT_LANGUAGE = "en"

_local = threading.local()


def _catalog(language: str) -> dict:
    if language == "es":
        from app.i18n_es import ES

        return ES
    return {}


def _settings_path():
    from app.config import user_data_dir

    return user_data_dir() / "settings.json"


def _read_settings() -> dict:
    try:
        data = json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def saved_language() -> str:
    """The preferred language: PHOTO_STUDIO_LANGUAGE, else settings.json, else English."""
    language = os.environ.get("PHOTO_STUDIO_LANGUAGE", "").strip().lower() or _read_settings().get("language")
    return language if language in LANGUAGES else DEFAULT_LANGUAGE


def save_language(language: str) -> None:
    """Remember `language` for the next launch (best effort: a read-only data folder is no error)."""
    if language not in LANGUAGES:
        return
    from app.config import atomic_write_text

    settings = _read_settings()
    settings["language"] = language
    try:
        atomic_write_text(_settings_path(), json.dumps(settings, indent=2))
    except OSError:
        pass


def set_language(language: str | None) -> None:
    """Make `language` this thread's (= this Streamlit run's) interface language."""
    _local.language = language if language in LANGUAGES else DEFAULT_LANGUAGE


def current_language() -> str:
    language = getattr(_local, "language", None)
    return language if language is not None else saved_language()


def t(text):
    """`text` in the current language; non-strings and untranslated strings come back unchanged."""
    if not isinstance(text, str) or not text:
        return text
    return _catalog(current_language()).get(text, text)


def t_option(value) -> str:
    """format_func for a choice widget whose values are English labels."""
    return t(value) if isinstance(value, str) else str(value)


def t_format(format_func):
    """Wrap an existing format_func so its result is translated too."""
    return lambda value: t(format_func(value))
