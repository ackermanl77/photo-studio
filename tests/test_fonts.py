"""Fonts for Text layers and the watermark (app/fonts.py). No font file ships with the tests (fonts
carry their own licences): the tests use whatever standard font this machine has and skip without one."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import fonts
from app.export import validate_export_settings
from app.layer_workspace import render_text_layer
from app.professional_tools import apply_watermark

SYSTEM_FONT = next((path for path in fonts._DEFAULT_CANDIDATES if Path(path).exists()), None)
needs_font = pytest.mark.skipif(SYSTEM_FONT is None, reason="no standard TrueType font on this machine")


@needs_font
def test_an_uploaded_font_is_installed_once_and_listed_by_name():
    data = Path(SYSTEM_FONT).read_bytes()
    name = fonts.install_font(data, "MyFont.ttf")
    assert name == fonts.font_display_name(SYSTEM_FONT)
    assert fonts.install_font(data, "again.ttf") == name
    assert len(list(fonts.uploaded_fonts_dir().iterdir())) == 1
    assert Path(fonts.available_fonts()[name]).parent == fonts.uploaded_fonts_dir()


def test_a_file_that_is_not_a_font_is_refused():
    with pytest.raises(ValueError):
        fonts.install_font(b"definitely not a font", "fake.ttf")
    assert not fonts.uploaded_fonts_dir().exists() or not any(fonts.uploaded_fonts_dir().iterdir())


def test_an_unknown_font_name_falls_back_to_the_default():
    default = fonts.load_font("", 20)
    missing = fonts.load_font("No Such Font Anywhere", 20)
    assert type(default) is type(missing)
    image = render_text_layer((120, 40), "Hi", 20, font="No Such Font Anywhere")
    assert np.asarray(image)[..., 3].any()


def test_text_layers_render_in_the_chosen_font():
    names = list(fonts.available_fonts())
    if len(names) < 2:
        pytest.skip("needs two installed fonts")
    first, second = names[0], names[-1]
    a = np.asarray(render_text_layer((200, 60), "Typeface", 30, font=first))
    b = np.asarray(render_text_layer((200, 60), "Typeface", 30, font=second))
    assert not np.array_equal(a, b)


def test_the_watermark_font_is_kept_in_export_settings_and_used():
    settings = validate_export_settings({"watermark": {"text": "©", "font": "Some Font"}})
    assert settings["watermark"]["font"] == "Some Font"
    assert validate_export_settings({"watermark": {"text": "©"}})["watermark"]["font"] == ""
    base = Image.new("RGB", (160, 80), (30, 30, 30))
    marked = apply_watermark(base, "Proof", "Center", 100, 24, font="No Such Font Anywhere")
    assert not np.array_equal(np.asarray(marked), np.asarray(base))
