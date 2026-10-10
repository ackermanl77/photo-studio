"""Fonts for Text layers and the watermark.

`available_fonts()` lists the TrueType/OpenType fonts installed on this machine (the usual system
and per-user font folders of Windows, Linux and macOS) plus the ones the user uploaded through the
Layers tab (`install_font()`, stored under `user_data_dir()/fonts`), by display name -- "Arial",
"Arial Bold", "Georgia Italic". A Text layer or a watermark stores that name (`""` = the automatic
default), so presets and projects stay readable; `load_font(name, size)` resolves it, and a name this
machine doesn't have falls back to the automatic default -- a short list of common fonts, then
Pillow's own built-in font -- instead of failing.

No font file is bundled with the app (fonts carry their own licences; an uploaded font is the user's
own choice). Streamlit-free: the backend uses it for the watermark too.
"""
from __future__ import annotations

import hashlib
import io
import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from app.config import user_data_dir

FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")
MAX_FONT_BYTES = 30 * 1024 * 1024  # keep the message in install_font() in step
# Styles shown without a suffix ("Arial", not "Arial Regular").
_PLAIN_STYLES = {"", "regular", "normal", "book", "roman", "plain"}

# The automatic default (a font name of ""), checked in order: common fonts likely to be installed,
# then Pillow's built-in font.
_DEFAULT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def uploaded_fonts_dir() -> Path:
    return user_data_dir() / "fonts"


def _system_font_dirs() -> list[Path]:
    home = Path.home()
    dirs = []
    if os.environ.get("WINDIR"):
        dirs.append(Path(os.environ["WINDIR"]) / "Fonts")
    if os.environ.get("LOCALAPPDATA"):
        dirs.append(Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Windows" / "Fonts")  # per-user installs
    dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".local" / "share" / "fonts",
             home / ".fonts", Path("/System/Library/Fonts"), Path("/Library/Fonts"), home / "Library" / "Fonts"]
    return dirs


def font_display_name(path: str | Path) -> str | None:
    """"Family Style" read from the font file itself, or None when it isn't a readable font."""
    try:
        family, style = ImageFont.truetype(str(path), 12).getname()
    except (OSError, ValueError):
        return None
    family, style = (family or "").strip(), (style or "").strip()
    if not family:
        return None
    return family if style.lower() in _PLAIN_STYLES else f"{family} {style}"


def _scan(folders: list[Path]) -> dict[str, str]:
    fonts: dict[str, str] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in FONT_EXTENSIONS and path.is_file():
                name = font_display_name(path)
                if name and name not in fonts:
                    fonts[name] = str(path)
    return fonts


@lru_cache(maxsize=1)
def _system_fonts() -> dict[str, str]:
    # Scanned once per process (opening every font file takes a moment); a font installed while the
    # app runs shows up after a restart -- or upload it instead.
    return _scan(_system_font_dirs())


def available_fonts() -> dict[str, str]:
    """Display name -> font file, sorted by name. An uploaded font wins over a system one of the
    same name."""
    fonts = dict(_system_fonts())
    fonts.update(_scan([uploaded_fonts_dir()]))
    return dict(sorted(fonts.items(), key=lambda item: item[0].lower()))


def font_path(name: str) -> str | None:
    return available_fonts().get(name) if name else None


def install_font(data: bytes, filename: str = "") -> str:
    """Store an uploaded .ttf/.otf/.ttc under user_data_dir()/fonts (named by content, so the same
    file uploaded twice is stored once) and return its display name. ValueError when it isn't a
    font or is too large."""
    if len(data) > MAX_FONT_BYTES:
        raise ValueError("The font file is larger than 30 MB.")
    suffix = Path(filename).suffix.lower()
    if suffix not in FONT_EXTENSIONS:
        suffix = ".ttf"
    try:
        ImageFont.truetype(io.BytesIO(data), 12)
    except (OSError, ValueError) as error:
        raise ValueError("That file isn't a TrueType or OpenType font.") from error
    folder = uploaded_fonts_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{hashlib.sha256(data).hexdigest()[:16]}{suffix}"
    if not target.exists():
        target.write_bytes(data)
    name = font_display_name(target)
    if name is None:
        target.unlink(missing_ok=True)
        raise ValueError("That file isn't a TrueType or OpenType font.")
    return name


def load_font(name: str, size: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """The font called `name` at `size` px; "" or a name this machine doesn't have gives the
    automatic default."""
    size = max(1, int(size))
    path = font_path(name)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    for candidate in _DEFAULT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1 -- a scalable fallback
    except TypeError:
        return ImageFont.load_default()  # older Pillow: a small, fixed-size bitmap font
