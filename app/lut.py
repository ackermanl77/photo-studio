"""3D (and 1D) LUTs in the Adobe/Resolve `.cube` format (F17): creative looks and camera/film
emulations, applied as the last Color step with a strength.

A LUT is chosen by name from the user's LUT library (`user_data_dir()/luts`, filled by "Import a
LUT" in the Color section or the CLI's `--lut FILE`), so recipes, presets and projects carry the
name, not a path. Interpolation is trilinear on float pixels (8- and 16-bit alike), run in chunks
so a 24 MP export doesn't need gigabytes of temporaries. Streamlit-free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import user_data_dir

MAX_LUT_BYTES = 64 * 1024 * 1024  # a 65^3 .cube is about 7 MB; keep the message in import_lut() in step
MAX_LUT_SIZE = 129
NAME_PATTERN = re.compile(r"^[\w][\w .()+&,'-]{0,79}$")
_CHUNK = 1 << 20  # pixels per interpolation chunk


@dataclass(frozen=True)
class Lut:
    """`table` is (N, N, N, 3) indexed [r][g][b] for a 3D LUT, or (N, 3) for a 1D LUT; values are the
    output colours. `domain_min`/`domain_max` map input values onto the grid."""

    table: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray
    title: str = ""

    @property
    def is_3d(self) -> bool:
        return self.table.ndim == 4

    @property
    def size(self) -> int:
        return self.table.shape[0]


def parse_cube(text: str) -> Lut:
    """Parse a `.cube` file. Raises ValueError for anything that isn't a valid 1D/3D LUT."""
    size_3d = size_1d = None
    domain_min, domain_max = np.zeros(3), np.ones(3)
    title = ""
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        keyword = line.split(None, 1)[0].upper()
        if keyword == "TITLE":
            title = line[5:].strip().strip('"')
        elif keyword == "LUT_3D_SIZE":
            size_3d = _int_value(line)
        elif keyword == "LUT_1D_SIZE":
            size_1d = _int_value(line)
        elif keyword == "DOMAIN_MIN":
            domain_min = _triple(line)
        elif keyword == "DOMAIN_MAX":
            domain_max = _triple(line)
        elif keyword in ("LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE"):
            low, high = (float(v) for v in line.split()[1:3])
            domain_min, domain_max = np.full(3, low), np.full(3, high)
        elif keyword[0].isdigit() or keyword[0] in "-.+":
            try:
                rows.append([float(v) for v in line.split()[:3]])
            except ValueError as error:
                raise ValueError(f"not a .cube LUT: bad data line {line[:40]!r}") from error
            if len(rows[-1]) != 3:
                raise ValueError("not a .cube LUT: a data line needs three numbers")
        # Other keywords (LUT_IN_VIDEO_RANGE, vendor extensions...) don't change the colours.
    if (size_3d is None) == (size_1d is None):
        raise ValueError("not a .cube LUT: it needs exactly one LUT_3D_SIZE or LUT_1D_SIZE line")
    size = size_3d or size_1d
    if not 2 <= size <= (MAX_LUT_SIZE if size_3d else 65536):
        raise ValueError(f"LUT size {size} is out of range")
    expected = size ** 3 if size_3d else size
    if len(rows) != expected:
        raise ValueError(f"the LUT declares {expected} entries but has {len(rows)}")
    if np.any(domain_max <= domain_min):
        raise ValueError("the LUT's DOMAIN_MAX must be above DOMAIN_MIN")
    data = np.asarray(rows, dtype=np.float32)
    if size_3d:
        # .cube order: red changes fastest, then green, then blue -> [b][g][r]; store [r][g][b].
        data = data.reshape(size, size, size, 3).transpose(2, 1, 0, 3)
    return Lut(np.ascontiguousarray(data), domain_min.astype(np.float32), domain_max.astype(np.float32), title)


def _int_value(line: str) -> int:
    try:
        return int(line.split()[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"not a .cube LUT: bad line {line[:40]!r}") from error


def _triple(line: str) -> np.ndarray:
    try:
        values = np.array([float(v) for v in line.split()[1:4]])
    except ValueError as error:
        raise ValueError(f"not a .cube LUT: bad line {line[:40]!r}") from error
    if values.shape != (3,):
        raise ValueError(f"not a .cube LUT: bad line {line[:40]!r}")
    return values


def apply_lut(rgb: np.ndarray, lut: Lut, amount: float = 1.0) -> np.ndarray:
    """`rgb` float (..., 3) in 0..1 through `lut`, blended with the original by `amount` (0..1).
    Returns float32, clipped to 0..1."""
    rgb = np.asarray(rgb, dtype=np.float32)
    shape = rgb.shape
    flat = rgb.reshape(-1, 3)
    out = np.empty_like(flat)
    for start in range(0, flat.shape[0], _CHUNK):
        chunk = flat[start:start + _CHUNK]
        out[start:start + _CHUNK] = _lut_3d(chunk, lut) if lut.is_3d else _lut_1d(chunk, lut)
    out = np.clip(out, 0.0, 1.0)
    if amount < 1.0:
        out = flat + (out - flat) * np.float32(max(0.0, amount))
    return out.reshape(shape)


def _grid(values: np.ndarray, lut: Lut) -> np.ndarray:
    scale = (lut.size - 1) / (lut.domain_max - lut.domain_min)
    return np.clip((values - lut.domain_min) * scale, 0.0, lut.size - 1)


def _lut_1d(chunk: np.ndarray, lut: Lut) -> np.ndarray:
    pos = _grid(chunk, lut)
    lo = np.floor(pos).astype(np.int32)
    hi = np.minimum(lo + 1, lut.size - 1)
    frac = pos - lo
    out = np.empty_like(chunk)
    for c in range(3):
        column = lut.table[:, c]
        out[:, c] = column[lo[:, c]] * (1 - frac[:, c]) + column[hi[:, c]] * frac[:, c]
    return out


def _lut_3d(chunk: np.ndarray, lut: Lut) -> np.ndarray:
    pos = _grid(chunk, lut)
    lo = np.minimum(np.floor(pos).astype(np.int32), lut.size - 2)
    frac = pos - lo
    r0, g0, b0 = lo[:, 0], lo[:, 1], lo[:, 2]
    fr, fg, fb = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]
    t = lut.table
    c000, c100 = t[r0, g0, b0], t[r0 + 1, g0, b0]
    c010, c110 = t[r0, g0 + 1, b0], t[r0 + 1, g0 + 1, b0]
    c001, c101 = t[r0, g0, b0 + 1], t[r0 + 1, g0, b0 + 1]
    c011, c111 = t[r0, g0 + 1, b0 + 1], t[r0 + 1, g0 + 1, b0 + 1]
    c00 = c000 + (c100 - c000) * fr
    c10 = c010 + (c110 - c010) * fr
    c01 = c001 + (c101 - c001) * fr
    c11 = c011 + (c111 - c011) * fr
    c0 = c00 + (c10 - c00) * fg
    c1 = c01 + (c11 - c01) * fg
    return c0 + (c1 - c0) * fb


# --- the LUT library -----------------------------------------------------------------------------

def lut_dir() -> Path:
    return user_data_dir() / "luts"


def is_valid_name(name: object) -> bool:
    return isinstance(name, str) and bool(NAME_PATTERN.match(name))


def list_luts() -> list[str]:
    """The names of the LUTs in the library, sorted case-insensitively."""
    folder = lut_dir()
    if not folder.is_dir():
        return []
    names = [path.stem for path in folder.glob("*.cube") if is_valid_name(path.stem)]
    return sorted(names, key=str.casefold)


def import_lut(content: bytes, filename: str) -> str:
    """Validate a `.cube` file and copy it into the library under its own name (replacing a LUT
    with the same name). Returns the name. Raises ValueError for a bad file."""
    if len(content) > MAX_LUT_BYTES:
        raise ValueError("That LUT is larger than 64 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("not a .cube LUT: the file isn't text") from error
    parse_cube(text)
    stem = re.sub(r"[^\w .()+&,'-]", "_", Path(filename).stem).strip(" .")[:80] or "LUT"
    if not is_valid_name(stem):
        stem = "LUT " + stem.lstrip("._- ")[:70]
    folder = lut_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.cube").write_text(text, encoding="utf-8")
    _load_cached.cache_clear()
    return stem


def load_lut(name: str) -> Lut:
    """The library LUT called `name`. Raises ValueError when it isn't there (a recipe or preset from
    another computer) or isn't a valid name."""
    if not is_valid_name(name):
        raise ValueError(f"Invalid LUT name: {name!r}")
    path = lut_dir() / f"{name}.cube"
    if not path.is_file():
        raise ValueError(f"LUT not found: {name} -- import it in Color > LUT first")
    stat = path.stat()
    return _load_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_cached(path: str, _mtime: int, _size: int) -> Lut:
    return parse_cube(Path(path).read_text(encoding="utf-8-sig"))
