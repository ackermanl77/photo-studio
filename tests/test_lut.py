"""3D/1D LUTs (.cube) in the Color chain (F17): parsing, interpolation, the library, the pipeline."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from app import cli
from app.edit_schema import OptionsError, parse_options, sanitize_recipe
from app.lut import apply_lut, import_lut, is_valid_name, list_luts, load_lut, lut_dir, parse_cube


def cube_3d(fn, size: int = 17, header: str = "") -> str:
    """A .cube file of `fn` (float rgb -> rgb) sampled on a size^3 grid, red fastest."""
    grid = np.linspace(0.0, 1.0, size)
    lines = [header, f"LUT_3D_SIZE {size}"] if header else [f"LUT_3D_SIZE {size}"]
    for b in grid:
        for g in grid:
            for r in grid:
                out = fn(np.array([r, g, b]))
                lines.append(" ".join(f"{v:.6f}" for v in out))
    return "\n".join(lines) + "\n"


INVERT = cube_3d(lambda c: 1.0 - c, size=2)
WARM = cube_3d(lambda c: np.clip(c * [1.1, 1.0, 0.85], 0, 1))


def test_parse_and_apply_a_3d_lut_exactly_on_linear_transforms():
    lut = parse_cube(INVERT)
    rgb = np.random.default_rng(1).random((50, 40, 3), dtype=np.float32)
    assert np.allclose(apply_lut(rgb, lut), 1.0 - rgb, atol=1e-6)  # trilinear is exact on an affine map
    assert np.allclose(apply_lut(rgb, lut, amount=0.5), 0.5 + 0 * rgb, atol=1e-6)  # halfway = mid-grey
    assert np.allclose(apply_lut(rgb, lut, amount=0.0), rgb)


def test_the_axis_order_red_fastest_is_respected():
    swap = parse_cube(cube_3d(lambda c: c[::-1], size=2))  # output (b, g, r)
    assert np.allclose(apply_lut(np.array([[0.9, 0.5, 0.1]], np.float32), swap), [[0.1, 0.5, 0.9]], atol=1e-6)


def test_1d_luts_domains_and_comments():
    text = "# a comment\nTITLE \"Gamma\"\nLUT_1D_SIZE 3\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n0 0 0\n0.25 0.25 0.25\n1 1 1\n"
    lut = parse_cube(text)
    assert not lut.is_3d and lut.title == "Gamma"
    # Domain 0..2 -> an input of 0.5 sits a quarter of the way along: half-way to the 0.25 entry.
    assert np.allclose(apply_lut(np.array([[0.5, 1.0, 0.0]], np.float32), lut), [[0.125, 0.25, 0.0]], atol=1e-6)


@pytest.mark.parametrize("text, message", [
    ("0 0 0\n", "LUT_3D_SIZE"),
    ("LUT_3D_SIZE 2\n0 0 0\n", "declares 8"),
    ("LUT_3D_SIZE 2\nLUT_1D_SIZE 2\n", "exactly one"),
    ("LUT_3D_SIZE 1\n0 0 0\n", "out of range"),
    ("LUT_3D_SIZE 2\n" + "0 0\n" * 8, "three numbers"),
])
def test_bad_files_are_rejected_with_a_reason(text, message):
    with pytest.raises(ValueError, match=message):
        parse_cube(text)


def test_the_library_imports_lists_and_loads_by_name():
    name = import_lut(WARM.encode(), "Warm Film (v2).cube")
    assert name == "Warm Film (v2)" and list_luts() == ["Warm Film (v2)"]
    assert (lut_dir() / "Warm Film (v2).cube").is_file()
    assert load_lut(name).size == 17
    with pytest.raises(ValueError, match="not found"):
        load_lut("Elsewhere")
    with pytest.raises(ValueError, match="Invalid"):
        load_lut("../secret")
    with pytest.raises(ValueError, match="not a .cube"):
        import_lut(b"\x89PNG\x00\x01", "photo.cube")
    assert not is_valid_name("../x") and not is_valid_name("a/b") and is_valid_name("Kodak 2383")


def test_options_and_recipes_carry_only_library_names():
    assert parse_options({"lut": "Warm", "lut_amount": 40}) == {"lut": "Warm", "lut_amount": 40}
    assert parse_options({"lut": ""}) == {"lut": None}
    with pytest.raises(OptionsError, match="LUT library"):
        parse_options({"lut": "C:\\luts\\warm.cube"})
    recipe = sanitize_recipe({"lut": "../../etc/passwd", "lut_amount": 300})
    assert recipe["lut"] == "" and recipe["lut_amount"] == 100


def _process(options: dict, image: Image.Image, name="photo.png") -> Image.Image:
    from app.backend import app

    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    response = app.test_client().post(
        "/process", data={"image": (io.BytesIO(buffer.getvalue()), name), "options": json.dumps(options)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.data[:300]
    return Image.open(io.BytesIO(response.data))


def test_the_pipeline_applies_a_library_lut_with_its_strength():
    name = import_lut(INVERT.encode(), "Invert.cube")
    image = Image.new("RGB", (8, 8), (200, 100, 20))
    full = np.asarray(_process({"lut": name, "output_format": "PNG"}, image).convert("RGB"))[4, 4]
    assert np.abs(full.astype(int) - [55, 155, 235]).max() <= 1
    half = np.asarray(_process({"lut": name, "lut_amount": 50, "output_format": "PNG"}, image).convert("RGB"))[4, 4]
    assert np.abs(half.astype(int) - [128, 128, 128]).max() <= 1
    from app.backend import app

    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    response = app.test_client().post(
        "/process", data={"image": (io.BytesIO(buffer.getvalue()), "x.png"), "options": json.dumps({"lut": "Missing"})},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400 and b"LUT not found: Missing" in response.data


def test_a_16_bit_export_runs_the_lut_on_float_pixels():
    import cv2

    from app.backend import app
    from app.export import _png16

    name = import_lut(INVERT.encode(), "Invert.cube")
    gradient = np.tile(np.linspace(0, 65535, 64).astype(np.uint16), (4, 1))
    source = _png16(np.stack([gradient] * 3, axis=-1), None, None)
    response = app.test_client().post(
        "/process",
        data={"image": (io.BytesIO(source), "g.png"),
              "options": json.dumps({"lut": name, "export": {"format": "PNG", "bit_depth": 16}})},
        content_type="multipart/form-data",
    )
    pixels = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert pixels.dtype == np.uint16
    assert np.abs(pixels[0, :, 0].astype(int) - (65535 - gradient[0].astype(int))).max() <= 2  # no 8-bit steps


def test_the_cli_imports_a_cube_file_or_uses_a_library_name(tmp_path):
    cube = tmp_path / "Teal Orange.cube"
    cube.write_text(WARM, encoding="utf-8")
    args = cli.build_parser().parse_args(["--input", "a.jpg", "--output", "b.jpg", "--lut", str(cube), "--lut-amount", "60"])
    assert cli.options_from_args(args) == {"lut": "Teal Orange", "lut_amount": 60.0}
    args = cli.build_parser().parse_args(["--input", "a.jpg", "--output", "b.jpg", "--lut", "Teal Orange"])
    assert cli.options_from_args(args) == {"lut": "Teal Orange"}


@pytest.mark.slow
def test_the_color_tab_imports_a_lut_selects_it_and_applies_it(monkeypatch):
    from tests.test_frontend_unified_app import _fake_process_via_backend, canvas_hash, open_tab, run_with_uploaded_image

    from app import backend_client

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    at = run_with_uploaded_image()
    open_tab(at, "🎨 Curves & Color")
    at.run(timeout=30)
    before = canvas_hash(at)
    assert at.selectbox(key="lut-name").value == "" and at.slider(key="lut-amount").disabled
    at.file_uploader(key="lut-upload").set_value(("Bleach Bypass.cube", INVERT.encode(), "application/octet-stream"))
    at.run(timeout=30)
    at.run(timeout=30)
    assert list_luts() == ["Bleach Bypass"]
    assert at.selectbox(key="lut-name").value == "Bleach Bypass"  # selected right after the import
    assert not at.slider(key="lut-amount").disabled
    assert canvas_hash(at) != before
