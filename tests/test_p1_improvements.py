"""Regression tests for the P1 items of the 2026-09 improvement report (see docs/PROJECT_STATUS.md,
"Recently Fixed"): lazy pywebview import and desktop Streamlit flags (T1/K1, K2 self-test), face
detection on OpenCV 5 (B6), OpenCV HSL conversion (P4), AI Upscale as an export action (P6),
preset/recipe/project hardening and one data dir (B10), EXIF/ICC through the pipeline and every
export (B3), lazy export (F6), per-rerun caching (P2), one pipeline for backend/CLI/batch (B1), the
options schema (B2), and the proxy preview (P1)."""

import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from app import backend_client
from app.backend import app as flask_app, process_image
from tests.test_frontend_unified_app import APP_PATH, _FakeFileStorage, _fake_process_via_backend

ASSETS = Path(__file__).parent / "assets"


def _jpeg_with_exif(size=(64, 48), color=(120, 100, 80), gps=True) -> bytes:
    exif = Image.Exif()
    exif[0x010F] = "TestMaker"
    exif[0x0110] = "TestModel"
    exif[0x8298] = "(c) Tester"
    exif.get_ifd(0x8769)[0x9003] = "2024:05:06 07:08:09"
    if gps:
        exif.get_ifd(0x8825)[0x0001] = "N"
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG", exif=exif.tobytes())
    return buffer.getvalue()


def _post(options, image=None, filename="in.jpg"):
    return flask_app.test_client().post(
        "/process",
        data={"image": (io.BytesIO(image or _jpeg_with_exif()), filename), "options": options if isinstance(options, str) else json.dumps(options)},
        content_type="multipart/form-data",
    )


# --- T1 / K1 / K2: desktop launcher -------------------------------------------------------------

def test_desktop_module_imports_without_pywebview(monkeypatch):
    monkeypatch.setitem(sys.modules, "webview", None)  # any `import webview` now raises ImportError
    sys.modules.pop("app.desktop", None)
    desktop = importlib.import_module("app.desktop")
    assert desktop._log_path("x.log").name == "x.log"


def test_desktop_streamlit_children_get_theme_telemetry_and_upload_flags():
    from app.config import MAX_UPLOAD_BYTES
    from app.desktop import _streamlit_server_flags

    flags = _streamlit_server_flags()
    pairs = dict(zip(flags[::2], flags[1::2]))
    assert pairs["--theme.base"] == "dark"
    assert pairs["--browser.gatherUsageStats"] == "false"
    assert pairs["--server.maxUploadSize"] == str(MAX_UPLOAD_BYTES // (1024 * 1024))


def test_desktop_self_test_reports_modules_and_capabilities(tmp_path, monkeypatch):
    from app import desktop

    report_path = tmp_path / "report.json"
    code = desktop._run_self_test(["PhotoStudio.exe", "--self-test", "--self-test-report", str(report_path)])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert all(status == "ok" for status in report["modules"].values()), report["modules"]
    assert report["capabilities"]["face_detection"] in ("yunet", "haar")
    assert code == (0 if report["capabilities"]["rawpy"] and report["capabilities"]["lensfun_lenses"] > 0 else 1)


def test_desktop_startup_failure_shows_one_dialog_and_logs_the_traceback(monkeypatch):
    """A failed start shows the app's own error box and exits with code 1 -- re-raising made the
    windowed exe's PyInstaller bootloader show a second "Unhandled exception" box."""
    from app import desktop
    from app.config import user_data_dir

    shown = []
    monkeypatch.setattr(desktop.sys, "argv", ["PhotoStudio.exe"])
    monkeypatch.setattr(desktop, "_acquire_single_instance", lambda: object())
    monkeypatch.setattr(desktop, "_prepare_child_environment", lambda argv: None)
    monkeypatch.setattr(desktop, "_start_backend", lambda: (_ for _ in ()).throw(RuntimeError("backend exploded")))
    monkeypatch.setattr(desktop, "_show_error", shown.append)

    with pytest.raises(SystemExit) as exit_info:
        desktop.main()

    assert exit_info.value.code == 1
    assert len(shown) == 1 and "backend exploded" in shown[0]
    log_text = (user_data_dir() / "logs" / "desktop-startup-error.log").read_text(encoding="utf-8")
    assert "RuntimeError: backend exploded" in log_text and "Traceback" in log_text


def test_gpu_build_keeps_only_the_headless_opencv():
    """basicsr/realesrgan pull in opencv-python, which shares the cv2 package with the app's
    opencv-python-headless; the gpu build must end with only the headless one installed."""
    script = (Path(__file__).resolve().parent.parent / "scripts" / "build_release.ps1").read_text(encoding="utf-8")
    uninstall = script.index("pip uninstall -y opencv-python")
    reinstall = script.index('"--force-reinstall", "--no-deps", "opencv-python-headless"')
    assert script.index("& $VenvPython -m pip install @requirements") < uninstall < reinstall < script.index("-m PyInstaller")
    assert 'if ($BuildProfile -eq "gpu")' in script[uninstall - 600 : uninstall]


# --- B6: face detection ---------------------------------------------------------------------------

def test_face_detection_finds_a_real_face_on_the_installed_opencv():
    """tests/assets/portrait_public_domain.jpg: Abraham Lincoln, Alexander Gardner, 1863 -- public
    domain (Wikimedia Commons, "Abraham Lincoln O-77 matte collodion print"). Fails loudly if face
    detection is dead again, instead of the old test's "returns a list" pass."""
    from app.diagnostics import check_environment
    from app.portrait_tools import detect_faces, face_detection_backend

    assert face_detection_backend() != "none"
    portrait = Image.open(ASSETS / "portrait_public_domain.jpg")
    faces = detect_faces(portrait)
    assert len(faces) == 1
    # Detection on a proxy maps boxes back to the original's coordinates.
    big = portrait.resize((portrait.width * 5, portrait.height * 5))
    (x, y, w, h), = detect_faces(big)
    assert w > 5 * 30 and x > 0 and y > 0
    assert any(check.name == "Face detection" and check.status == "ok" for check in check_environment())


# --- P4: OpenCV HSL -------------------------------------------------------------------------------

def test_opencv_hsl_conversion_matches_the_reference_numpy_math():
    from app.professional_color import _hsl_to_rgb_reference, _rgb_to_hsl_reference, hsl_to_rgb, rgb_to_hsl

    rgb = np.random.default_rng(1).random((50, 40, 3), dtype=np.float32)
    rgb[0] = 0.5  # greys: hue undefined -> 0 in both
    fast, reference = rgb_to_hsl(rgb), _rgb_to_hsl_reference(rgb)
    hue_error = np.abs(((fast[..., 0] - reference[..., 0] + 180) % 360) - 180)
    assert hue_error.max() < 1e-3
    assert np.abs(fast[..., 1:] - reference[..., 1:]).max() < 1e-4
    assert np.abs(hsl_to_rgb(fast) - _hsl_to_rgb_reference(reference)).max() < 1e-4


# --- B2: options schema ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "options, fragment",
    [
        ({"exposure": None}, "exposure must not be null"),
        ("[1, 2]", "options must be a JSON object"),
        ({"exposre": 1.5}, "Unknown option(s): exposre"),
        ({"hsl_bands": {"Teal": {"hue": 10}}}, "Unknown HSL band(s): Teal"),
        ({"ai_upscale_factor": 3}, "ai_upscale_factor must be one of 2, 4"),
        ({"exposure": 50}, "exposure must be between"),
        ({"schema_version": 99}, "schema_version"),
    ],
)
def test_bad_options_get_a_json_400_naming_the_problem(options, fragment):
    response = _post(options)
    assert response.status_code == 400
    assert fragment in response.get_json()["error"]


def test_unexpected_backend_errors_are_json_not_html(monkeypatch):
    import app.backend as backend

    def boom(*args, **kwargs):
        raise ZeroDivisionError("unexpected")

    monkeypatch.setattr(backend, "run_pipeline", boom)
    response = _post({})
    assert response.status_code == 500
    assert response.get_json()["error"] == "Internal error: ZeroDivisionError"
    assert response.get_json()["request_id"]

    def out_of_memory(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(backend, "run_pipeline", out_of_memory)
    assert _post({}).status_code == 507


def test_any_aspect_ratio_is_accepted():
    response = _post({"aspect": "2:1", "output_format": "PNG"})
    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.data)).size == (64, 32)


# --- B1: one pipeline -------------------------------------------------------------------------------

def test_cli_and_backend_produce_identical_pixels_for_the_same_edit(tmp_path, monkeypatch):
    from app import cli

    source = tmp_path / "in.png"
    Image.fromarray(np.random.default_rng(2).integers(0, 255, (40, 60, 3), dtype=np.uint8)).save(source)
    output = tmp_path / "out.png"
    monkeypatch.setattr(sys, "argv", [
        "cli", "--input", str(source), "--output", str(output), "--exposure", "1.2", "--contrast", "1.3",
        "--saturation", "1.4", "--temperature", "20", "--texture", "30", "--vibrance", "25", "--dehaze", "20",
        "--sharpen", "--aspect", "1", "1",
    ])
    cli.main()
    options = {"exposure": 1.2, "contrast": 1.3, "saturation": 1.4, "temperature": 20, "texture": 30, "vibrance": 25, "dehaze": 20, "sharpen": True, "aspect": "1:1"}
    backend_result = process_image(_FakeFileStorage("in.png", source.read_bytes()), options)
    assert np.array_equal(np.asarray(Image.open(output).convert("RGB")), np.asarray(backend_result))


def test_cli_input_dir_applies_every_flag_and_keeps_exif(tmp_path, monkeypatch):
    from app import cli

    folder, out = tmp_path / "in", tmp_path / "out"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(_jpeg_with_exif())
    monkeypatch.setattr(sys, "argv", ["cli", "--input-dir", str(folder), "--output-dir", str(out), "--exposure", "1.4", "--aspect", "1", "1"])
    cli.main()
    result = Image.open(out / "a.jpg")
    assert result.size == (48, 48)  # --aspect used to be silently ignored in folder mode
    assert result.getexif()[0x010F] == "TestMaker"


def test_cli_recipe_file_and_auto_enhance(tmp_path, monkeypatch):
    from app import cli

    recipe = tmp_path / "look.json"
    recipe.write_text(json.dumps({"version": 1, "presets": {"Warm": {"temperature": 40, "exposure": 5.0, "ai_upscale": True}}}), encoding="utf-8")
    args = cli.build_parser().parse_args(["--input", "x.png", "--output", "y.png", "--recipe", str(recipe), "--auto-enhance", "--exposure", "1.3"])
    options = cli.options_from_args(args)
    assert options["temperature"] == 40
    assert options["exposure"] == 1.3  # an explicit flag beats both the recipe and --auto-enhance
    assert options["denoise"] == 3 and options["sharpen"] is True
    assert "ai_upscale" not in options


# --- B3: EXIF/ICC through the stage hops and exports ------------------------------------------------

def test_exif_and_icc_survive_both_stage_hops():
    from app.professional_color import srgb_icc_bytes

    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[0x010F] = "TestMaker"
    Image.new("RGB", (32, 24), (90, 100, 110)).save(buffer, format="JPEG", exif=exif.tobytes(), icc_profile=srgb_icc_bytes())
    stage1 = _post({"stop_after": "detail", "exposure": 1.1}, buffer.getvalue())
    assert stage1.status_code == 200
    stage1_image = Image.open(io.BytesIO(stage1.data))
    assert stage1_image.getexif()[0x010F] == "TestMaker"
    assert stage1_image.info.get("icc_profile")
    stage2 = flask_app.test_client().post(
        "/process",
        data={"image": (io.BytesIO(stage1.data), "stage_a.png"), "options": json.dumps({"resume_from_detail": True})},
        content_type="multipart/form-data",
    )
    stage2_image = Image.open(io.BytesIO(stage2.data))
    assert stage2_image.getexif()[0x010F] == "TestMaker"
    assert stage2_image.info.get("icc_profile")


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "TIFF"])
def test_export_metadata_policies(fmt):
    from app.export import encode_export

    source = Image.open(io.BytesIO(_jpeg_with_exif()))
    keep_all, _, _ = encode_export(source, {"format": fmt, "long_edge": 32})
    exif = Image.open(io.BytesIO(keep_all)).getexif()
    assert exif[0x010F] == "TestMaker" and exif.get(0x0112) == 1
    assert exif.get_ifd(0x8769)[0xA002] == 32  # PixelXDimension is the exported size
    assert exif.get_ifd(0x8825)
    no_gps = Image.open(io.BytesIO(encode_export(source, {"format": fmt, "remove_gps": True})[0])).getexif()
    assert not no_gps.get_ifd(0x8825) and no_gps[0x010F] == "TestMaker"
    stripped = Image.open(io.BytesIO(encode_export(source, {"format": fmt, "metadata": "none"})[0])).getexif()
    assert 0x010F not in stripped


def test_camera_only_policy_keeps_copyright_and_capture_time():
    from app.export import encode_export

    source = Image.open(io.BytesIO(_jpeg_with_exif()))
    exif = Image.open(io.BytesIO(encode_export(source, {"metadata": "camera"})[0])).getexif()
    assert exif[0x8298] == "(c) Tester"
    assert exif.get_ifd(0x8769)[0x9003] == "2024:05:06 07:08:09"
    assert not exif.get_ifd(0x8825)


def test_cli_jpeg_delivery_keeps_exif(tmp_path):
    from app.export import write_export
    from app.image_processor import PhotoProcessor

    source = tmp_path / "in.jpg"
    source.write_bytes(_jpeg_with_exif())
    output = write_export(PhotoProcessor(source).image, tmp_path / "out.jpg", {"metadata": "all"})
    assert Image.open(output).getexif()[0x010F] == "TestMaker"


# --- B10: presets, recipes, projects, data dir ------------------------------------------------------

def test_sanitize_recipe_clamps_coerces_and_drops_unknown_keys():
    from app.edit_schema import sanitize_recipe

    clean = sanitize_recipe({
        "exposure": 5.0, "highlights": "loud", "preset": "Nope", "denoise": 3.6, "ai_upscale": True, "bogus": 1,
        "mask_layers": [{"kind": "Radial", "params": {}, "soften": 99}, {"kind": "Laser"}, "junk"],
        "curve_extra_points": [[10, 20], [999, 1], "x"],
    })
    assert clean["exposure_ev"] == 5.0 and clean["highlights"] == 0 and clean["preset"] == "None" and clean["denoise"] == 4
    assert "ai_upscale" not in clean and "bogus" not in clean
    assert len(clean["mask_layers"]) == 1 and clean["mask_layers"][0]["soften"] == 40
    assert clean["curve_extra_points"] == [[10, 20]]


def test_recipe_controls_match_the_real_widgets():
    """Every widget is generated from the app.controls registry (F15) -- checked against what
    AppTest actually renders: same key, label, range, step and default, so the sanitizer
    (app.edit_schema.RECIPE_CONTROLS, derived from the same registry) can't drift from the UI."""
    from app.controls import CONTROLS
    from app.edit_schema import RECIPE_CONTROLS

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("photo.jpg", _jpeg_with_exif(), "image/jpeg"))
    at.run(timeout=30)
    assert not at.exception
    sliders = {slider.key: slider for slider in at.slider}
    checkboxes = {checkbox.key: checkbox for checkbox in at.checkbox}
    selectboxes = {selectbox.key: selectbox for selectbox in at.selectbox}
    number_inputs = {number.key: number for number in at.number_input}
    color_pickers = {picker.key: picker for picker in at.color_picker}
    checked = 0
    for control in CONTROLS:
        assert RECIPE_CONTROLS[control.key].default == control.default
        if control.widget_type == "slider":
            widget = sliders[control.widget_key]
            assert (widget.min, widget.max, widget.value, widget.label) == (control.minimum, control.maximum, control.default, control.label), control.key
            if control.step is not None:
                assert widget.step == control.step, control.key
        elif control.widget_type == "checkbox":
            assert checkboxes[control.widget_key].value == control.default, control.key
            assert checkboxes[control.widget_key].label == control.label, control.key
        elif control.widget_type == "selectbox":
            assert selectboxes[control.widget_key].value == control.default, control.key
        elif control.widget_type == "number_input":
            widget = number_inputs[control.widget_key]
            assert (widget.min, widget.max, widget.value) == (control.minimum, control.maximum, control.default), control.key
        elif control.widget_type == "color_picker":
            assert color_pickers[control.widget_key].value.upper() == control.default, control.key
        else:
            assert control.widget_type == "state"  # the free-form crop box, set by the crop editor
            continue
        checked += 1
    assert set(RECIPE_CONTROLS) == {control.key for control in CONTROLS}
    assert checked > 60


def test_applying_an_out_of_range_preset_no_longer_crashes_the_page(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    from app.custom_presets import PresetStore

    PresetStore().save("Hand edited", {"exposure": 5.0, "contrast": "high", "future_key": 1})
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("photo.jpg", _jpeg_with_exif(), "image/jpeg"))
    at.run(timeout=30)
    at.button(key="p-load-btn").click()
    at.run(timeout=30)
    at.run(timeout=30)
    assert not at.exception
    assert at.slider(key="t-ev").value == 5.0  # migrated 2.2*log2(5) EV, clamped to the slider
    assert at.slider(key="t-contrast").value == 0  # "high" isn't a number -> default


def test_corrupt_presets_file_degrades_to_an_error_with_a_reset(monkeypatch):
    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    from app.custom_presets import PresetStore

    store = PresetStore()
    store.path.write_text("{truncated", encoding="utf-8")
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("photo.jpg", _jpeg_with_exif(), "image/jpeg"))
    at.run(timeout=30)
    assert not at.exception
    assert any("can't be read" in error.value for error in at.error)
    at.button(key="p-reset-btn").click()
    at.run(timeout=30)
    at.run(timeout=30)
    assert not at.exception
    assert store.names() == []
    assert list(store.path.parent.glob("presets.json.corrupt-*"))


def test_preset_writes_are_atomic(monkeypatch):
    from app import config
    from app.custom_presets import PresetStore

    store = PresetStore()
    store.save("A", {"exposure": 1.1})

    def failing_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(config.os, "replace", failing_replace)
    with pytest.raises(OSError):
        store.save("B", {"exposure": 1.2})
    assert store.names() == ["A"]  # the old file is intact, not truncated
    assert not list(store.path.parent.glob(".presets.json.*.tmp"))


def test_import_presets_sanitizes_recipes():
    from app.custom_presets import PresetStore
    from app.preset_exchange import import_presets

    store = PresetStore()
    import_presets(store, json.dumps({"version": 1, "presets": {"X": {"exposure": 9, "junk": 1}}}))
    assert store.get("X") == {"exposure_ev": 5.0}  # migrated to EV, clamped to the slider
    with pytest.raises(ValueError, match="newer version"):
        import_presets(store, json.dumps({"version": 1, "recipe_version": 99, "presets": {"Y": {}}}))


def test_presets_migrate_from_the_legacy_folder(monkeypatch, tmp_path):
    from app.custom_presets import PresetStore

    home = tmp_path / "home"
    (home / ".photo-studio").mkdir(parents=True)
    (home / ".photo-studio" / "presets.json").write_text(json.dumps({"Old": {"exposure": 1.2}}), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("PHOTO_STUDIO_DATA_DIR", str(tmp_path / "new-data"))
    store = PresetStore()
    assert store.path == tmp_path / "new-data" / "presets.json"
    assert store.names() == ["Old"]


def test_project_manifest_is_versioned_and_checked_at_save(monkeypatch):
    import zipfile

    from app import project_bundle
    from app.project_bundle import ProjectBundle

    bundle = ProjectBundle("a.jpg", _jpeg_with_exif(), {"exposure": 1.1})
    content = bundle.to_bytes()
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(content)).read("project.json"))
    assert manifest["app_version"] and manifest["recipe_version"] == 2
    assert ProjectBundle.load(content).recipe == {"exposure": 1.1}

    manifest["recipe_version"] = 99
    newer = io.BytesIO()
    with zipfile.ZipFile(newer, "w") as archive:
        archive.writestr("project.json", json.dumps(manifest))
        archive.writestr(manifest["files"]["source"], _jpeg_with_exif())
    with pytest.raises(ValueError, match="newer version"):
        ProjectBundle.load(newer.getvalue())

    def too_big(content, **kwargs):
        raise ValueError("Project expands beyond the configured size limit")

    monkeypatch.setattr(project_bundle, "validate_project_archive", too_big)
    with pytest.raises(ValueError, match="couldn't be reopened"):
        bundle.to_bytes()


# --- P6: AI Upscale safety --------------------------------------------------------------------------

def test_upscale_output_is_checked_before_inference(monkeypatch):
    from app.image_processor import PhotoProcessor

    monkeypatch.setattr("app.ai_upscale.ai_upscale_available", lambda: True)
    monkeypatch.setattr("app.pipeline.MAX_UPSCALE_OUTPUT_PIXELS", 1000)
    monkeypatch.setattr(PhotoProcessor, "apply_ai_upscale", lambda self, factor=2: pytest.fail("inference must not run"))
    response = _post({"ai_upscale": True, "ai_upscale_factor": 4})
    assert response.status_code == 400
    assert "MP limit" in response.get_json()["error"]


def test_backend_client_turns_an_oversized_result_into_a_clean_error(monkeypatch):
    class _Response:
        ok = True

        def __init__(self):
            buffer = io.BytesIO()
            Image.new("L", (100, 100)).save(buffer, format="PNG")
            self.content = buffer.getvalue()

    monkeypatch.setattr(backend_client.requests, "post", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(RuntimeError, match="too large"):
        backend_client.process_via_backend(b"x", "x.png", {})


# --- P2 / F6 / P1: frontend ---------------------------------------------------------------------------

def _app_with(monkeypatch, content, name="photo.jpg", stub=None):
    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", stub or _fake_process_via_backend)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value((name, content, "image/jpeg"))
    at.run(timeout=60)
    return at


def test_unrelated_reruns_do_not_redecode_encode_or_detect_again(monkeypatch):
    import app.export as export_module
    import app.portrait_tools as portrait_module
    from app.image_processor import PhotoProcessor

    counts = {"decode": 0, "encode": 0, "faces": 0}
    original_from_bytes = PhotoProcessor.from_bytes.__func__
    original_encode = export_module.encode_export
    original_detect = portrait_module.detect_faces

    def counting_from_bytes(cls, content, filename):
        if filename == "unique-p2.jpg":
            counts["decode"] += 1
        return original_from_bytes(cls, content, filename)

    def counting_encode(*args, **kwargs):
        counts["encode"] += 1
        return original_encode(*args, **kwargs)

    def counting_detect(image):
        counts["faces"] += 1
        return original_detect(image)

    monkeypatch.setattr(PhotoProcessor, "from_bytes", classmethod(counting_from_bytes))
    monkeypatch.setattr(export_module, "encode_export", counting_encode)
    monkeypatch.setattr(portrait_module, "detect_faces", counting_detect)
    at = _app_with(monkeypatch, _jpeg_with_exif(color=(11, 22, 33)), "unique-p2.jpg")
    decodes, faces = counts["decode"], counts["faces"]
    for value in (True, False, True):
        at.checkbox(key="waveform-toggle").set_value(value)
        at.run(timeout=30)
    assert not at.exception
    assert counts["decode"] == decodes  # the source is decoded once per upload
    assert counts["faces"] == faces <= 1
    assert counts["encode"] == 0  # the Save button encodes nothing until clicked


def test_save_button_writes_exif_and_icc_when_clicked(monkeypatch):
    import streamlit as st

    captured = {}
    original = st.download_button

    def capture(label, data, *args, **kwargs):
        if kwargs.get("key") == "exp-download-btn":
            captured["data"] = data
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture)
    at = _app_with(monkeypatch, _jpeg_with_exif(color=(44, 55, 66)))
    assert callable(captured["data"])
    saved = Image.open(io.BytesIO(captured["data"]()))
    assert saved.getexif()[0x010F] == "TestMaker"
    assert saved.getexif().get_ifd(0x8769)[0x9003] == "2024:05:06 07:08:09"
    at.selectbox(key="exp-metadata").set_value("none")
    at.run(timeout=30)
    assert 0x010F not in Image.open(io.BytesIO(captured["data"]())).getexif()


def test_large_photos_preview_through_a_proxy_and_export_at_full_resolution(monkeypatch):
    import streamlit as st

    seen_sizes = []

    def recording_stub(image_bytes_, filename, options, *args, **kwargs):
        seen_sizes.append(Image.open(io.BytesIO(image_bytes_)).size)
        return _fake_process_via_backend(image_bytes_, filename, options, *args, **kwargs)

    exports = []

    def fake_export(image_bytes_, filename, options, settings, *args):
        from app.export import encode_export

        exports.append(options)
        return encode_export(process_image(_FakeFileStorage(filename, image_bytes_), options), settings)[0]

    monkeypatch.setattr(backend_client, "export_via_backend", fake_export)
    captured = {}
    original = st.download_button

    def capture(label, data, *args, **kwargs):
        if kwargs.get("key") == "exp-download-btn":
            captured["data"] = data
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture)
    buffer = io.BytesIO()
    Image.new("RGB", (4000, 3000), (120, 90, 60)).save(buffer, format="JPEG")
    at = _app_with(monkeypatch, buffer.getvalue(), "big.jpg", recording_stub)
    assert not at.exception
    assert max(seen_sizes[0]) == 2048  # Stage 1 ran on the proxy
    markdown = next(m.value for m in at.markdown if "Live View" in m.value)
    assert "(4000×3000)" in markdown  # the label shows the export's size, not the proxy's

    at.selectbox(key="crop-aspect").set_value("1:1 Square")
    at.run(timeout=60)
    assert "(3000×3000)" in next(m.value for m in at.markdown if "Live View" in m.value)
    exported = Image.open(io.BytesIO(captured["data"]()))
    assert exported.size == (3000, 3000)
    assert "preview_scale" not in exports[-1]

    at.selectbox(key="cv-zoom").set_value(2.0)
    at.run(timeout=60)
    at.run(timeout=60)
    assert seen_sizes[-1] == (4000, 3000)  # the 1:1 zoom views render real pixels


def test_skin_smoothing_can_be_limited_to_the_detected_faces():
    """B6: with "Only around detected faces" (the default) a skin-coloured, textured patch far from
    the face keeps its texture; without it -- or with no face in the photo -- it is smoothed."""
    from app.pipeline import run_pipeline
    from app.portrait_tools import detect_faces, face_detection_backend, face_region_mask

    if face_detection_backend() == "none":
        pytest.skip("no face detector in this OpenCV build")
    portrait = Image.open(ASSETS / "portrait_public_domain.jpg").convert("RGB")
    canvas = Image.new("RGB", (portrait.width * 2, portrait.height), (90, 90, 90))
    canvas.paste(portrait, (0, 0))
    rng = np.random.default_rng(3)
    patch = np.clip(np.array([200, 150, 120]) + rng.normal(0, 8, (120, 120, 3)), 0, 255).astype(np.uint8)
    patch_box = (portrait.width + 100, 150, portrait.width + 220, 270)  # well away from the face
    canvas.paste(Image.fromarray(patch), patch_box[:2])
    faces = detect_faces(canvas)
    assert len(faces) == 1
    assert np.asarray(face_region_mask(canvas.size, faces).crop(patch_box)).max() == 0

    def patch_std(**options):
        result = run_pipeline(canvas, dict({"portrait_smoothing": 1.0}, **options))
        return float(np.asarray(result.crop(patch_box), dtype=np.float32).std(axis=(0, 1)).mean())

    original = float(np.asarray(canvas.crop(patch_box), dtype=np.float32).std(axis=(0, 1)).mean())  # texture per channel
    assert patch_std() == pytest.approx(original, abs=0.5)  # default: faces only
    assert patch_std(portrait_faces_only=False) < original * 0.8
