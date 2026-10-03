"""Regression tests for the P2 items (and the remaining P1 work) of the 2026-09 improvement report
-- see docs/PROJECT_STATUS.md, "Recently Fixed". Frontend behaviour is driven through AppTest, the
same harness tests/test_frontend_unified_app.py uses."""

import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from app import ai_upscale
from tests.synthetic_assets import encode, photo_like


# --- P7: Real-ESRGAN weights, tiling and serialization ---------------------------------------------

class _FakeResponse:
    def __init__(self, content: bytes):
        self._stream = io.BytesIO(content)
        self.headers = {"Content-Length": str(len(content))}

    def read(self, size=-1):
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_weights_download_is_verified_and_atomic(monkeypatch):
    content = b"pretend model weights" * 1000
    monkeypatch.setattr(ai_upscale, "WEIGHTS_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(ai_upscale, "WEIGHTS_SIZE_BYTES", len(content))
    seen = []

    path = ai_upscale.download_weights(progress=lambda done, total: seen.append((done, total)), opener=lambda url: _FakeResponse(content))

    assert path.read_bytes() == content and ai_upscale.weights_present()
    assert seen[-1] == (len(content), len(content))
    assert not list(path.parent.glob("*.part"))


def test_a_tampered_download_is_discarded(monkeypatch):
    monkeypatch.setattr(ai_upscale, "WEIGHTS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="checksum"):
        ai_upscale.download_weights(opener=lambda url: _FakeResponse(b"not the model"))
    folder = ai_upscale.weights_dir()
    assert not ai_upscale.weights_path().exists()
    assert not list(folder.glob("*.part"))


def test_inference_is_tiled_and_shrinks_tiles_when_vram_is_low(monkeypatch):
    assert ai_upscale._tile_size("cpu") == ai_upscale._TILE == 512
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1 * 1024**3, 16 * 1024**3))
    assert ai_upscale._tile_size("cuda") == ai_upscale._SMALL_TILE
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (12 * 1024**3, 16 * 1024**3))
    assert ai_upscale._tile_size("cuda") == ai_upscale._TILE


def test_upscale_runs_under_the_inference_lock(monkeypatch):
    held = []

    class _Upsampler:
        def enhance(self, bgr, outscale):
            held.append(ai_upscale._INFERENCE_LOCK.locked())
            return np.repeat(np.repeat(bgr, outscale, axis=0), outscale, axis=1), None

    monkeypatch.setattr(ai_upscale, "ai_upscale_available", lambda: True)
    monkeypatch.setattr(ai_upscale, "_load_upsampler", lambda: _Upsampler())
    result = ai_upscale.upscale(Image.new("RGB", (4, 3), (1, 2, 3)), 2)
    assert result.size == (8, 6) and held == [True]
    assert not ai_upscale._INFERENCE_LOCK.locked()


# --- B5: Clarity and Highlights/Shadows ---------------------------------------------------------------

def _halves(dark=40, bright=220, size=(200, 120), noise=6, seed=1):
    rng = np.random.default_rng(seed)
    array = np.full((size[1], size[0], 3), float(dark))
    array[:, size[0] // 2:] = bright
    array += rng.normal(0, noise, array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def test_clarity_is_local_contrast_not_global_contrast():
    from app.image_processor import PhotoProcessor

    source = photo_like((300, 200), seed=3)
    clarity = np.asarray(PhotoProcessor.from_image(source).apply_texture_clarity(0, 80).image, float)
    contrast = np.asarray(PhotoProcessor.from_image(source).apply_contrast(1.8).image, float)
    original = np.asarray(source, float)
    # Global contrast moves the overall tonal range a lot; clarity barely moves the average...
    assert abs(clarity.mean() - original.mean()) < 2.0
    assert np.abs(contrast - original).mean() > 3 * np.abs(clarity - original).mean()
    # ...but does raise mid-scale local contrast, and a negative value lowers it.
    soft = np.asarray(PhotoProcessor.from_image(source).apply_texture_clarity(0, -80).image, float)
    import cv2

    def local_detail(a):
        lum = a.mean(axis=2)
        return np.abs(lum - cv2.GaussianBlur(lum, (0, 0), 4)).mean()

    assert local_detail(clarity) > local_detail(original) > local_detail(soft)


def test_clarity_and_tone_look_the_same_on_a_proxy_and_at_full_resolution():
    from app.image_processor import PhotoProcessor

    full = photo_like((1000, 700), seed=4)
    proxy = full.resize((500, 350), Image.Resampling.LANCZOS)

    def edit(image):
        return PhotoProcessor.from_image(image).apply_texture_clarity(0, 80).apply_highlights_shadows(-50, 60).image

    downscaled = np.asarray(edit(full).resize((500, 350), Image.Resampling.LANCZOS), float)
    difference = np.abs(downscaled - np.asarray(edit(proxy), float))
    assert difference.mean() < 1.5


def test_shadows_lift_dark_regions_without_haloing_the_bright_side():
    from app.image_processor import PhotoProcessor

    source = _halves()
    lifted = np.asarray(PhotoProcessor.from_image(source).apply_highlights_shadows(0, 80).image, float)
    original = np.asarray(source, float)
    assert lifted[:, :80].mean() - original[:, :80].mean() > 20  # the dark half is lifted
    assert abs(lifted[:, 110:].mean() - original[:, 110:].mean()) < 2  # the bright half isn't
    assert np.abs(lifted[:, 101:104] - original[:, 101:104]).mean() < 6  # no halo at the edge
    # Detail (the noise texture) survives the lift instead of being flattened.
    assert lifted[:, 10:80].std() > 0.8 * original[:, 10:80].std()


# --- F4 / B9: rotate, flip, straighten, numeric and custom crop, constrain crop ------------------------

from tests.test_frontend_unified_app import canvas_dims, open_tab, run_with_uploaded_image, ui_source  # noqa: E402  (also applies the backend stub fixture below)
from tests.test_frontend_unified_app import _stub_backend  # noqa: E402,F401


def test_rotate_right_swaps_the_output_dimensions_and_undo_restores_them():
    at = run_with_uploaded_image(size=(40, 30))
    at.button(key="geo-rot-right-btn").click()
    at.run(timeout=30)
    assert not at.exception
    assert at.selectbox(key="geo-rotate").value == 90
    assert canvas_dims(at) == (30, 40)
    at.button(key="hist-undo-btn").click()
    at.run(timeout=30)
    assert canvas_dims(at) == (40, 30)


def test_numeric_crop_fields_set_the_free_form_box():
    at = run_with_uploaded_image(size=(40, 30))
    at.selectbox(key="crop-aspect").set_value("Free-form")
    at.run(timeout=30)
    at.number_input(key="crop-num-width").set_value(20)
    at.run(timeout=30)
    at.number_input(key="crop-num-height").set_value(10)
    at.run(timeout=30)
    assert not at.exception
    assert canvas_dims(at) == (20, 10)


def test_custom_ratio_with_orientation_swap():
    at = run_with_uploaded_image(size=(40, 30))
    at.selectbox(key="crop-aspect").set_value("Custom")
    at.run(timeout=30)
    at.number_input(key="crop-custom-w").set_value(2)
    at.number_input(key="crop-custom-h").set_value(1)
    at.run(timeout=30)
    assert canvas_dims(at) == (40, 20)
    at.checkbox(key="crop-swap").set_value(True)
    at.run(timeout=30)
    assert not at.exception
    assert canvas_dims(at) == (15, 30)


def test_auto_straighten_sets_the_angle_from_the_photos_lines(monkeypatch):
    import app.geometry as geometry

    monkeypatch.setattr(geometry, "estimate_straighten_angle", lambda image: 3.5)
    at = run_with_uploaded_image(size=(40, 30))
    at.button(key="geo-auto-btn").click()
    at.run(timeout=30)
    at.run(timeout=30)
    assert not at.exception
    assert at.slider(key="geo-straighten").value == 3.5


def test_constrain_crop_leaves_no_black_corners_after_lens_distortion():
    from app.pipeline import expected_output_size, run_pipeline

    source = photo_like((600, 400))
    loose = np.asarray(run_pipeline(source, {"lens_distortion": 35}))
    constrained_image = run_pipeline(source, {"lens_distortion": 35, "constrain_crop": True})
    constrained = np.asarray(constrained_image)
    assert (loose.max(axis=2) < 8).mean() > 0.03
    assert (constrained.max(axis=2) < 8).mean() == 0
    assert constrained_image.size == expected_output_size(source.size, {"lens_distortion": 35, "constrain_crop": True})
    for options in ({"straighten": 8, "constrain_crop": True}, {"lens_perspective_v": 60, "constrain_crop": True}):
        assert (np.asarray(run_pipeline(source, options)).max(axis=2) < 8).mean() == 0


def test_auto_level_measures_a_tilted_horizon():
    from PIL import ImageDraw

    from app.geometry import estimate_straighten_angle, straighten

    base = Image.new("RGB", (800, 600), (200, 200, 200))
    draw = ImageDraw.Draw(base)
    for y in range(100, 600, 80):
        draw.line([(0, y), (800, y)], fill=(20, 20, 20), width=4)
    assert abs(estimate_straighten_angle(straighten(base, 6, border=1)) + 6) < 0.5
    assert estimate_straighten_angle(photo_like()) == 0.0


def test_geometry_and_flips_are_proxy_consistent_and_orientation_correct():
    from app.geometry import apply_geometry

    marker = Image.new("RGB", (40, 20), (0, 0, 0))
    marker.putpixel((0, 0), (255, 0, 0))  # top-left red
    rotated = apply_geometry(marker, {"rotate": 90})
    assert rotated.size == (20, 40) and rotated.getpixel((19, 0)) == (255, 0, 0)  # now top-right
    flipped = apply_geometry(marker, {"flip_horizontal": True})
    assert flipped.getpixel((39, 0)) == (255, 0, 0)


# --- F2: exposure in EV, -100..100 units, migration, per-section reset ---------------------------------

def test_exposure_is_in_stops_of_linear_light():
    from app.image_processor import _linear_to_srgb, _srgb_to_linear, exposure_lut

    assert exposure_lut(0) == list(range(256))
    lut = exposure_lut(1.0)
    # A mid-gray doubles in linear light (well below the highlight shoulder).
    linear_in = _srgb_to_linear(np.array([60 / 255.0]))[0]
    assert abs(lut[60] / 255.0 - _linear_to_srgb(np.array([2 * linear_in]))[0]) < 1 / 255
    assert lut[-1] == 255 and all(a <= b for a, b in zip(lut, lut[1:]))  # monotonic, no clipping wrap


def test_old_recipes_migrate_to_the_new_units():
    from app.edit_schema import migrate_recipe, sanitize_recipe
    from app.recipe_options import recipe_to_options

    old = {"exposure": 1.3, "contrast": 1.15, "saturation": 0.5, "mask_target_exposure": 1.0, "mask_target_saturation": 1.2}
    clean = sanitize_recipe(old)
    assert clean == {"exposure_ev": 0.85, "contrast_amount": 15, "saturation_amount": -50, "mask_target_exposure_ev": 0.0, "mask_target_saturation_amount": 20}
    assert migrate_recipe({"exposure": 2.0}, 1) == {"exposure_ev": 2.2}
    assert migrate_recipe({"exposure_ev": 1.0}, 2) == {"exposure_ev": 1.0}
    # A batch run of an old preset keeps its exact multipliers; a new one converts its amounts.
    assert recipe_to_options(old)["exposure"] == 1.3
    new = recipe_to_options({"exposure_ev": 0.5, "contrast_amount": 20, "mask_target_saturation_amount": -30})
    assert new["exposure_ev"] == 0.5 and new["contrast"] == 1.2 and abs(new["mask_saturation"] - 0.7) < 1e-9


def test_section_reset_only_resets_its_own_controls():
    at = run_with_uploaded_image()
    at.slider(key="t-ev").set_value(1.0)
    at.slider(key="c-temp").set_value(30)
    at.run(timeout=30)
    at.button(key="reset-tone-btn").click()
    at.run(timeout=30)
    assert not at.exception
    assert at.slider(key="t-ev").value == 0.0
    assert at.slider(key="c-temp").value == 30


def test_auto_enhance_recipe_measures_the_photo():
    from app.image_processor import auto_enhance_recipe

    dark = auto_enhance_recipe(Image.new("RGB", (64, 64), (25, 25, 25)))
    assert dark["exposure_ev"] > 1 and dark["denoise"] == 0
    cast = auto_enhance_recipe(Image.new("RGB", (64, 64), (200, 150, 110)))
    assert cast["temperature"] < 0  # a warm cast is cooled toward neutral (positive temperature warms)
    flat = np.random.default_rng(0).normal(128, 30, (64, 64, 3))
    noisy = auto_enhance_recipe(Image.fromarray(np.clip(flat, 0, 255).astype(np.uint8)))
    assert noisy["denoise"] == 3


# --- B8: per-mask adjustment sets, add/subtract/intersect, radial parameters ---------------------------

def test_mask_stack_combine_modes():
    from app.advanced_masks import MaskLayer, MaskStack

    left = Image.new("L", (10, 1), 0)
    left.paste(255, (0, 0, 6, 1))
    right = Image.new("L", (10, 1), 0)
    right.paste(255, (4, 0, 10, 1))

    def combined(mode):
        return list(np.asarray(MaskStack((10, 1)).add(MaskLayer("a", left)).add(MaskLayer("b", right, combine=mode)).combined())[0])

    assert combined("add") == [255] * 10
    assert combined("subtract") == [255] * 4 + [0] * 6
    assert combined("intersect") == [0] * 4 + [255] * 2 + [0] * 4


def test_radial_selection_uses_its_center_radius_and_feather():
    from app.advanced_masks import radial_mask

    mask = np.asarray(radial_mask((100, 100), (0.2, 0.2), 0.15, 0.1))
    assert mask[20, 20] == 255 and mask[80, 80] == 0


def test_two_masks_can_carry_opposite_adjustments_in_one_edit():
    from app.pipeline import run_pipeline

    source = Image.new("RGB", (40, 20), (120, 120, 120))
    left = Image.new("L", (40, 20), 0)
    left.paste(255, (0, 0, 20, 20))
    right = Image.new("L", (40, 20), 0)
    right.paste(255, (20, 0, 40, 20))
    adjustments = [
        {"exposure_ev": 1.0, "contrast": 0, "saturation": 0, "temperature": 0, "tint": 0, "clarity": 0},
        {"exposure_ev": -1.0, "contrast": 0, "saturation": 0, "temperature": 0, "tint": 0, "clarity": 0},
    ]
    result = np.asarray(run_pipeline(source, {"local_adjustments": adjustments}, local_masks=[left, right]))
    assert result[:, :20].mean() > 140 and result[:, 20:].mean() < 100
    with pytest.raises(ValueError, match="local masks"):
        run_pipeline(source, {"local_adjustments": adjustments}, local_masks=[left])


def test_backend_accepts_a_local_masks_file():
    import base64

    from app.backend import app as flask_app

    mask = io.BytesIO()
    Image.new("L", (16, 16), 255).save(mask, format="PNG")
    payload = json.dumps({"masks": [base64.b64encode(mask.getvalue()).decode("ascii")]}).encode()
    options = {"local_adjustments": [{"exposure_ev": -2.0}]}
    response = flask_app.test_client().post(
        "/process",
        data={"image": (io.BytesIO(encode(Image.new("RGB", (16, 16), (150, 150, 150)), "PNG")), "in.png"),
              "local_masks": (io.BytesIO(payload), "local_masks.json"), "options": json.dumps(options)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert np.asarray(Image.open(io.BytesIO(response.data))).mean() < 100


def test_masks_tab_own_adjustments_reach_the_canvas():
    at = run_with_uploaded_image(size=(40, 30))
    at.selectbox(key="m-new-kind").set_value("Radial")
    at.run(timeout=30)
    at.button(key="m-add-btn").click()
    at.run(timeout=30)
    layer_id = at.session_state["mask_layers"][0]["id"]
    before = canvas_hash_of(at)
    at.checkbox(key=f"m-own-{layer_id}").set_value(True)
    at.run(timeout=30)
    at.slider(key=f"m-own-ev-{layer_id}").set_value(1.5)
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["mask_layers"][0]["adjustments"]["exposure_ev"] == 1.5
    assert canvas_hash_of(at) != before
    # The shared Target sliders no longer apply to it, so they're disabled.
    assert at.slider(key="m-tev").disabled


def canvas_hash_of(at):
    from tests.test_frontend_unified_app import canvas_hash

    return canvas_hash(at)


# --- F3: true 1:1 view, navigator / F11: native shortcuts, non-colour overlays ---------------------------

def test_zoom_views_show_real_pixels():
    from app.professional_tools import zoom_preview

    gradient = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (100, 12))[:, :3000].reshape(100, 3000), mode="L").convert("RGB")
    fit = zoom_preview(gradient, 0.0)
    one_to_one = zoom_preview(gradient, 1.0, (0.0, 0.5))
    double = zoom_preview(gradient, 2.0, (0.0, 0.5))
    assert fit.width == 1200  # downscaled to the viewport
    assert one_to_one.size == (1200, 100)
    assert np.array_equal(np.asarray(one_to_one), np.asarray(gradient)[:, :1200])  # never resampled
    assert double.size == (1200, 200) and np.array_equal(np.asarray(double)[::2, ::2], np.asarray(gradient)[:, :600])


def test_shortcut_buttons_toggle_zoom_original_and_grid():
    at = run_with_uploaded_image(size=(40, 30))
    assert at.selectbox(key="cv-zoom").value == 0.0
    at.button(key="kb-zoom-btn").click()
    at.run(timeout=30)
    assert at.selectbox(key="cv-zoom").value == 1.0
    at.button(key="kb-orig-btn").click()
    at.run(timeout=30)
    assert at.checkbox(key="cv-orig").value is True
    at.button(key="kb-grid-btn").click()
    at.run(timeout=30)
    assert at.selectbox(key="grid-type").value == "Rule of Thirds"
    at.button(key="kb-grid-btn").click()
    at.run(timeout=30)
    assert at.selectbox(key="grid-type").value == "None"
    assert not at.exception
    source = APP_SOURCE = ui_source()
    for shortcut in ('shortcut="Ctrl+Z"', 'shortcut="Ctrl+Shift+Z"', 'shortcut="Ctrl+E"', 'shortcut="Z"', 'shortcut="B"', 'shortcut="G"'):
        assert shortcut in source
    assert "window.parent._photoStudioUndoRedoShortcutsBound" not in APP_SOURCE  # the injected JS is gone


def test_navigator_click_centers_the_zoomed_view():
    at = run_with_uploaded_image(size=(40, 30))
    at.selectbox(key="cv-zoom").set_value(2.0)
    at.run(timeout=30)
    at.text_area(key="cv-nav-wand-data").set_value(json.dumps({"x": 0.9, "y": 0.1}))
    at.run(timeout=30)
    assert not at.exception
    assert at.slider(key="cv-panx").value == 0.9 and at.slider(key="cv-pany").value == 0.1


def test_mask_overlay_styles_do_not_rely_on_colour_alone():
    from app.professional_tools import OVERLAY_STYLES, apply_mask_overlay

    base = Image.new("RGB", (60, 40), (100, 100, 100))
    mask = Image.new("L", (60, 40), 0)
    mask.paste(255, (10, 10, 50, 30))
    results = {style: np.asarray(apply_mask_overlay(base, mask, style=style)) for style in OVERLAY_STYLES}
    hatched, outline = results["Hatched"], results["Outline"]
    assert (hatched[15:25, 15:45] != 100).any() and (hatched[15:25, 15:45] == 100).all(axis=2).any()  # stripes
    assert (outline[20, 30] == 100).all() and (outline[10, 30] != 100).any()  # edge only


# --- F1: hidden tabs skip display-only work ----------------------------------------------------------

def test_hidden_tabs_skip_face_detection_and_editor_iframes(monkeypatch):
    import app.portrait_tools as portrait_module

    calls = []
    original = portrait_module.detect_faces
    monkeypatch.setattr(portrait_module, "detect_faces", lambda image: calls.append(1) or original(image))
    at = run_with_uploaded_image(size=(40, 30), color=(12, 34, 56))
    assert calls == [] and not any("faces" in caption.value for caption in at.caption)
    # The Portrait tab's widgets still exist (their state is kept), only the caption work is skipped.
    assert at.slider(key="port-sm").value == 0.0
    # Portrait is a sub-tab of Local (F9): Local open on another sub-tab still skips the work.
    open_tab(at, "◉ Local", "◉ Masks")
    at.run(timeout=30)
    assert calls == []
    open_tab(at, "◉ Local", "◌ Portrait")
    at.run(timeout=30)
    assert not at.exception
    assert len(calls) == 1 and any("Detected faces" in caption.value for caption in at.caption)


# --- F5: image layers keep their proportions and can be placed; sizes are relative -------------------

def test_image_layers_are_fitted_not_stretched_and_can_be_placed():
    from app.layer_workspace import render_image_layer

    logo = Image.new("RGBA", (20, 10), (255, 0, 0, 255))
    fitted = np.asarray(render_image_layer((100, 100), logo))
    opaque_rows = np.where(fitted[:, :, 3].max(axis=1) > 0)[0]
    assert len(opaque_rows) == 50  # 2:1 kept (the canvas is square) -- it used to fill 100x100
    moved = np.asarray(render_image_layer((100, 100), logo, x_pct=10, y_pct=90, scale_pct=20))
    ys, xs = np.where(moved[:, :, 3] > 0)
    assert abs(xs.mean() - 10) <= 1 and abs(ys.mean() - 90) <= 1 and np.ptp(xs) + 1 == 20
    rotated = np.asarray(render_image_layer((100, 100), logo, scale_pct=50, rotation_deg=90))
    ys, xs = np.where(rotated[:, :, 3] > 128)
    assert np.ptp(ys) > np.ptp(xs)  # now taller than wide


def test_an_added_image_layer_can_be_moved_on_the_canvas():
    from tests.test_frontend_unified_app import canvas_hash, image_bytes

    at = run_with_uploaded_image(size=(40, 30))
    at.get_by_key("l-new-file").set_value(("logo.png", image_bytes((255, 0, 0), (10, 10)), "image/png"))
    at.run(timeout=30)
    layer = at.session_state["image_layers"][0]
    assert layer["stretch"] is False and layer["scale_pct"] == 100.0
    before = canvas_hash(at)
    at.slider(key=f"l-img-x-{layer['id']}").set_value(20.0)
    at.run(timeout=30)
    assert not at.exception
    assert canvas_hash(at) != before


def test_text_size_is_relative_so_a_proxy_matches_the_export():
    from app.layer_workspace import render_text_layer

    full = render_text_layer((800, 600), "Hi", 0.10 * 600)
    proxy = render_text_layer((400, 300), "Hi", 0.10 * 300)
    full_box, proxy_box = full.getbbox(), proxy.getbbox()
    assert abs((full_box[2] - full_box[0]) / 2 - (proxy_box[2] - proxy_box[0])) <= 2


# --- F7: batch from memory, every format, streamed ZIP --------------------------------------------------

def test_batch_with_duplicate_names_keeps_both_files_and_writes_no_temp_copies(tmp_path):
    import zipfile

    from app.batch_release import FileLike, stream_to_zip
    from app.config import user_data_dir

    red = encode(Image.new("RGB", (32, 24), (200, 30, 30)), "JPEG")
    blue = encode(Image.new("RGB", (32, 24), (30, 30, 200)), "PNG")
    report = stream_to_zip([FileLike("photo.jpg", red), FileLike("photo.jpg", blue), FileLike("scan.png", blue)], tmp_path / "out.zip", profile="Web")
    assert [item.status for item in report] == ["ok", "ok", "ok"]
    with zipfile.ZipFile(tmp_path / "out.zip") as archive:
        names = archive.namelist()
        first, second = (np.asarray(Image.open(io.BytesIO(archive.read(name)))).mean(axis=(0, 1)) for name in ("photo-edited.jpg", "photo-edited-2.jpg"))
    assert "report.csv" in names
    assert first[0] > first[2] and second[2] > second[0]  # neither upload overwrote the other
    assert not (user_data_dir() / "batch-input").exists()


def test_batch_processor_accepts_raw_and_tiff_and_streams_a_zip(monkeypatch):
    from tests.test_frontend_unified_app import image_bytes

    at = run_with_uploaded_image()
    at.radio(key="more-mode").set_value("Batch Processor")
    at.run(timeout=30)
    for extension in ("tif", "dng", "raf", "webp"):
        assert extension in at.get_by_key("b-up").proto.type or f".{extension}" in at.get_by_key("b-up").proto.type
    at.get_by_key("b-up").set_value([("same.jpg", image_bytes(), "image/jpeg"), ("same.jpg", image_bytes((10, 200, 10)), "image/jpeg")])
    at.run(timeout=30)
    at.button(key="b-run-btn").click()
    at.run(timeout=30)
    assert not at.exception
    assert any("Done — 2/2 succeeded" in caption.value for caption in at.caption)
    zip_path = __import__("pathlib").Path(at.session_state["_batch_result"][0])
    assert zip_path.exists() and zip_path.parent.name == "batch-output"


# --- F8: Library <-> editor ------------------------------------------------------------------------------

def test_rescanning_a_folder_keeps_ratings(tmp_path):
    from app.library import index_folder
    from app.major_features import PhotoCatalog

    photo = tmp_path / "photos" / "a.jpg"
    photo.parent.mkdir()
    photo.write_bytes(encode(photo_like((64, 48)), "JPEG"))
    database = tmp_path / "catalog.sqlite"
    index_folder(photo.parent, database)
    PhotoCatalog(database).upsert(photo, rating=4, tags=["keeper"])
    index_folder(photo.parent, database)
    assert PhotoCatalog(database).get(photo) == {"path": str(photo), "rating": 4, "tags": ["keeper"]}


def test_library_thumbnails_are_cached_and_cover_raw(tmp_path):
    from app.library import index_folder, scan_folder, thumbnail

    folder = tmp_path / "shoot"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(encode(photo_like((640, 480)), "JPEG"))
    first = thumbnail(folder / "a.jpg", tmp_path / "thumbs")
    stamp = first.stat().st_mtime_ns
    assert thumbnail(folder / "a.jpg", tmp_path / "thumbs") == first and first.stat().st_mtime_ns == stamp
    assert max(Image.open(first).size) <= 320
    pytest.importorskip("rawpy")
    from tests.synthetic_assets import dng_bytes

    (folder / "b.dng").write_bytes(dng_bytes())
    assert [path.name for path in scan_folder(folder)] == ["a.jpg", "b.dng"]
    assert Image.open(thumbnail(folder / "b.dng", tmp_path / "thumbs")).size[0] > 0
    assert len(index_folder(folder, tmp_path / "catalog.sqlite")) == 2


def test_library_shows_the_catalog_without_a_rescan_and_edit_opens_the_photo(monkeypatch, tmp_path):
    from app.config import user_data_file
    from app.major_features import PhotoCatalog

    photo = tmp_path / "keeper.jpg"
    photo.write_bytes(encode(Image.new("RGB", (30, 20), (200, 10, 10)), "JPEG"))
    PhotoCatalog(user_data_file("catalog.sqlite")).upsert(photo, rating=5)

    at = run_with_uploaded_image(size=(40, 30))
    open_tab(at, "✦ Presets & Library", "☰ Library & Batch")
    at.radio(key="more-mode").set_value("Library Catalog")
    at.run(timeout=30)
    assert not at.exception
    assert any(image.captions and "keeper.jpg" in image.captions[0] for image in at.image)
    edit_buttons = [button for button in at.button if button.key and button.key.startswith("lib-edit-") and button.key != "lib-edit-current-btn"]
    open_tab(at, "✦ Presets & Library", "☰ Library & Batch")  # AppTest resends the old tab otherwise
    edit_buttons[0].click()
    at.run(timeout=30)
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state["restored_source_name"] == "keeper.jpg"
    assert canvas_dims(at) == (30, 20)


# --- P1: grain looks the same on the proxy and the export -----------------------------------------------

def test_film_grain_is_resolution_independent():
    from app.effects import apply_film_grain

    def grain_std(size):
        flat = Image.new("RGB", size, (128, 128, 128))
        return np.asarray(apply_film_grain(flat, 60), float).std()

    def downscaled_grain_std(size, target):
        flat = Image.new("RGB", size, (128, 128, 128))
        return np.asarray(apply_film_grain(flat, 60).resize(target, Image.Resampling.LANCZOS), float).std()

    proxy = grain_std((2048, 1365))
    export_seen_at_proxy_size = downscaled_grain_std((5400, 3600), (2048, 1365))
    assert 0.6 < export_seen_at_proxy_size / proxy < 1.6  # per-pixel noise made this ratio ~0.4


# --- F6: file name template and saveable export presets --------------------------------------------------

def test_export_presets_save_and_apply_and_the_file_name_follows_the_template(monkeypatch):
    import streamlit as st

    names = []
    original = st.download_button

    def capture(label, data, *args, **kwargs):
        if kwargs.get("key") == "exp-download-btn":
            names.append(kwargs.get("file_name"))
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture)
    at = run_with_uploaded_image(size=(40, 30))
    at.text_input(key="exp-name-template").set_value("{stem}_{width}x{height}")
    at.selectbox(key="exp-fmt").set_value("PNG")
    at.run(timeout=30)
    assert names[-1] == "photo_40x30.png"
    at.text_input(key="exp-preset-name").set_value("Web PNG")
    at.button(key="exp-preset-save-btn").click()
    at.run(timeout=30)
    at.run(timeout=30)

    at.selectbox(key="exp-fmt").set_value("JPEG")
    at.text_input(key="exp-name-template").set_value("{stem}")
    at.run(timeout=30)
    at.button(key="exp-preset-apply-btn").click()
    at.run(timeout=30)
    assert not at.exception
    assert at.selectbox(key="exp-fmt").value == "PNG"
    assert at.text_input(key="exp-name-template").value == "{stem}_{width}x{height}"
