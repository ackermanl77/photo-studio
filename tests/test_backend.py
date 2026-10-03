import io
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.backend import app
from app.config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from app.image_processor import PhotoProcessor
from app.layer_workspace import WorkspaceLayer, layers_to_json
from tests.legacy_project import PhotoProject
from app.raw_import import is_raw_path


def image_bytes(color=(100, 110, 120), size=(12, 12), format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", size, color=color).save(stream, format=format)
    return stream.getvalue()


def layer_stack_bytes(*layers: tuple[tuple[int, int, int], float, str]) -> bytes:
    """Build a /process-ready `layers` JSON payload from (color, opacity, blend_mode) tuples,
    the same wire format app/frontend_unified_app.py's Layers tab now sends for its whole stack."""
    workspace_layers = [
        WorkspaceLayer(f"Layer {index}", Image.new("RGBA", (20, 20), (*color, 255)), opacity, blend_mode)
        for index, (color, opacity, blend_mode) in enumerate(layers)
    ]
    return layers_to_json(workspace_layers).encode("utf-8")


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_backend_security_defaults():
    assert app.config["MAX_CONTENT_LENGTH"] == MAX_UPLOAD_BYTES
    assert MAX_IMAGE_PIXELS > 0
    assert app.config["MAX_CONTENT_LENGTH"] > 0


def test_raw_extensions_are_recognized_without_changing_normal_images():
    assert is_raw_path("camera.CR3")
    assert is_raw_path("camera.nef")
    assert not is_raw_path("photo.jpg")


def test_process_endpoint_returns_requested_jpeg():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes()), "input.png"),
            "options": json.dumps({"output_format": "JPEG", "quality": 85, "exposure": 1.1}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert Image.open(io.BytesIO(response.data)).format == "JPEG"


def test_process_endpoint_applies_the_hsl_color_mixer():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(color=(220, 40, 40), size=(12, 12))), "input.png"),
            "options": json.dumps({"hsl_bands": {"Red": {"hue": 0, "sat": -100, "lum": 0}}}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)), dtype=np.int32)
    # A full desaturation of the Red band on a solid red image should land close to gray (G ~= B).
    assert abs(int(result[0, 0, 1]) - int(result[0, 0, 2])) < 5


def test_process_endpoint_applies_lens_corrections():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(color=(200, 200, 200), size=(40, 40))), "input.png"),
            "options": json.dumps({"lens_vignette_amount": -80, "lens_vignette_midpoint": 30}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)), dtype=np.int32)
    assert result[:4, :4].mean() < result[18:22, 18:22].mean() - 10  # corners darkened


def test_process_endpoint_applies_color_noise_denoise():
    client = app.test_client()
    stream = io.BytesIO()
    rng = np.random.default_rng(0)
    noisy = np.clip(np.full((40, 40, 3), 128, dtype=np.float32) + rng.normal(0, 20, (40, 40, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(noisy, "RGB").save(stream, format="PNG")

    same_strength = client.post(
        "/process",
        data={"image": (io.BytesIO(stream.getvalue()), "noisy.png"), "options": json.dumps({"denoise": 4})},
        content_type="multipart/form-data",
    )
    color_boosted = client.post(
        "/process",
        data={"image": (io.BytesIO(stream.getvalue()), "noisy.png"), "options": json.dumps({"denoise": 4, "denoise_color": 9})},
        content_type="multipart/form-data",
    )

    assert same_strength.status_code == 200
    assert color_boosted.status_code == 200
    assert same_strength.data != color_boosted.data


def test_process_endpoint_applies_post_crop_vignette():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(color=(200, 200, 200), size=(40, 40))), "input.png"),
            "options": json.dumps({"effects_vignette_amount": -80, "effects_vignette_midpoint": 30}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)), dtype=np.int32)
    assert result[:4, :4].mean() < result[18:22, 18:22].mean() - 10  # corners darkened


def test_process_endpoint_applies_film_grain():
    client = app.test_client()
    baseline = client.post(
        "/process",
        data={"image": (io.BytesIO(image_bytes(color=(120, 120, 120), size=(30, 30))), "input.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    grainy = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(color=(120, 120, 120), size=(30, 30))), "input.png"),
            "options": json.dumps({"effects_grain_amount": 80}),
        },
        content_type="multipart/form-data",
    )

    assert baseline.status_code == 200
    assert grainy.status_code == 200
    assert baseline.data != grainy.data


def test_process_endpoint_applies_ai_upscale_last_when_enabled(monkeypatch):
    """AI Upscale is the one deliberate exception to this app's CPU-first design (see "Known
    Issues" in docs/PROJECT_STATUS.md) -- confirms process_image() calls
    PhotoProcessor.apply_ai_upscale with the requested factor when options["ai_upscale"] is set,
    and that it runs after Effects (the already-vignetted/grained image is what gets "upscaled").
    Mocks the actual Real-ESRGAN inference (apply_ai_upscale itself) so this doesn't need the
    optional torch/realesrgan packages or a real GPU -- app.ai_upscale.ai_upscale_available is
    also mocked True so the gate in process_image() doesn't short-circuit before reaching it."""
    calls = []

    def fake_apply_ai_upscale(self, factor=2):
        calls.append(factor)
        self.image = self.image.resize((self.image.width * factor, self.image.height * factor))
        return self

    monkeypatch.setattr(PhotoProcessor, "apply_ai_upscale", fake_apply_ai_upscale)
    monkeypatch.setattr("app.ai_upscale.ai_upscale_available", lambda: True)

    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(size=(10, 10))), "input.png"),
            "options": json.dumps({"ai_upscale": True, "ai_upscale_factor": 4}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert calls == [4]
    result = Image.open(io.BytesIO(response.data))
    assert result.size == (40, 40)


def test_process_endpoint_skips_ai_upscale_when_not_requested(monkeypatch):
    """Off by default -- an expensive, optional operation -- so a plain options dict (or
    ai_upscale explicitly False) must never call apply_ai_upscale at all."""

    def fail_if_called(self, factor=2):
        raise AssertionError("apply_ai_upscale should not run when the ai_upscale option is falsy")

    monkeypatch.setattr(PhotoProcessor, "apply_ai_upscale", fail_if_called)

    client = app.test_client()
    response = client.post(
        "/process",
        data={"image": (io.BytesIO(image_bytes()), "input.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200


def test_process_endpoint_returns_a_clean_error_when_ai_upscale_requested_but_unavailable(monkeypatch):
    """On a machine without requirements-gpu.txt installed, requesting ai_upscale must fail with
    the same clean JSON 400 error every other bad-input case gets (RuntimeError is already in the
    /process route's except tuple), not an ImportError/AttributeError deep inside PhotoProcessor."""
    monkeypatch.setattr("app.ai_upscale.ai_upscale_available", lambda: False)

    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes()), "input.png"),
            "options": json.dumps({"ai_upscale": True}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.is_json
    assert "requirements-gpu.txt" in response.get_json()["error"]


def test_process_endpoint_stop_after_detail_skips_repair_onward_steps():
    """options["stop_after"] == "detail" should return right after the Detail section, before
    Repair/Layers/Masks/Portrait/Crop/Effects -- confirmed by requesting a crop that would
    otherwise visibly change the output dimensions."""
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(size=(40, 40))), "input.png"),
            "options": json.dumps({"stop_after": "detail", "aspect": "1:1", "crop_box": [0, 0, 10, 10]}),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    result = Image.open(io.BytesIO(response.data))
    assert result.size == (40, 40)  # crop never applied -- stopped before Crop


def test_process_endpoint_resume_from_detail_skips_lens_tone_color_detail():
    """options["resume_from_detail"] should treat the uploaded image as already Lens/Tone/Color/
    Detail-processed and skip reapplying any of that -- confirmed by sending a large Exposure
    value that would visibly blow out the image if it were actually reapplied."""
    client = app.test_client()
    stage_a = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(color=(100, 100, 100), size=(20, 20))), "input.png"),
            "options": json.dumps({"stop_after": "detail"}),
        },
        content_type="multipart/form-data",
    )
    assert stage_a.status_code == 200

    resumed = client.post(
        "/process",
        data={
            "image": (io.BytesIO(stage_a.data), "stage_a.png"),
            "options": json.dumps({"resume_from_detail": True, "exposure": 5.0}),
        },
        content_type="multipart/form-data",
    )
    assert resumed.status_code == 200
    assert Image.open(io.BytesIO(resumed.data)).getpixel((10, 10)) == (100, 100, 100)


def test_process_endpoint_two_stage_split_matches_single_pass_output():
    """The whole point of the split (see docs/PROJECT_STATUS.md's Known Issues): chaining
    stop_after="detail" then resume_from_detail with the remaining options must produce the same
    output as a single unsplit call with the combined options -- otherwise the cache split
    app/frontend_unified_app.py performs would be a genuine behavior change, not just a
    performance one."""
    client = app.test_client()
    base = image_bytes(color=(90, 140, 200), size=(30, 30))
    detail_options = {"exposure": 1.2, "saturation": 1.1, "texture": 20}
    repair_onward_options = {"aspect": "1:1"}

    single_pass = client.post(
        "/process",
        data={"image": (io.BytesIO(base), "input.png"), "options": json.dumps({**detail_options, **repair_onward_options})},
        content_type="multipart/form-data",
    )
    assert single_pass.status_code == 200

    stage_a = client.post(
        "/process",
        data={"image": (io.BytesIO(base), "input.png"), "options": json.dumps({**detail_options, "stop_after": "detail"})},
        content_type="multipart/form-data",
    )
    assert stage_a.status_code == 200
    stage_b = client.post(
        "/process",
        data={"image": (io.BytesIO(stage_a.data), "stage_a.png"), "options": json.dumps({**repair_onward_options, "resume_from_detail": True})},
        content_type="multipart/form-data",
    )
    assert stage_b.status_code == 200

    single_pixels = np.array(Image.open(io.BytesIO(single_pass.data)))
    split_pixels = np.array(Image.open(io.BytesIO(stage_b.data)))
    assert single_pixels.shape == split_pixels.shape
    assert np.array_equal(single_pixels, split_pixels)  # PNG is lossless, so this should be exact


def test_process_endpoint_resume_from_detail_still_preserves_icc_profile():
    """The "detail" stop_after branch explicitly re-embeds icc_profile before returning (it isn't
    covered by the shared re-tagging at the very end of process_image(), since resume_from_detail
    skips straight past that source's own ICC-normalization step) -- confirms it actually survives
    both hops of a real split (source -> stage A -> stage B), not just the single-pass path
    test_process_endpoint_normalizes_a_real_embedded_profile_to_srgb already covers."""
    from app.professional_color import srgb_icc_bytes

    profile = srgb_icc_bytes()
    image = Image.new("RGB", (10, 10), color=(180, 90, 40))
    source = io.BytesIO()
    image.save(source, format="PNG", icc_profile=profile)
    source.seek(0)
    client = app.test_client()

    stage_a = client.post(
        "/process",
        data={"image": (source, "profiled.png"), "options": json.dumps({"stop_after": "detail"})},
        content_type="multipart/form-data",
    )
    assert stage_a.status_code == 200
    assert Image.open(io.BytesIO(stage_a.data)).info.get("icc_profile")

    resumed = client.post(
        "/process",
        data={"image": (io.BytesIO(stage_a.data), "stage_a.png"), "options": json.dumps({"resume_from_detail": True})},
        content_type="multipart/form-data",
    )
    assert resumed.status_code == 200
    output_profile = Image.open(io.BytesIO(resumed.data)).info.get("icc_profile")
    assert output_profile  # survived the resume_from_detail hop too


def test_process_endpoint_applies_vibrance():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(color=(180, 150, 150), size=(12, 12))), "input.png"),
            "options": json.dumps({"vibrance": 90}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    baseline = client.post(
        "/process",
        data={"image": (io.BytesIO(image_bytes(color=(180, 150, 150), size=(12, 12))), "input.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    assert baseline.data != response.data


def test_process_endpoint_applies_lens_perspective():
    client = app.test_client()
    grid = Image.new("RGB", (40, 40), (30, 30, 30))
    ImageDraw.Draw(grid).line([(20, 0), (20, 40)], fill=(220, 220, 220), width=2)
    stream = io.BytesIO()
    grid.save(stream, format="PNG")

    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(stream.getvalue()), "grid.png"),
            "options": json.dumps({"lens_perspective_v": 60}),
        },
        content_type="multipart/form-data",
    )
    baseline = client.post(
        "/process",
        data={"image": (io.BytesIO(stream.getvalue()), "grid.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert baseline.data != response.data


def test_process_endpoint_applies_color_grading():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(color=(20, 20, 20), size=(20, 20))), "input.png"),
            "options": json.dumps({"cg_shadows_hue": 220, "cg_shadows_intensity": 100}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)), dtype=np.int32)
    pixel = result[10, 10]
    assert pixel[2] > pixel[0]  # tinted toward blue (hue ~220 degrees)


def test_process_endpoint_applies_dehaze():
    client = app.test_client()
    hazy = np.full((40, 40, 3), 190, dtype=np.uint8)
    hazy[10:30, 10:30] = 90  # a darker patch buried in uniform haze
    stream = io.BytesIO()
    Image.fromarray(hazy, "RGB").save(stream, format="PNG")

    baseline = client.post(
        "/process",
        data={"image": (io.BytesIO(stream.getvalue()), "hazy.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    dehazed = client.post(
        "/process",
        data={"image": (io.BytesIO(stream.getvalue()), "hazy.png"), "options": json.dumps({"dehaze": 90})},
        content_type="multipart/form-data",
    )

    assert baseline.status_code == 200
    assert dehazed.status_code == 200
    assert baseline.data != dehazed.data


def test_process_endpoint_preserves_icc_profile():
    """Malformed/placeholder bytes under "icc_profile" (this fake marker isn't a real ICC
    profile) -- normalize_to_srgb() can't open it, so process_image() falls back to the old
    preserve-as-is behavior instead of failing the whole edit over an unparseable color tag."""
    image = Image.new("RGB", (12, 12), color=(100, 110, 120))
    source = io.BytesIO()
    profile = b"test-icc-profile"
    image.save(source, format="JPEG", icc_profile=profile)
    source.seek(0)
    client = app.test_client()

    response = client.post(
        "/process",
        data={
            "image": (source, "profiled.jpg"),
            "options": json.dumps({"output_format": "JPEG"}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.data)).info.get("icc_profile") == profile


def test_process_endpoint_normalizes_a_real_embedded_profile_to_srgb():
    """A genuinely valid embedded ICC profile (unlike the fake marker bytes above) should be
    converted through normalize_to_srgb() and the output re-tagged with a real sRGB profile
    rather than the original bytes -- confirms the actual color-management code path runs
    end-to-end through /process, not just that invalid profiles degrade gracefully."""
    from app.professional_color import srgb_icc_bytes

    profile = srgb_icc_bytes()
    image = Image.new("RGB", (12, 12), color=(180, 90, 40))
    source = io.BytesIO()
    image.save(source, format="PNG", icc_profile=profile)
    source.seek(0)
    client = app.test_client()

    response = client.post(
        "/process",
        data={"image": (source, "profiled.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = Image.open(io.BytesIO(response.data))
    from PIL import ImageCms

    output_profile = result.info.get("icc_profile")
    assert output_profile  # re-tagged with a real profile (not silently dropped)
    assert ImageCms.getProfileName(ImageCms.ImageCmsProfile(io.BytesIO(output_profile))).strip()  # a well-formed ICC profile, not garbage
    pixel = np.array(result, dtype=np.int16)[0, 0]
    assert (np.abs(pixel - np.array([180, 90, 40])) <= 2).all()  # near-identity for an already-sRGB source


def test_process_endpoint_rejects_unknown_preset():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes()), "input.png"),
            "options": json.dumps({"preset": "missing"}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Unknown preset" in response.get_json()["error"]


def test_process_endpoint_composites_layer():
    client = app.test_client()
    base = image_bytes((40, 40, 40), size=(20, 20))

    baseline = client.post(
        "/process",
        data={"image": (io.BytesIO(base), "input.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    with_layer = client.post(
        "/process",
        data={
            "image": (io.BytesIO(base), "input.png"),
            "layers": (io.BytesIO(layer_stack_bytes(((220, 30, 30), 1.0, "normal"))), "layers.json"),
            "options": json.dumps({}),
        },
        content_type="multipart/form-data",
    )

    assert baseline.status_code == 200
    assert with_layer.status_code == 200
    assert baseline.data != with_layer.data


def test_process_endpoint_composites_multiple_layers_in_order():
    """Two stacked layers must both reach the final image, applied bottom-to-top in the same
    order they were serialized -- the multi-layer replacement for the old single-`layer_file`
    API, which could only ever composite exactly one overlay."""
    client = app.test_client()
    base = image_bytes((0, 0, 0), size=(20, 20))
    # A half-opacity red layer, then a fully-opaque blue layer on top -- the blue layer should
    # completely hide the red one, since it's later in the stack and fully opaque.
    payload = layer_stack_bytes(((255, 0, 0), 0.5, "normal"), ((0, 0, 255), 1.0, "normal"))

    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(base), "input.png"),
            "layers": (io.BytesIO(payload), "layers.json"),
            "options": json.dumps({}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)).convert("RGB"))
    assert tuple(int(v) for v in result[10, 10]) == (0, 0, 255)


def test_process_endpoint_applies_a_non_default_blend_mode_from_the_layer_payload():
    client = app.test_client()
    base = image_bytes((100, 100, 100), size=(20, 20))
    multiply_payload = layer_stack_bytes(((200, 200, 200), 1.0, "multiply"))

    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(base), "input.png"),
            "layers": (io.BytesIO(multiply_payload), "layers.json"),
            "options": json.dumps({}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)).convert("RGB"))
    # multiply(100/255, 200/255) * 255 ~= 78
    assert 65 <= int(result[10, 10, 0]) <= 90


def test_process_endpoint_masks_apply_after_layer_composite():
    """The combined mask must see the image AFTER the layer is composited (matching the
    unified UI's Layers -> Masks tab order) -- otherwise a masked adjustment could miss pixels
    the layer just changed. This is the ordering bug class documented under "Recently Fixed"
    in docs/PROJECT_STATUS.md, reproduced here for the backend's own pipeline."""
    base = image_bytes((40, 40, 40), size=(20, 20))
    mask_buffer = io.BytesIO()
    Image.new("L", (20, 20), 255).save(mask_buffer, format="PNG")

    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(base), "input.png"),
            "layers": (io.BytesIO(layer_stack_bytes(((250, 250, 250), 1.0, "normal"))), "layers.json"),
            "local_mask": (io.BytesIO(mask_buffer.getvalue()), "mask.png"),
            "options": json.dumps({"mask_exposure": 0.5}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)))
    # Halving exposure on the post-layer (white, 250) image lands at ~125; if masking had seen
    # the pre-layer (dark, 40) image instead, it would land at ~20.
    assert abs(result.mean() - 125.0) < 2.0


def test_process_endpoint_applies_portrait_smoothing():
    rng = np.random.default_rng(0)
    noisy = np.clip(np.array([180, 120, 90], dtype=np.float32) + rng.uniform(-25, 25, size=(24, 24, 3)), 0, 255).astype("uint8")
    stream = io.BytesIO()
    Image.fromarray(noisy, "RGB").save(stream, format="PNG")
    payload = stream.getvalue()

    client = app.test_client()
    unsmoothed = client.post(
        "/process",
        data={"image": (io.BytesIO(payload), "face.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    smoothed = client.post(
        "/process",
        data={"image": (io.BytesIO(payload), "face.png"), "options": json.dumps({"portrait_smoothing": 0.8})},
        content_type="multipart/form-data",
    )

    assert unsmoothed.status_code == 200
    assert smoothed.status_code == 200
    unsmoothed_arr = np.array(Image.open(io.BytesIO(unsmoothed.data))).astype(int)
    smoothed_arr = np.array(Image.open(io.BytesIO(smoothed.data))).astype(int)
    assert np.abs(unsmoothed_arr - smoothed_arr).mean() > 1.0


def test_process_endpoint_skips_pil_validation_for_raw_uploads(monkeypatch):
    """validate_image_bytes() uses PIL, which can't decode camera RAW -- for a RAW filename,
    process_image() must skip straight to PhotoProcessor.from_bytes() (stubbed here) instead of
    rejecting the raw camera bytes as "Invalid image content"."""

    def fake_from_bytes(cls, content, filename):
        processor = cls.__new__(cls)
        processor.input_path = None
        processor.image = Image.new("RGB", (8, 8), (50, 60, 70))
        return processor

    monkeypatch.setattr(PhotoProcessor, "from_bytes", classmethod(fake_from_bytes))

    client = app.test_client()
    response = client.post(
        "/process",
        data={"image": (io.BytesIO(b"not a real raw file"), "photo.cr2"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200


def test_process_endpoint_removes_the_selected_region():
    """PhotoProcessor.remove_objects() (cv2.inpaint) via the removal_mask form field -- this
    endpoint plumbing existed before the unified UI ever sent it (see docs/PROJECT_STATUS.md's
    Repair tab entry); this is the first test exercising it through /process."""
    size = (40, 40)
    object_box = (15, 15, 25, 25)
    base = Image.new("RGB", size, (30, 30, 30))
    ImageDraw.Draw(base).rectangle(object_box, fill=(220, 20, 20))
    base_buffer = io.BytesIO()
    base.save(base_buffer, format="PNG")

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(object_box, fill=255)
    mask_buffer = io.BytesIO()
    mask.save(mask_buffer, format="PNG")

    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(base_buffer.getvalue()), "input.png"),
            "removal_mask": (io.BytesIO(mask_buffer.getvalue()), "mask.png"),
            "options": json.dumps({}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = np.array(Image.open(io.BytesIO(response.data)))
    # Inpainting blends the removed region in from the surrounding dark background -- the red
    # object (220, 20, 20) should be gone, replaced by something close to the background (30).
    removed_region_mean = result[15:25, 15:25].mean()
    assert abs(removed_region_mean - 30) < 10


def test_process_endpoint_applies_a_free_form_crop_box():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(size=(40, 30))), "input.png"),
            "options": json.dumps({"crop_box": [10, 5, 30, 25]}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = Image.open(io.BytesIO(response.data))
    assert result.size == (20, 20)


def test_process_endpoint_crop_box_takes_priority_over_aspect():
    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes(size=(40, 30))), "input.png"),
            "options": json.dumps({"crop_box": [0, 0, 12, 8], "aspect": "1:1"}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = Image.open(io.BytesIO(response.data))
    # A 1:1 aspect fit on a 40x30 source would land at 30x30 -- confirms crop_box won, not aspect.
    assert result.size == (12, 8)


def test_process_endpoint_returns_clean_json_error_for_a_cv2_failure(monkeypatch):
    """apply_denoise() now calls cv2.fastNlMeansDenoisingColored (a real cv2.error source, e.g.
    for a degenerate input) instead of the old flat median filter -- regression test for a real
    gap found in review: cv2.error wasn't in the /process route's except tuple, so it used to
    fall through to Flask's generic HTML 500 instead of the same clean JSON error every other
    failure mode gets. Raises cv2.error directly (rather than trying to construct an image that
    actually triggers OpenCV's internal assertion) to test the route's error handling in
    isolation from OpenCV's own input-validation behavior."""

    def fake_apply_denoise(self, radius=3, color_radius=0):
        raise cv2.error("synthetic failure for this test")

    monkeypatch.setattr(PhotoProcessor, "apply_denoise", fake_apply_denoise)

    client = app.test_client()
    response = client.post(
        "/process",
        data={
            "image": (io.BytesIO(image_bytes()), "input.png"),
            "options": json.dumps({"denoise": 3}),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.is_json
    assert "synthetic failure for this test" in response.get_json()["error"]


def test_process_endpoint_still_validates_non_raw_uploads():
    client = app.test_client()
    response = client.post(
        "/process",
        data={"image": (io.BytesIO(b"not an image"), "photo.jpg"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "Invalid image content" in response.get_json()["error"]


def test_photo_project_round_trip(tmp_path):
    project_path = tmp_path / "session.photoproject"
    original = PhotoProject("input.png", image_bytes(), {"exposure": 1.2}, b"remove", b"local")

    original.save(project_path)
    restored = PhotoProject.load(project_path)

    assert restored.source_name == "input.png"
    assert restored.source_bytes == original.source_bytes
    assert restored.recipe == {"exposure": 1.2}
    assert restored.removal_mask == b"remove"
    assert restored.local_mask == b"local"
