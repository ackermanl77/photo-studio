"""HDR merge and panorama stitching (F18)."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.merge import (
    MergeError,
    SourcePhoto,
    _crop_to_content,
    encode_16bit_tiff,
    load_source,
    merge_hdr,
    stitch_panorama,
)


def _scene(width=480, height=320, seed=3) -> np.ndarray:
    """A textured "radiance" with a bright window that clips in a normal exposure."""
    import cv2

    rng = np.random.default_rng(seed)
    img = (cv2.GaussianBlur(rng.random((height, width, 3)).astype(np.float32), (0, 0), 2) * 255).astype(np.uint8)
    for _ in range(int(width * height / 400)):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (x, y), int(rng.integers(3, 18)), colour, -1)
    radiance = img.astype(np.float32) / 255.0
    radiance[40:140, 280:420] *= 6.0
    return radiance


def _bracket(radiance, shift=(0, 0)):
    shots = []
    for ev, seconds in ((-2, 1 / 400), (0, 1 / 100), (2, 1 / 25)):
        shot = np.clip(radiance * (2.0 ** ev) ** (1 / 2.2), 0, 1).astype(np.float32)
        shots.append(SourcePhoto(f"ev{ev}", shot, seconds))
    if shift != (0, 0):
        shots[2] = SourcePhoto("ev2", np.roll(shots[2].pixels, shift, axis=(0, 1)), 1 / 25)
    return shots


def _clipped(pixels) -> float:
    return float((pixels >= 0.999).mean())


def test_exposure_fusion_recovers_the_clipped_window():
    shots = _bracket(_scene())
    fused = merge_hdr(shots, "fusion")
    assert fused.shape == shots[1].pixels.shape and fused.dtype == np.float32
    assert _clipped(shots[1].pixels) > 0.05 and _clipped(fused) < 0.01
    window = fused[40:140, 280:420]
    assert window.std() > 0.05  # detail, not a flat white block
    assert 0.0 <= fused.min() and fused.max() <= 1.0


def test_alignment_undoes_a_hand_held_shift():
    radiance = _scene()
    aligned = merge_hdr(_bracket(radiance, shift=(3, -4)), "fusion", align=True)
    reference = merge_hdr(_bracket(radiance), "fusion")
    unaligned = merge_hdr(_bracket(radiance, shift=(3, -4)), "fusion", align=False)
    inner = (slice(10, -10), slice(10, -10))
    assert np.abs(aligned[inner] - reference[inner]).mean() < 0.5 * np.abs(unaligned[inner] - reference[inner]).mean()


def test_the_hdr_merge_needs_different_exposure_times():
    shots = _bracket(_scene())
    merged = merge_hdr(shots, "debevec")
    assert _clipped(merged) < 0.01 and merged.mean() > 0.15
    with pytest.raises(MergeError, match="exposure time"):
        merge_hdr([SourcePhoto(s.name, s.pixels, None) for s in shots], "debevec")
    with pytest.raises(MergeError, match="same exposure time"):
        merge_hdr([SourcePhoto(s.name, s.pixels, 1 / 100) for s in shots], "debevec")
    with pytest.raises(MergeError):
        merge_hdr(shots[:1])
    with pytest.raises(MergeError):
        merge_hdr(shots, "magic")


@pytest.mark.parametrize("scans", [True, False])
def test_overlapping_frames_stitch_back_into_the_strip(scans):
    strip = _scene(1500, 420, seed=7)
    strip[40:140, 280:420] /= 6.0  # no clipped block here
    frames = [SourcePhoto(f"p{i}", strip[:, x:x + 700].copy(), None) for i, x in enumerate((0, 400, 800))]
    pano = stitch_panorama(frames, scans=scans)
    height, width = pano.shape[:2]
    assert 1300 <= width <= 1560 and 360 <= height <= 440
    assert (pano.reshape(-1, 3).max(axis=1) <= 0.0).mean() < 0.01  # cropped: no empty border left


def test_frames_that_dont_overlap_are_refused():
    a, b = _scene(400, 300, seed=1), _scene(400, 300, seed=2)
    with pytest.raises(MergeError, match="couldn't be stitched"):
        stitch_panorama([SourcePhoto("a", a, None), SourcePhoto("b", b, None)])


def test_crop_to_content_finds_the_inner_rectangle():
    pixels = np.ones((100, 200, 3), np.float32)
    mask = np.zeros((100, 200), np.uint8)
    mask[10:90, 20:180] = 1
    mask[10:20, 20:40] = 0  # a missing corner
    cropped = _crop_to_content(pixels, mask)
    assert cropped.shape[0] * cropped.shape[1] >= 0.8 * 80 * 160


def test_the_result_is_a_16_bit_source_and_exposure_times_are_read():
    from app.high_bit import decode_high_bit

    gradient = np.tile(np.linspace(0, 1, 512, dtype=np.float32), (8, 1))[..., None].repeat(3, axis=2)
    tiff = encode_16bit_tiff(gradient)
    decoded = decode_high_bit(tiff, "merged.tif")
    assert decoded is not None and np.abs(decoded - gradient).max() < 1 / 30000

    image = Image.new("RGB", (32, 24), (90, 90, 90))
    exif = Image.Exif()
    exif.get_ifd(0x8769)[33434] = 0.004  # 1/250 s
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    photo = load_source(buffer.getvalue(), "bracket.jpg")
    assert photo.exposure_time == pytest.approx(0.004)
    assert photo.pixels.shape == (24, 32, 3) and photo.pixels.dtype == np.float32


@pytest.mark.slow
def test_the_merge_tool_opens_the_result_in_the_editor(monkeypatch):
    from tests.test_frontend_unified_app import _fake_process_via_backend, run_with_uploaded_image

    from app import backend_client

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    at = run_with_uploaded_image()
    at.radio(key="more-mode").set_value("HDR & Panorama")
    at.run(timeout=30)
    files = []
    for shot in _bracket(_scene(240, 160)):
        buffer = BytesIO()
        Image.fromarray((shot.pixels * 255).astype(np.uint8)).save(buffer, format="PNG")
        files.append((f"{shot.name}.png", buffer.getvalue(), "image/png"))
    at.get_by_key("merge-up").set_value(files)
    at.run(timeout=30)
    assert not at.exception
    at.button(key="merge-run").click()
    at.run(timeout=60)
    assert not at.exception
    name, tiff, _preview, size = at.session_state["_merge_result"]
    assert name == "ev-2-hdr.tif" and size == "240 × 160"
    at.button(key="merge-open").click()
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["restored_source_name"] == "ev-2-hdr.tif"
