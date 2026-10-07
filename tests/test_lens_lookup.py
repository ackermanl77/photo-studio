"""F13: Lens -> "From the photo's lens" (app.lens_lookup, the optional lensfunpy database)."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app import lens_lookup
from tests.synthetic_assets import photo_like

lensfunpy = pytest.importorskip("lensfunpy")


def _photo_with_exif(make="Canon", model="Canon EOS 5D Mark III", lens="EF24-105mm f/4L IS USM", focal=24.0, aperture=4.0, size=(600, 400)) -> Image.Image:
    exif = Image.Exif()
    if make:
        exif[0x010F], exif[0x0110] = make, model
    sub = exif.get_ifd(0x8769)
    if lens:
        sub[0xA434] = lens
    if focal:
        sub[0x920A] = focal
    if aperture:
        sub[0x829D] = aperture
    buffer = io.BytesIO()
    photo_like(size).save(buffer, "JPEG", exif=exif)
    return Image.open(io.BytesIO(buffer.getvalue()))


def test_a_known_zoom_is_matched_and_its_sliders_follow_the_focal_length():
    wide, message = lens_lookup.find_lens(_photo_with_exif(focal=24.0))
    long, _ = lens_lookup.find_lens(_photo_with_exif(focal=105.0, aperture=8.0))
    assert wide is not None and wide.lens == "Canon EF 24-105mm f/4L IS USM" and "24 mm f/4" in message
    assert wide.sliders["lens_distortion"] < 0 < long.sliders["lens_distortion"]  # barrel wide, pincushion long
    assert wide.sliders["lens_vignette_amount"] > long.sliders["lens_vignette_amount"] > 0  # wide open vignettes more
    assert set(wide.sliders) == {"lens_distortion", "lens_vignette_amount", "lens_vignette_midpoint", "lens_chromatic_aberration"}


def test_the_fitted_distortion_approximates_lensfuns_own_correction():
    database = lensfunpy.Database()
    camera = database.find_cameras("Canon", "Canon EOS 5D Mark III")[0]
    lens = database.find_lenses(camera, None, "EF24-105mm f/4L IS USM")[0]
    width, height = 600, 400
    k1 = lens_lookup.fit_sliders(lens, camera.crop_factor, (width, height), 24.0, 4.0)["lens_distortion"] / 100.0 * 0.5
    modifier = lensfunpy.Modifier(lens, camera.crop_factor, width, height)
    modifier.initialize(24.0, 4.0, 1000.0, scale=1.0)
    coords = modifier.apply_geometry_distortion()
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    reference = np.stack([coords[..., 0] - xx, coords[..., 1] - yy], axis=-1)
    vx, vy = xx - (width - 1) / 2, yy - (height - 1) / 2
    r2 = (vx ** 2 + vy ** 2) / max(width, height) ** 2
    ours = np.stack([vx * k1 * r2, vy * k1 * r2], axis=-1)
    residual = np.sqrt(((reference - ours) ** 2).mean())
    assert residual < 0.6 * np.sqrt((reference ** 2).mean())  # most of the field explained by one term


@pytest.mark.parametrize("kwargs, reason", [
    ({"focal": None}, "no focal length"),
    ({"lens": "Totally Unknown 50mm Lens"}, "no calibration"),
    ({"make": None, "lens": None}, "no calibration"),
])
def test_unmatched_photos_say_why(kwargs, reason):
    match, message = lens_lookup.find_lens(_photo_with_exif(**kwargs))
    assert match is None and reason in message


def test_without_lensfunpy_the_lookup_explains_how_to_install_it(monkeypatch):
    monkeypatch.setattr(lens_lookup, "lensfun_available", lambda: False)
    match, message = lens_lookup.find_lens(_photo_with_exif())
    assert match is None and "requirements-raw.txt" in message


def test_the_lens_button_fills_the_sliders_through_the_ui(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import backend_client
    from tests.test_frontend_unified_app import APP_PATH, _fake_process_via_backend

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)

    buffer = io.BytesIO()
    _photo_with_exif(size=(60, 40)).save(buffer, "JPEG", exif=_photo_with_exif().getexif())
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("photo.jpg", buffer.getvalue(), "image/jpeg"))
    at.run(timeout=30)
    at.button(key="lens-auto-btn").click()
    at.run(timeout=30)
    assert not at.exception
    assert at.slider(key="lens-distortion").value < 0 and at.slider(key="lens-vignette-amt").value > 0
    assert any("Matched Canon EF 24-105mm" in caption.value for caption in at.caption)
