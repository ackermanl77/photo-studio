"""Lens corrections from the photo's own lens (F13): look the camera and lens up in the lensfun
database (the optional `lensfunpy` package, requirements-raw.txt) from the photo's EXIF, then fit
the Lens sliders to lensfun's calibration at the photo's focal length and aperture.

The sliders are simple parametric models (app.lens_corrections: one radial k1 term, a linear
vignette ramp past a midpoint, a radial red/blue scale), so instead of translating lensfun's
coefficients -- whose models and normalization differ -- each slider is least-squares fitted to the
correction field lensfun's own Modifier computes for this frame. The result lands in the ordinary
sliders: it composes with constrain crop, masks and the proxy preview like any manual setting, and
stays adjustable. Offline, CPU-only; a lens lensfun doesn't know simply isn't matched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# EXIF tags (IFD0 and the Exif sub-IFD).
_MAKE, _MODEL, _EXIF_IFD = 0x010F, 0x0110, 0x8769
_LENS_MAKE, _LENS_MODEL, _FOCAL_LENGTH, _F_NUMBER = 0xA433, 0xA434, 0x920A, 0x829D
# The fit runs on a small grid with the photo's aspect ratio; every model is scale-free.
_FIT_LONG_EDGE = 240
# Our sliders' scales (app.lens_corrections): k1 at +-100, red/blue scale at +-100.
_K1_AT_FULL = 0.5
_CA_SCALE_AT_FULL = 0.01


@dataclass
class LensMatch:
    camera: str
    lens: str
    focal_length: float
    aperture: float
    sliders: dict = field(default_factory=dict)  # recipe key -> fitted value

    @property
    def summary(self) -> str:
        return f"{self.lens} at {self.focal_length:g} mm f/{self.aperture:g}"


def lensfun_available() -> bool:
    try:
        import lensfunpy  # noqa: F401
    except ImportError:
        return False
    return True


def _exif_fields(image: Image.Image) -> dict:
    exif = image.getexif()
    sub = exif.get_ifd(_EXIF_IFD)

    def text(value) -> str:
        return str(value).strip("\x00 ").strip() if value else ""

    def number(value) -> float | None:
        try:
            return float(value) if value else None
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    return {
        "make": text(exif.get(_MAKE)), "model": text(exif.get(_MODEL)),
        "lens_make": text(sub.get(_LENS_MAKE)), "lens_model": text(sub.get(_LENS_MODEL)),
        "focal_length": number(sub.get(_FOCAL_LENGTH)), "aperture": number(sub.get(_F_NUMBER)),
    }


def find_lens(image: Image.Image) -> tuple[LensMatch | None, str]:
    """Look the photo's camera and lens up in lensfun and fit the Lens sliders. Returns the match
    (None when there isn't one) and a one-line reason a person can act on."""
    if not lensfun_available():
        return None, "The lens database isn't installed (lensfunpy, in requirements-raw.txt)."
    import lensfunpy

    fields = _exif_fields(image)
    if not fields["focal_length"]:
        return None, "This photo has no focal length in its EXIF, so its lens can't be looked up."
    database = lensfunpy.Database()
    cameras = database.find_cameras(fields["make"], fields["model"]) if fields["make"] and fields["model"] else []
    camera = cameras[0] if cameras else None
    if fields["lens_model"]:
        lenses = database.find_lenses(camera, fields["lens_make"] or None, fields["lens_model"])
    elif camera is not None:
        # No lens tag: a fixed-lens camera's own lens is the only one lensfun lists for it.
        lenses = database.find_lenses(camera)
        lenses = lenses if len(lenses) == 1 else []
    else:
        lenses = []
    if not lenses:
        described = fields["lens_model"] or f"{fields['make']} {fields['model']}".strip() or "this photo's lens"
        return None, f"lensfun has no calibration for {described}."
    lens = lenses[0]
    crop_factor = camera.crop_factor if camera is not None else lens.crop_factor
    aperture = fields["aperture"] or lens.min_aperture or 8.0
    match = LensMatch(
        camera=f"{fields['make']} {fields['model']}".strip(), lens=_lens_name(lens),
        focal_length=round(fields["focal_length"], 1), aperture=round(aperture, 1),
    )
    match.sliders = fit_sliders(lens, crop_factor, image.size, fields["focal_length"], aperture)
    return match, f"Matched {match.summary} (lensfun)."


def _lens_name(lens) -> str:
    """lensfun's model usually starts with the maker already ("Canon EF 24-105mm ...")."""
    maker, model = (lens.maker or "").strip(), (lens.model or "").strip()
    return model if model.lower().startswith(maker.lower()) else f"{maker} {model}".strip()


def _fit_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    scale = _FIT_LONG_EDGE / max(width, height)
    return max(8, round(width * scale)), max(8, round(height * scale))


def fit_sliders(lens, crop_factor: float, size: tuple[int, int], focal_length: float, aperture: float) -> dict:
    """The Lens slider values (recipe keys) that best reproduce lensfun's correction for `lens` at
    this focal length/aperture on a `size` frame."""
    import lensfunpy

    width, height = _fit_size(size)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    vx, vy = xx - center_x, yy - center_y

    def modifier(pixel_format=np.float32):
        mod = lensfunpy.Modifier(lens, crop_factor, width, height)
        mod.initialize(focal_length, aperture, 1000.0, scale=1.0, pixel_format=pixel_format)
        return mod

    sliders = {"lens_distortion": 0, "lens_vignette_amount": 0, "lens_vignette_midpoint": 50, "lens_chromatic_aberration": 0}

    # Distortion: our cv2.undistort samples the source at c + v * (1 + k1 * r^2), r = |v| / max(w, h).
    coords = modifier().apply_geometry_distortion()
    if coords is not None:
        dx, dy = coords[..., 0] - xx, coords[..., 1] - yy
        r2 = (vx ** 2 + vy ** 2) / float(max(width, height)) ** 2
        basis_x, basis_y = vx * r2, vy * r2
        k1 = float((dx * basis_x + dy * basis_y).sum() / max(1e-12, (basis_x ** 2 + basis_y ** 2).sum()))
        sliders["lens_distortion"] = int(round(np.clip(k1 / _K1_AT_FULL * 100.0, -100, 100)))

    # Vignetting: the gain lensfun applies to a flat frame, fitted to 1 + a * ramp(distance; midpoint).
    flat = np.full((height, width, 3), 0.5, dtype=np.float32)
    if modifier().apply_color_modification(flat):
        gain = flat[..., 1].astype(np.float64) / 0.5 - 1.0
        distance = np.hypot(vx, vy) / np.hypot(center_x, center_y)
        best = (np.inf, 0.0, 50)
        for midpoint in range(1, 100):
            m = midpoint / 100.0
            ramp = np.clip((distance - m) / (1.0 - m), 0.0, 1.0)
            denominator = float((ramp ** 2).sum())
            if denominator <= 0:
                continue
            # Fitted within the slider's range, so the midpoint is chosen for the amount we can use.
            amount = float(np.clip((gain * ramp).sum() / denominator, -1.0, 1.0))
            error = float(((gain - amount * ramp) ** 2).sum())
            if error < best[0]:
                best = (error, amount, midpoint)
        amount = int(round(np.clip(best[1] * 100.0, -100, 100)))
        if amount:
            sliders["lens_vignette_amount"], sliders["lens_vignette_midpoint"] = amount, best[2]

    # Lateral CA: red sampled at c + v * (1 - s), blue at c + v * (1 + s), relative to green.
    subpixel = modifier().apply_subpixel_distortion()
    if subpixel is not None:
        red_x, red_y = subpixel[..., 0, 0] - subpixel[..., 1, 0], subpixel[..., 0, 1] - subpixel[..., 1, 1]
        blue_x, blue_y = subpixel[..., 2, 0] - subpixel[..., 1, 0], subpixel[..., 2, 1] - subpixel[..., 1, 1]
        norm = 2.0 * float((vx ** 2 + vy ** 2).sum())
        scale = float((-(red_x * vx + red_y * vy) + (blue_x * vx + blue_y * vy)).sum() / max(1e-12, norm))
        sliders["lens_chromatic_aberration"] = int(round(np.clip(scale / _CA_SCALE_AT_FULL * 100.0, -100, 100)))
    return sliders
