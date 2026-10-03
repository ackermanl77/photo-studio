from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

from app.ai_upscale import upscale as ai_upscale
from app.dehaze import apply_dehaze
from app.effects import apply_film_grain
from app.lens_corrections import apply_lens_corrections, correct_vignette
from app.professional_color import apply_color_grading, apply_hsl_mixer, apply_vibrance
from app.raw_import import open_image


# Rec. 709 luma weights -- the luminance Clarity and Highlights/Shadows act on (the old
# Highlights/Shadows used mean(R, G, B), which treats a saturated blue as darker than it looks).
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# ITU-R 601 -- the weights PIL's convert("L") (and so ImageEnhance.Contrast/Color) uses; the
# 16-bit path's Contrast/Saturation use them too, so a 16-bit export matches the 8-bit preview.
_LUMA_601 = np.array([0.299, 0.587, 0.114], dtype=np.float32)
# Spatial scales as a fraction of the image's long edge, so a reduced-size preview proxy and the
# full-resolution export get the same look (B5 in the improvement report).
CLARITY_SIGMA_FRACTION = 0.012
TONE_BASE_RADIUS_FRACTION = 0.02


def _luminance(array: np.ndarray) -> np.ndarray:
    """(H, W, 3) float32 0-255 -> (H, W) float32 0-1."""
    return (array @ _LUMA) / 255.0


def guided_filter(guide: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al.'s guided filter, self-guided, built from cv2.boxFilter (no opencv-contrib): an
    edge-preserving blur whose output follows the guide's edges, so a tone change computed from it
    doesn't bleed (halo) across a sharp boundary the way a Gaussian base would."""
    size = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, -1, size)
    variance = cv2.boxFilter(guide * guide, -1, size) - mean_i * mean_i
    a = variance / (variance + eps)
    b = mean_i - a * mean_i
    return cv2.boxFilter(a, -1, size) * guide + cv2.boxFilter(b, -1, size)


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1 / 2.4) - 0.055)


# Linear values above the knee roll off smoothly toward 1.0 instead of clipping hard, so +EV
# compresses highlights the way a film/RAW editor's shoulder does rather than blowing them flat.
_SHOULDER_KNEE = 0.8


def exposure_curve(values: np.ndarray, ev: float) -> np.ndarray:
    """Exposure of `ev` stops on sRGB-encoded values (0..1): decode to linear light, multiply by
    2**ev, roll the highlights off above _SHOULDER_KNEE, encode back. Shared by the 8-bit LUT below
    and the 16-bit path's float arrays, so both compute the same curve."""
    linear = _srgb_to_linear(values) * (2.0 ** float(ev))
    over = linear > _SHOULDER_KNEE
    headroom = 1.0 - _SHOULDER_KNEE
    shouldered = _SHOULDER_KNEE + headroom * (1.0 - np.exp(-(linear[over] - _SHOULDER_KNEE) / headroom))
    # Faded in over the first stop of brightening, so 0 EV is exactly the identity and a small
    # change never jumps.
    weight = min(1.0, max(0.0, float(ev)))
    linear[over] = (1.0 - weight) * np.minimum(linear[over], 1.0) + weight * shouldered
    return _linear_to_srgb(linear)


def exposure_lut(ev: float) -> list[int]:
    """256-entry LUT for an exposure change of `ev` stops (exposure_curve()). One stop = twice the
    light, the way Lightroom/darktable define exposure (the old slider multiplied the gamma-encoded
    values, and its 0.5-1.5 range was only about -1..+0.6 EV)."""
    curve = exposure_curve(np.arange(256, dtype=np.float64) / 255.0, ev)
    return [int(v) for v in np.clip(np.round(curve * 255.0), 0, 255)]


def _clarity_float(arr: np.ndarray, clarity: float) -> np.ndarray:
    """_apply_clarity()'s math on a 0-255 float array, unclipped and unquantized (16-bit path)."""
    luminance = _luminance(arr)
    sigma = max(1.0, CLARITY_SIGMA_FRACTION * max(arr.shape[:2]))
    blurred = cv2.GaussianBlur(luminance, (0, 0), sigma)
    midtones = 4.0 * luminance * (1.0 - luminance)
    new_luminance = np.clip(luminance + (clarity / 100.0) * 1.5 * (luminance - blurred) * midtones, 0.0, 1.0)
    return arr + ((new_luminance - luminance) * 255.0)[..., None]


def _apply_clarity(arr: np.ndarray, clarity: float) -> np.ndarray:
    """Clarity = midtone local contrast on luminance only: unsharp-mask the luminance with a large
    radius (~1.2 % of the long edge), weight the boost toward the midtones (4*L*(1-L)), and add the
    luminance change back to all three channels so colors don't shift. Negative values soften
    local contrast instead. It used to be ImageEnhance.Contrast -- the same global operation as the
    Contrast slider -- despite promising "midtone contrast at a larger scale than Texture"."""
    luminance = _luminance(arr)
    sigma = max(1.0, CLARITY_SIGMA_FRACTION * max(arr.shape[:2]))
    blurred = cv2.GaussianBlur(luminance, (0, 0), sigma)
    midtones = 4.0 * luminance * (1.0 - luminance)
    new_luminance = np.clip(luminance + (clarity / 100.0) * 1.5 * (luminance - blurred) * midtones, 0.0, 1.0)
    arr = arr + ((new_luminance - luminance) * 255.0)[..., None]
    return np.clip(arr, 0, 255).astype(np.uint8)


def auto_enhance_recipe(image: Image.Image) -> dict:
    """Image-adaptive "Auto Enhance" settings (recipe keys, see app.edit_schema.RECIPE_CONTROLS),
    measured on a <=512 px proxy -- it used to write the same constants (exposure 1.12, contrast
    1.15, saturation 1.08, denoise 3, sharpen) whatever the photo:
    - exposure: the EV that moves median luminance to ~0.45 (sRGB), within +-2 stops;
    - black/white points: the 0.1 / 99.9 % luminance percentiles, fed to the tone curve when the
      photo doesn't already use the full range;
    - white balance: gray-world (the average color treated as neutral), at half strength;
    - vibrance: more for a flat, low-chroma photo, none for an already colorful one;
    - denoise: only when the estimated noise (Immerkaer's Laplacian method) is high.
    Every key is always present, so the result also resets what a previous Auto Enhance set."""
    from app.professional_color import estimate_white_balance

    proxy = image.convert("RGB")
    proxy.thumbnail((512, 512), Image.Resampling.LANCZOS)
    arr = np.asarray(proxy, dtype=np.float32)
    luminance = arr @ _LUMA
    median = max(1.0, float(np.median(luminance))) / 255.0
    ev = float(np.clip(2.2 * np.log2(0.45 / median), -2.0, 2.0))
    ev = round(round(ev / 0.05) * 0.05, 2) if abs(ev) >= 0.1 else 0.0
    low, high = (float(v) for v in np.percentile(luminance, (0.1, 99.9)))
    black, white = int(np.clip(round(low), 0, 64)), int(np.clip(round(high), 192, 255))
    mean_color = Image.new("RGB", (1, 1), tuple(int(c) for c in arr.reshape(-1, 3).mean(axis=0)))
    temperature, tint = estimate_white_balance(mean_color, 0.5, 0.5)
    chroma = float((arr.max(axis=2) - arr.min(axis=2)).mean()) / 255.0
    gray = luminance.astype(np.float64)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    laplacian = cv2.filter2D(gray, -1, kernel)[1:-1, 1:-1]
    noise_sigma = float(np.sqrt(np.pi / 2.0) / 6.0 * np.abs(laplacian).mean()) if laplacian.size else 0.0
    return {
        "exposure_ev": ev,
        "curve_enabled": black > 2 or white < 253,
        "curve_black": black, "curve_shadows": 64, "curve_mid": 128, "curve_highlights": 192, "curve_white": white,
        "temperature": int(round(temperature / 2)), "tint": int(round(tint / 2)),
        "vibrance": int(np.clip(round((0.25 - chroma) * 120), 0, 30)),
        "denoise": 3 if noise_sigma > 4.0 else 0,
    }


class PhotoProcessor:
    # Multiplier for the few pixel-unit radii below (texture blur, sharpen, dehaze patch), which
    # are tuned for a full-resolution photo. app.pipeline sets it to the preview scale when
    # rendering a reduced-size interactive proxy, so the proxy looks like a downscaled copy of the
    # export (and a built-in preset's own sharpen call scales with it too). 1.0 = unchanged.
    pixel_scale: float = 1.0
    # High-precision mode (B4, app.high_bit): when not None, a float32 HxWx3 array (sRGB-encoded,
    # 0..1) that is the real working image. The tone/colour/mask-blend/geometry/crop/vignette
    # methods compute on it directly; every other step still runs on the 8-bit `image` and its
    # *change* is added onto the array (see the `image` setter), so the float data keeps its
    # precision through steps that have no float implementation. None (the default) = the plain
    # 8-bit behaviour, byte for byte.
    hdr: np.ndarray | None = None
    _image_stale: bool = False

    @property
    def image(self) -> Image.Image:
        if self.hdr is not None and self._image_stale:
            from app.high_bit import quantize8

            self._image = Image.fromarray(quantize8(self.hdr), mode="RGB")
            self._image_stale = False
        return self._image

    @image.setter
    def image(self, value: Image.Image) -> None:
        if self.hdr is None:
            self._image = value
            return
        current = self.image
        if value is current:
            return
        value = value.convert("RGB")
        new = np.asarray(value, dtype=np.float32)
        if value.size == current.size:
            # An 8-bit step on the quantized image: carry its change over to the float data.
            self.hdr = np.clip(self.hdr + (new - np.asarray(current, dtype=np.float32)) / 255.0, 0.0, 1.0).astype(np.float32)
        else:
            # A size-changing 8-bit step (only AI Upscale) -- precision restarts from its result.
            self.hdr = new / 255.0
        self._image = value
        self._image_stale = False

    def enable_high_precision(self, array: np.ndarray | None = None) -> "PhotoProcessor":
        """Switch to high-precision mode, from `array` (a 16-bit source's float pixels, same size)
        or from the current 8-bit image."""
        from app.high_bit import from_image

        if array is None or array.shape[:2] != (self._image.height, self._image.width):
            array = from_image(self._image)
        self.hdr = np.ascontiguousarray(array, dtype=np.float32)
        self._image_stale = True
        return self

    def _set_hdr(self, array: np.ndarray) -> "PhotoProcessor":
        self.hdr = np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)
        self._image_stale = True
        return self

    def apply_geometry(self, options: dict) -> "PhotoProcessor":
        """The pipeline's Geometry step (app.geometry.apply_geometry) on whichever data is real."""
        from app.geometry import apply_geometry

        if self.hdr is not None:
            return self._set_hdr(apply_geometry(self.hdr, options))
        self.image = apply_geometry(self.image, options)
        return self

    def __init__(self, input_path: str | Path):
        self.input_path = Path(input_path)
        self.image = open_image(self.input_path)

    @classmethod
    def from_bytes(cls, content: bytes, filename: str) -> "PhotoProcessor":
        from app.raw_import import open_bytes

        processor = cls.__new__(cls)
        processor.input_path = Path(filename)
        processor.image = open_bytes(content, filename)
        return processor

    def apply_brightness(self, value: float) -> "PhotoProcessor":
        if self.hdr is not None:
            return self._set_hdr(self.hdr * float(value)) if value != 1.0 else self
        enhancer = ImageEnhance.Brightness(self.image)
        self.image = enhancer.enhance(value)
        return self

    def apply_contrast(self, value: float) -> "PhotoProcessor":
        if self.hdr is not None:
            # ImageEnhance.Contrast: blend toward the mean (ITU-R 601) luminance.
            if value == 1.0:
                return self
            mean = float((self.hdr @ _LUMA_601).mean())
            return self._set_hdr(mean + float(value) * (self.hdr - mean))
        enhancer = ImageEnhance.Contrast(self.image)
        self.image = enhancer.enhance(value)
        return self

    def apply_saturation(self, value: float) -> "PhotoProcessor":
        if self.hdr is not None:
            # ImageEnhance.Color: blend toward each pixel's (ITU-R 601) gray.
            if value == 1.0:
                return self
            gray = (self.hdr @ _LUMA_601)[..., None]
            return self._set_hdr(gray + float(value) * (self.hdr - gray))
        enhancer = ImageEnhance.Color(self.image)
        self.image = enhancer.enhance(value)
        return self

    def apply_exposure(self, value: float) -> "PhotoProcessor":
        if self.hdr is not None:
            return self._set_hdr(self.hdr * float(value)) if value != 1.0 else self
        arr = np.array(self.image, dtype=np.float32)
        self.image = Image.fromarray(np.clip(arr * value, 0, 255).astype(np.uint8), mode="RGB")
        return self

    def apply_exposure_ev(self, ev: float = 0.0) -> "PhotoProcessor":
        """Exposure in stops, in linear light (see exposure_lut)."""
        if ev and self.hdr is not None:
            return self._set_hdr(exposure_curve(self.hdr, ev))
        if ev:
            self.image = self.image.convert("RGB").point(exposure_lut(ev) * 3)
        return self

    def apply_temperature_tint(self, temperature: float = 0.0, tint: float = 0.0) -> "PhotoProcessor":
        temperature = max(-100.0, min(100.0, float(temperature))) / 100.0
        tint = max(-100.0, min(100.0, float(tint))) / 100.0
        if self.hdr is not None and not temperature and not tint:
            return self
        arr = self.hdr * 255.0 if self.hdr is not None else np.array(self.image, dtype=np.float32)
        arr[:, :, 0] *= 1.0 + temperature * 0.25
        arr[:, :, 2] *= 1.0 - temperature * 0.25
        arr[:, :, 1] *= 1.0 - tint * 0.2
        arr[:, :, 0] *= 1.0 + tint * 0.1
        arr[:, :, 2] *= 1.0 + tint * 0.1
        if self.hdr is not None:
            return self._set_hdr(arr / 255.0)
        self.image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        return self

    def apply_hsl_mixer(self, adjustments: dict[str, dict[str, float]]) -> "PhotoProcessor":
        self.image = apply_hsl_mixer(self.image, adjustments)
        return self

    def apply_vibrance(self, amount: float = 0.0) -> "PhotoProcessor":
        self.image = apply_vibrance(self.image, amount)
        return self

    def apply_color_grading(
        self,
        shadows_hue: float = 0.0, shadows_intensity: float = 0.0,
        midtones_hue: float = 0.0, midtones_intensity: float = 0.0,
        highlights_hue: float = 0.0, highlights_intensity: float = 0.0,
        balance: float = 0.0,
    ) -> "PhotoProcessor":
        self.image = apply_color_grading(
            self.image, shadows_hue, shadows_intensity, midtones_hue, midtones_intensity,
            highlights_hue, highlights_intensity, balance,
        )
        return self

    def apply_dehaze(self, strength: float = 0.0) -> "PhotoProcessor":
        self.image = apply_dehaze(self.image, strength, pixel_scale=self.pixel_scale)
        return self

    def apply_lens_corrections(
        self,
        distortion: float = 0.0,
        vignette_amount: float = 0.0,
        vignette_midpoint: float = 50.0,
        chromatic_aberration: float = 0.0,
        perspective_vertical: float = 0.0,
        perspective_horizontal: float = 0.0,
    ) -> "PhotoProcessor":
        self.image = apply_lens_corrections(
            self.image, distortion, vignette_amount, vignette_midpoint, chromatic_aberration,
            perspective_vertical, perspective_horizontal,
        )
        return self

    def apply_curves(self, points: list[tuple[int, int]]) -> "PhotoProcessor":
        if len(points) < 2:
            raise ValueError("Curves require at least two points")
        ordered = sorted((int(x), int(y)) for x, y in points)
        x_values = np.array([point[0] for point in ordered], dtype=np.float32)
        y_values = np.array([point[1] for point in ordered], dtype=np.float32)
        if x_values[0] < 0 or x_values[-1] > 255 or np.any(np.diff(x_values) <= 0):
            raise ValueError("Curve input values must be unique and within 0-255")
        if self.hdr is not None:
            return self._set_hdr(np.interp(self.hdr * 255.0, x_values, y_values).astype(np.float32) / 255.0)
        lookup = np.interp(np.arange(256), x_values, y_values).clip(0, 255).astype(np.uint8)
        arr = np.array(self.image)
        self.image = Image.fromarray(lookup[arr], mode="RGB")
        return self

    def remove_objects(self, mask: np.ndarray | Image.Image | str | Path, radius: float = 3.0) -> "PhotoProcessor":
        mask_array = self._mask_array(mask)
        source = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        repaired = cv2.inpaint(source, mask_array, max(1.0, float(radius)), cv2.INPAINT_TELEA)
        self.image = Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB), mode="RGB")
        return self

    def apply_masked_adjustments(
        self,
        mask: np.ndarray | Image.Image | str | Path,
        exposure: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        exposure_ev: float = 0.0,
        temperature: float = 0.0,
        tint: float = 0.0,
        clarity: float = 0.0,
    ) -> "PhotoProcessor":
        """Apply an adjustment set only where `mask` is selected (blended by the mask's value)."""
        mask_array = self._mask_array(mask).astype(np.float32) / 255.0
        local = PhotoProcessor.from_image(self.image)
        local.pixel_scale = self.pixel_scale
        if self.hdr is not None:
            local.enable_high_precision(self.hdr)
            local.apply_exposure(exposure).apply_exposure_ev(exposure_ev).apply_contrast(contrast).apply_saturation(saturation)
            local.apply_temperature_tint(temperature, tint).apply_texture_clarity(0.0, clarity)
            blend = mask_array[..., None]
            return self._set_hdr(self.hdr * (1.0 - blend) + local.hdr * blend)
        original = np.array(self.image)
        edited = (
            local
            .apply_exposure(exposure)
            .apply_exposure_ev(exposure_ev)
            .apply_contrast(contrast)
            .apply_saturation(saturation)
            .apply_temperature_tint(temperature, tint)
            .apply_texture_clarity(0.0, clarity)
            .as_numpy()
        )
        blend = mask_array[..., None]
        self.image = Image.fromarray(np.clip(original * (1.0 - blend) + edited * blend, 0, 255).astype(np.uint8), mode="RGB")
        return self

    @classmethod
    def from_image(cls, image: Image.Image) -> "PhotoProcessor":
        processor = cls.__new__(cls)
        processor.input_path = None
        processor.image = image.convert("RGB").copy()
        return processor

    def _mask_array(self, mask: np.ndarray | Image.Image | str | Path) -> np.ndarray:
        if isinstance(mask, (str, Path)):
            mask = Image.open(mask)
        if isinstance(mask, Image.Image):
            mask = mask.convert("L")
            if mask.size != self.image.size:
                raise ValueError("Mask dimensions must match the image dimensions")
            return np.array(mask)
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        if mask_array.shape != (self.image.height, self.image.width):
            raise ValueError("Mask dimensions must match the image dimensions")
        return np.clip(mask_array, 0, 255).astype(np.uint8)

    def apply_highlights_shadows(self, highlights: float = 0.0, shadows: float = 0.0) -> "PhotoProcessor":
        """Edge-aware Highlights/Shadows: luminance is split into an edge-preserving base (a guided
        filter over ~2 % of the long edge) and the detail on top of it; the tone shift is computed
        from the *base*, then the detail is added back unchanged. A shadowed face lifts as a region
        while its texture -- and the bright background next to it -- stays put, instead of every
        dark pixel getting the same flat offset (the old version, which also used mean(RGB) rather
        than luminance). `highlights`/`shadows` keep their old scale: roughly 8-bit levels at the
        extreme of each range."""
        if not highlights and not shadows:
            return self
        arr = self.hdr * 255.0 if self.hdr is not None else np.array(self.image, dtype=np.float32)
        luminance = _luminance(arr)
        radius = max(1, int(round(TONE_BASE_RADIUS_FRACTION * max(self.image.size))))
        base = guided_filter(luminance, radius, 0.01)
        detail = luminance - base
        shadow_weight = np.clip((0.5 - base) / 0.5, 0.0, 1.0)
        highlight_weight = np.clip((base - 0.5) / 0.5, 0.0, 1.0)
        # Smoothstep so the weights ease in from the midtones instead of starting with a kink.
        shadow_weight = shadow_weight * shadow_weight * (3 - 2 * shadow_weight)
        highlight_weight = highlight_weight * highlight_weight * (3 - 2 * highlight_weight)
        new_base = np.clip(base + (float(shadows) / 255.0) * shadow_weight * 1.6 + (float(highlights) / 255.0) * highlight_weight * 1.6, 0.0, 1.0)
        delta = (new_base + detail - luminance) * 255.0
        arr += delta[..., None]
        if self.hdr is not None:
            return self._set_hdr(arr / 255.0)
        self.image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        return self

    def apply_texture_clarity(self, texture: float = 0.0, clarity: float = 0.0) -> "PhotoProcessor":
        texture = max(-100.0, min(100.0, float(texture)))
        clarity = max(-100.0, min(100.0, float(clarity)))
        if texture:
            radius = max(1, int(round(3 - abs(texture) / 60))) * self.pixel_scale
            blurred = self.image.filter(ImageFilter.GaussianBlur(radius=radius))
            original = np.array(self.image, dtype=np.float32)
            detail = original - np.array(blurred, dtype=np.float32)
            original += detail * (texture / 100.0)
            self.image = Image.fromarray(np.clip(original, 0, 255).astype(np.uint8), mode="RGB")
        if clarity and self.hdr is not None:
            self._set_hdr(_clarity_float(self.hdr * 255.0, clarity) / 255.0)
        elif clarity:
            self.image = Image.fromarray(_apply_clarity(np.array(self.image, dtype=np.float32), clarity), mode="RGB")
        return self

    def crop(self, left: int, top: int, right: int, bottom: int) -> "PhotoProcessor":
        width, height = self.image.size
        box = (max(0, left), max(0, top), min(width, right), min(height, bottom))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Crop coordinates must describe a non-empty region")
        if self.hdr is not None:
            return self._set_hdr(self.hdr[box[1]:box[3], box[0]:box[2]])
        self.image = self.image.crop(box)
        return self

    def fit_aspect_ratio(self, width_ratio: int, height_ratio: int) -> "PhotoProcessor":
        if width_ratio <= 0 or height_ratio <= 0:
            raise ValueError("Aspect-ratio values must be positive")
        width, height = self.image.size
        target = width_ratio / height_ratio
        current = width / height
        if current > target:
            new_width = int(height * target)
            left = (width - new_width) // 2
            return self.crop(left, 0, left + new_width, height)
        new_height = int(width / target)
        top = (height - new_height) // 2
        return self.crop(0, top, width, top + new_height)

    def apply_denoise(self, radius: int = 3, color_radius: int = 0) -> "PhotoProcessor":
        """Edge-aware denoising via OpenCV's Non-Local Means (`cv2.fastNlMeansDenoisingColored`).
        Classical, not a neural network -- this app is explicitly CPU-first with no GPU dependency
        and avoids heavy new dependencies (see docs/PROJECT_STATUS.md), and a real learned denoiser
        needs both (a several-MB-plus model file, and a slow-without-a-GPU inference pass). Still a
        meaningful upgrade over the flat `ImageFilter.MedianFilter` this used to be: Non-Local Means
        looks for similar patches across the whole search window instead of just averaging pixels
        in a fixed neighborhood, so it smooths noise while preserving edges/detail far better than a
        median blur does. `radius` keeps the same 0-9 range the UI's "Denoise" slider already used
        for the old filter's kernel size, now driving Non-Local Means' "h" (luminance) filter-
        strength parameter instead (empirically timed at well under a second for a multi-megapixel
        photo).

        `color_radius` (0-9, default 0) independently drives the same call's `hColor` parameter --
        `fastNlMeansDenoisingColored` already accepts separate luminance/color strengths, so
        RawTherapee's/darktable's "split luminance vs. chroma noise reduction" (color noise usually
        wants heavier smoothing than luminance detail can afford to lose) needed no new algorithm,
        just exposing the parameter this call already had. `color_radius=0` means "no override" --
        `hColor` falls back to the same strength as `h`, exactly this method's pre-existing
        behavior, so a caller that never passes `color_radius` sees no change at all.
        """
        radius = max(1, int(radius))
        strength = float(radius) * 1.8
        color_strength = float(color_radius) * 1.8 if color_radius else strength
        array = np.array(self.image.convert("RGB"))
        denoised = cv2.fastNlMeansDenoisingColored(array, None, strength, color_strength, 7, 21)
        self.image = Image.fromarray(denoised, mode="RGB")
        return self

    def apply_sharpen(self, radius: float = 1.0, percent: float = 150.0) -> "PhotoProcessor":
        radius = max(1, int(round(radius)))
        if self.pixel_scale != 1.0:
            radius = max(0.1, radius * self.pixel_scale)
        percent = max(0, int(percent))
        self.image = self.image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent))
        return self

    def apply_post_crop_vignette(self, amount: float = 0.0, midpoint: float = 50.0) -> "PhotoProcessor":
        """A creative "Effects" vignette applied to the final, already-cropped frame -- unlike the
        Lens tab's vignette (a corrective tool run before Crop, aimed at a lens's own optical
        falloff), this is meant to be dialed in *after* framing, the same distinction Lightroom
        draws between its Lens Corrections vignette and its separate Effects > Post-Crop Vignette.
        Reuses `correct_vignette()` unchanged -- it's the identical radial darken/brighten math,
        just applied at a different pipeline position and for a different (creative, not
        corrective) reason, so a second implementation would just be duplicated code."""
        if self.hdr is not None:
            return self._set_hdr(correct_vignette(self.hdr, amount, midpoint))
        self.image = correct_vignette(self.image, amount, midpoint)
        return self

    def apply_film_grain(self, amount: float = 0.0) -> "PhotoProcessor":
        self.image = apply_film_grain(self.image, amount)
        return self

    def apply_ai_upscale(self, factor: int = 2) -> "PhotoProcessor":
        """Real-ESRGAN neural super-resolution -- the one deliberate exception to this app's
        CPU-first, no-ML-dependency design (see "Known Issues" in docs/PROJECT_STATUS.md). A thin
        delegating wrapper into app.ai_upscale.upscale(), same pattern as every other apply_*
        method here that delegates into a feature module. Requires the optional
        requirements-gpu.txt packages (torch, realesrgan) -- raises RuntimeError via
        app.ai_upscale.upscale() if they aren't installed; callers should check
        app.ai_upscale.ai_upscale_available() first (both app.backend.process_image() and the
        frontend's control already do)."""
        self.image = ai_upscale(self.image, factor)
        return self

    def auto_enhance(self) -> "PhotoProcessor":
        self.apply_exposure(1.12)
        self.apply_contrast(1.15)
        self.apply_saturation(1.08)
        self.apply_denoise(3)
        self.apply_sharpen()
        return self

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(output)
        return output

    def as_numpy(self) -> np.ndarray:
        return np.array(self.image)
