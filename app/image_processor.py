from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

from app.dehaze import apply_dehaze
from app.lens_corrections import apply_lens_corrections
from app.professional_color import apply_color_grading, apply_hsl_mixer, apply_vibrance
from app.raw_import import open_image


class PhotoProcessor:
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
        enhancer = ImageEnhance.Brightness(self.image)
        self.image = enhancer.enhance(value)
        return self

    def apply_contrast(self, value: float) -> "PhotoProcessor":
        enhancer = ImageEnhance.Contrast(self.image)
        self.image = enhancer.enhance(value)
        return self

    def apply_saturation(self, value: float) -> "PhotoProcessor":
        enhancer = ImageEnhance.Color(self.image)
        self.image = enhancer.enhance(value)
        return self

    def apply_exposure(self, value: float) -> "PhotoProcessor":
        arr = np.array(self.image, dtype=np.float32)
        self.image = Image.fromarray(np.clip(arr * value, 0, 255).astype(np.uint8), mode="RGB")
        return self

    def apply_temperature_tint(self, temperature: float = 0.0, tint: float = 0.0) -> "PhotoProcessor":
        temperature = max(-100.0, min(100.0, float(temperature))) / 100.0
        tint = max(-100.0, min(100.0, float(tint))) / 100.0
        arr = np.array(self.image, dtype=np.float32)
        arr[:, :, 0] *= 1.0 + temperature * 0.25
        arr[:, :, 2] *= 1.0 - temperature * 0.25
        arr[:, :, 1] *= 1.0 - tint * 0.2
        arr[:, :, 0] *= 1.0 + tint * 0.1
        arr[:, :, 2] *= 1.0 + tint * 0.1
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
        self.image = apply_dehaze(self.image, strength)
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
    ) -> "PhotoProcessor":
        mask_array = self._mask_array(mask).astype(np.float32) / 255.0
        original = np.array(self.image)
        edited = (
            PhotoProcessor.from_image(self.image)
            .apply_exposure(exposure)
            .apply_contrast(contrast)
            .apply_saturation(saturation)
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
        arr = np.array(self.image, dtype=np.float32)
        luminance = arr.mean(axis=2)
        highlights_mask = np.clip((luminance - 128.0) / 127.0, 0.0, 1.0)[..., None]
        shadows_mask = np.clip((128.0 - luminance) / 128.0, 0.0, 1.0)[..., None]
        arr += highlights_mask * float(highlights) + shadows_mask * float(shadows)
        self.image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        return self

    def apply_texture_clarity(self, texture: float = 0.0, clarity: float = 0.0) -> "PhotoProcessor":
        texture = max(-100.0, min(100.0, float(texture)))
        clarity = max(-100.0, min(100.0, float(clarity)))
        if texture:
            radius = max(1, int(round(3 - abs(texture) / 60)))
            blurred = self.image.filter(ImageFilter.GaussianBlur(radius=radius))
            original = np.array(self.image, dtype=np.float32)
            detail = original - np.array(blurred, dtype=np.float32)
            original += detail * (texture / 100.0)
            self.image = Image.fromarray(np.clip(original, 0, 255).astype(np.uint8), mode="RGB")
        if clarity:
            self.image = ImageEnhance.Contrast(self.image).enhance(1.0 + clarity / 100.0)
        return self

    def crop(self, left: int, top: int, right: int, bottom: int) -> "PhotoProcessor":
        width, height = self.image.size
        box = (max(0, left), max(0, top), min(width, right), min(height, bottom))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Crop coordinates must describe a non-empty region")
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

    def apply_white_balance(self, red_gain: float = 1.0, blue_gain: float = 1.0) -> "PhotoProcessor":
        arr = np.array(self.image, dtype=np.float32)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * red_gain, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * blue_gain, 0, 255)
        self.image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        return self

    def apply_denoise(self, radius: int = 3) -> "PhotoProcessor":
        """Edge-aware denoising via OpenCV's Non-Local Means (`cv2.fastNlMeansDenoisingColored`).
        Classical, not a neural network -- this app is explicitly CPU-first with no GPU dependency
        and avoids heavy new dependencies (see docs/PROJECT_STATUS.md), and a real learned denoiser
        needs both (a several-MB-plus model file, and a slow-without-a-GPU inference pass). Still a
        meaningful upgrade over the flat `ImageFilter.MedianFilter` this used to be: Non-Local Means
        looks for similar patches across the whole search window instead of just averaging pixels
        in a fixed neighborhood, so it smooths noise while preserving edges/detail far better than a
        median blur does. `radius` keeps the same 0-9 range the UI's "Denoise" slider already used
        for the old filter's kernel size, now driving Non-Local Means' "h" filter-strength parameter
        instead (empirically timed at well under a second for a multi-megapixel photo).
        """
        radius = max(1, int(radius))
        strength = float(radius) * 1.8
        array = np.array(self.image.convert("RGB"))
        denoised = cv2.fastNlMeansDenoisingColored(array, None, strength, strength, 7, 21)
        self.image = Image.fromarray(denoised, mode="RGB")
        return self

    def apply_sharpen(self, radius: float = 1.0, percent: float = 150.0) -> "PhotoProcessor":
        radius = max(1, int(round(radius)))
        percent = max(0, int(percent))
        self.image = self.image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent))
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
