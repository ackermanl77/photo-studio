"""HDR merge and panorama stitching (F18), classical OpenCV -- no model.

- **HDR**: bracketed exposures of one scene are aligned (``createAlignMTB``, median-threshold bitmaps,
  robust to the exposure differences) and combined, either by **exposure fusion**
  (``createMergeMertens``: each pixel weighted by contrast, saturation and well-exposedness; needs no
  exposure times and gives a natural, already displayable result) or by a true **HDR merge**
  (``createCalibrateDebevec`` + ``createMergeDebevec`` from the EXIF exposure times, then the
  Reinhard tone map) for a stronger, flatter look.
- **Panorama**: overlapping frames are stitched by OpenCV's ``Stitcher`` (feature matching,
  bundle adjustment, spherical warp, multi-band blending); the ragged edge can be cropped to the
  largest rectangle of real pixels.

Either result is a float32 RGB array in 0..1, written as a 16-bit TIFF (:func:`encode_16bit_tiff`)
and opened as a new source, so the whole edit runs on it in high precision. Inputs are decoded at
full precision (16-bit PNG/TIFF and RAW at 16 bits) through ``app.high_bit``. Streamlit-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image

HDR_METHODS = ("fusion", "debevec")
MAX_INPUTS = 12


class MergeError(ValueError):
    """Why the photos couldn't be merged (shown to the user)."""


@dataclass
class SourcePhoto:
    name: str
    pixels: np.ndarray  # float32 RGB 0..1
    exposure_time: Optional[float]  # seconds, from EXIF; None when unknown


def _exposure_time(image: Image.Image) -> Optional[float]:
    try:
        exif = image.getexif()
        value = exif.get_ifd(0x8769).get(33434) or exif.get(33434)  # ExposureTime
        if value:
            value = float(value)
            return value if value > 0 else None
    except Exception:  # noqa: BLE001 - unreadable EXIF just means "unknown"
        return None
    return None


def load_source(content: bytes, filename: str) -> SourcePhoto:
    """Decode one input at full precision, orientation applied, with its exposure time."""
    from app.high_bit import decode_high_bit
    from app.raw_import import open_bytes

    image = open_bytes(content, filename)
    exposure = _exposure_time(image)  # RAW files decoded here carry no EXIF: fusion doesn't need it
    pixels = decode_high_bit(content, filename)
    if pixels is None:
        from PIL import ImageOps

        pixels = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32) / 255.0
    return SourcePhoto(filename, np.ascontiguousarray(pixels[..., :3], dtype=np.float32), exposure)


def _same_size(photos: Sequence[SourcePhoto]) -> list[np.ndarray]:
    """Bracketed frames from one camera share a size; a stray one is resized to the first."""
    import cv2

    height, width = photos[0].pixels.shape[:2]
    out = []
    for photo in photos:
        pixels = photo.pixels
        if pixels.shape[:2] != (height, width):
            pixels = cv2.resize(pixels, (width, height), interpolation=cv2.INTER_AREA)
        out.append(pixels)
    return out


def _to8(pixels: np.ndarray) -> np.ndarray:
    return np.clip(pixels * 255.0 + 0.5, 0, 255).astype(np.uint8)


def merge_hdr(photos: Sequence[SourcePhoto], method: str = "fusion", align: bool = True,
              progress: Optional[Callable[[str], None]] = None) -> np.ndarray:
    """Merge bracketed exposures into one float32 RGB image (0..1)."""
    import cv2

    if method not in HDR_METHODS:
        raise MergeError(f"Unknown HDR method: {method!r}")
    if not 2 <= len(photos) <= MAX_INPUTS:
        raise MergeError(f"An HDR merge needs 2 to {MAX_INPUTS} photos of the same scene.")
    frames = _same_size(photos)
    # OpenCV works in BGR; 8-bit copies drive the alignment and the Debevec calibration.
    bgr8 = [cv2.cvtColor(_to8(frame), cv2.COLOR_RGB2BGR) for frame in frames]
    if align:
        if progress:
            progress("Aligning the photos…")
        aligner = cv2.createAlignMTB()
        gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in bgr8]
        shifts = [aligner.calculateShift(gray[0], frame) for frame in gray[1:]]
        for index, shift in enumerate(shifts, start=1):
            matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            size = (frames[index].shape[1], frames[index].shape[0])
            frames[index] = cv2.warpAffine(frames[index], matrix, size, borderMode=cv2.BORDER_REPLICATE)
            bgr8[index] = cv2.warpAffine(bgr8[index], matrix, size, borderMode=cv2.BORDER_REPLICATE)
    if progress:
        progress("Merging the exposures…")
    if method == "fusion":
        bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        fused = cv2.createMergeMertens().process(bgr)
        result = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)
        # Mertens' weights can overshoot slightly; stretch the 0.1-99.9 % range back into 0..1.
        low, high = np.percentile(result, (0.1, 99.9))
        result = (result - min(low, 0.0)) / max(high - min(low, 0.0), 1e-6)
    else:
        times = [photo.exposure_time for photo in photos]
        if any(t is None for t in times):
            raise MergeError("The HDR merge needs every photo's exposure time (EXIF); "
                             "use exposure fusion for photos without it.")
        if len(set(times)) < 2:
            raise MergeError("The photos all have the same exposure time: bracket the exposures.")
        times_arr = np.array(times, dtype=np.float32)
        response = cv2.createCalibrateDebevec().process(bgr8, times_arr)
        radiance = cv2.createMergeDebevec().process(bgr8, times_arr, response)
        mapped = cv2.createTonemapReinhard(gamma=2.2, intensity=0.0, light_adapt=0.8, color_adapt=0.0).process(radiance)
        result = cv2.cvtColor(np.nan_to_num(mapped), cv2.COLOR_BGR2RGB)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _crop_to_content(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The largest axis-aligned rectangle inside `mask` (non-zero = real pixels), found by shrinking
    the bounding box from its emptiest side until no empty pixel is left."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return pixels
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    while top < bottom and left < right:
        window = mask[top:bottom + 1, left:right + 1] == 0
        if not window.any():
            break
        sides = {"top": window[0].mean(), "bottom": window[-1].mean(),
                 "left": window[:, 0].mean(), "right": window[:, -1].mean()}
        side = max(sides, key=sides.get)
        if side == "top":
            top += 1
        elif side == "bottom":
            bottom -= 1
        elif side == "left":
            left += 1
        else:
            right -= 1
    return pixels[top:bottom + 1, left:right + 1]


def stitch_panorama(photos: Sequence[SourcePhoto], crop: bool = True, scans: bool = False,
                    progress: Optional[Callable[[str], None]] = None) -> np.ndarray:
    """Stitch overlapping frames into one float32 RGB image (0..1). `scans`: flat subjects
    (documents, a mural) with an affine model instead of the spherical one."""
    import cv2

    if not 2 <= len(photos) <= MAX_INPUTS:
        raise MergeError(f"A panorama needs 2 to {MAX_INPUTS} overlapping photos.")
    if progress:
        progress("Matching and stitching the photos…")
    bgr = [cv2.cvtColor(_to8(photo.pixels), cv2.COLOR_RGB2BGR) for photo in photos]
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS if scans else cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(bgr)
    if status != cv2.Stitcher_OK:
        reasons = {cv2.Stitcher_ERR_NEED_MORE_IMGS: "the photos don't overlap enough (aim for about a third)",
                   cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "the overlapping parts couldn't be matched",
                   cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "the camera positions couldn't be worked out"}
        raise MergeError("The panorama couldn't be stitched: " + reasons.get(status, f"error {status}") + ".")
    result = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if crop:
        gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
        mask = cv2.erode((gray > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
        result = _crop_to_content(result, mask)
    return result


def encode_16bit_tiff(pixels: np.ndarray) -> bytes:
    """A 16-bit RGB TIFF of float32 0..1 pixels (the editor decodes it at full precision)."""
    import tifffile

    buffer = BytesIO()
    tifffile.imwrite(buffer, np.clip(pixels * 65535.0 + 0.5, 0, 65535).astype(np.uint16), photometric="rgb",
                     compression="zlib")
    return buffer.getvalue()


def preview_image(pixels: np.ndarray, long_edge: int = 1200) -> Image.Image:
    image = Image.fromarray(_to8(pixels))
    image.thumbnail((long_edge, long_edge))
    return image
