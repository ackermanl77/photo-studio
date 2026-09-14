import numpy as np
from PIL import Image

from app.portrait_tools import detect_faces, skin_mask, smooth_skin


def test_portrait_tools_preserve_image_dimensions():
    image = Image.new("RGB", (160, 120), (150, 100, 80))

    mask = skin_mask(image)
    softened = smooth_skin(image, strength=0.3)
    faces = detect_faces(image)

    assert mask.mode == "L"
    assert mask.size == image.size
    assert softened.size == image.size
    assert isinstance(faces, list)


def test_skin_mask_is_grayscale_and_bounded():
    image = Image.new("RGB", (20, 20), (180, 120, 100))
    values = np.asarray(skin_mask(image))
    assert values.dtype == np.uint8
    assert values.min() >= 0 and values.max() <= 255
