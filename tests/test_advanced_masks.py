import numpy as np
from PIL import Image

from app.advanced_masks import MaskLayer, MaskStack, color_mask, gradient_mask, luminance_mask, radial_mask, soften_mask


def test_mask_stack_combines_named_layers_and_toggle():
    first = Image.new("L", (10, 10), 80)
    second = Image.new("L", (10, 10), 200)
    stack = MaskStack((10, 10)).add(MaskLayer("shadows", first)).add(MaskLayer("subject", second, opacity=0.5))

    assert np.asarray(stack.combined()).max() == 100
    stack.set_enabled("subject", False)
    assert np.asarray(stack.combined()).max() == 80
    stack.remove("shadows")
    assert np.asarray(stack.combined()).max() == 0


def test_mask_generators_return_soft_grayscale_masks():
    image = Image.new("RGB", (20, 10), (120, 80, 40))
    masks = [gradient_mask(image.size), gradient_mask(image.size, "vertical", True), radial_mask(image.size), luminance_mask(image), color_mask(image, (120, 80, 40))]

    assert all(mask.mode == "L" and mask.size == image.size for mask in masks)
    assert soften_mask(masks[0], 2).size == image.size


def test_inverted_mask_layer_changes_selected_range():
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((0, 0), 255)
    layer = MaskLayer("selection", mask, invert=True)

    values = np.asarray(layer.normalized(mask.size))
    assert values[0, 0] == 0
    assert values[1, 1] == 255
