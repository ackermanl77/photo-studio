import numpy as np
from PIL import Image

from app.advanced_masks import MaskLayer, MaskStack, color_mask, gradient_mask, luminance_mask, magic_wand_mask, radial_mask, soften_mask


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


def test_magic_wand_mask_selects_only_the_clicked_contiguous_region():
    """Blue background with one red patch -- clicking inside the patch (the default,
    contiguous=True) must select it without bleeding into the unrelated blue background."""
    image = Image.new("RGB", (20, 10), (40, 40, 200))
    for x in range(8):
        for y in range(10):
            image.putpixel((x, y), (220, 40, 40))

    mask = np.asarray(magic_wand_mask(image, click_x=0.2, click_y=0.5, tolerance=40, contiguous=True))

    assert mask[5, 2] > 200  # inside the clicked red patch
    assert mask[5, 15] < 50  # blue background, well outside tolerance


def test_magic_wand_mask_contiguous_ignores_a_disconnected_matching_region():
    """Two separate red patches far apart on a blue background -- clicking one must not select
    the other in contiguous mode, but non-contiguous mode (a plain color-range select) should
    catch both, since flood fill is the only thing that makes them different."""
    image = Image.new("RGB", (30, 10), (40, 40, 200))
    for start in (2, 25):
        for dx in range(3):
            for y in range(10):
                image.putpixel((start + dx, y), (220, 40, 40))

    contiguous = np.asarray(magic_wand_mask(image, click_x=3 / 30, click_y=0.5, tolerance=40, contiguous=True))
    everywhere = np.asarray(magic_wand_mask(image, click_x=3 / 30, click_y=0.5, tolerance=40, contiguous=False))

    assert contiguous[5, 26] < 50  # the far, disconnected patch untouched
    assert everywhere[5, 26] > 200  # but selected once contiguity isn't required


def test_magic_wand_mask_tolerance_controls_how_similar_a_color_must_be():
    image = Image.new("RGB", (10, 1), (100, 100, 100))
    image.putpixel((5, 0), (140, 100, 100))  # a nearby-but-different color

    tight = np.asarray(magic_wand_mask(image, click_x=0.05, click_y=0.0, tolerance=10, contiguous=False))
    loose = np.asarray(magic_wand_mask(image, click_x=0.05, click_y=0.0, tolerance=80, contiguous=False))

    assert tight[0, 5] < 50
    assert loose[0, 5] > 200
