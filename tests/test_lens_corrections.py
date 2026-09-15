import numpy as np
from PIL import Image, ImageDraw

from app.lens_corrections import LENS_PROFILES, apply_lens_corrections, correct_chromatic_aberration, correct_distortion, correct_perspective, correct_vignette


def _grid_image(size=(80, 80)) -> Image.Image:
    image = Image.new("RGB", size, (30, 30, 30))
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 8):
        draw.line([(x, 0), (x, size[1])], fill=(220, 220, 220))
    for y in range(0, size[1], 8):
        draw.line([(0, y), (size[0], y)], fill=(220, 220, 220))
    return image


def test_correct_distortion_is_a_no_op_at_zero():
    image = _grid_image()
    result = correct_distortion(image, 0)
    assert result is image


def test_correct_distortion_changes_edges_more_than_the_center():
    """A radial (k1) correction should barely touch the pixel nearest the optical center (r ~ 0)
    but noticeably move pixels near the corners (r large) -- true regardless of the sign convention
    chosen for the "amount" slider."""
    image = _grid_image((100, 100))
    corrected = correct_distortion(image, 80)
    before = np.asarray(image, dtype=np.float32)
    after = np.asarray(corrected, dtype=np.float32)

    assert corrected.size == image.size
    center_change = np.abs(after[48:52, 48:52] - before[48:52, 48:52]).mean()
    corner_change = np.abs(after[:10, :10] - before[:10, :10]).mean()
    assert corner_change > center_change


def test_correct_vignette_is_a_no_op_at_zero():
    image = Image.new("RGB", (40, 40), (150, 150, 150))
    assert correct_vignette(image, 0) is image


def test_correct_vignette_darkens_corners_for_negative_amount():
    image = Image.new("RGB", (60, 60), (200, 200, 200))
    result = correct_vignette(image, -80, midpoint=30)
    pixels = np.asarray(result, dtype=np.int32)
    center = pixels[28:32, 28:32].mean()
    corner = pixels[:6, :6].mean()
    assert corner < center - 20


def test_correct_vignette_brightens_corners_for_positive_amount():
    image = Image.new("RGB", (60, 60), (120, 120, 120))
    result = correct_vignette(image, 80, midpoint=30)
    pixels = np.asarray(result, dtype=np.int32)
    center = pixels[28:32, 28:32].mean()
    corner = pixels[:6, :6].mean()
    assert corner > center + 5


def test_correct_chromatic_aberration_is_a_no_op_at_zero():
    image = Image.new("RGB", (30, 30), (10, 200, 10))
    assert correct_chromatic_aberration(image, 0) is image


def test_correct_chromatic_aberration_leaves_the_green_channel_untouched():
    """Green is the fixed reference channel -- only Red and Blue should move."""
    image = _grid_image((50, 50))
    result = correct_chromatic_aberration(image, 100)
    before = np.asarray(image)
    after = np.asarray(result)
    assert np.array_equal(before[..., 1], after[..., 1])
    assert not np.array_equal(before[..., 0], after[..., 0])
    assert not np.array_equal(before[..., 2], after[..., 2])


def test_apply_lens_corrections_combines_all_three_and_skips_zero_terms():
    image = _grid_image((60, 60))
    result = apply_lens_corrections(image, distortion=0, vignette_amount=-50, vignette_midpoint=40, chromatic_aberration=0)
    only_vignette = correct_vignette(image, -50, 40)
    assert np.array_equal(np.asarray(result), np.asarray(only_vignette))

    all_zero = apply_lens_corrections(image, 0, 0, 50, 0)
    assert all_zero is image


def test_lens_profiles_have_the_expected_four_fields_each():
    for name, values in LENS_PROFILES.items():
        assert set(values) == {"distortion", "vignette_amount", "vignette_midpoint", "chromatic_aberration"}
    assert LENS_PROFILES["None"] == {"distortion": 0, "vignette_amount": 0, "vignette_midpoint": 50, "chromatic_aberration": 0}


def test_correct_perspective_is_a_no_op_at_zero():
    image = _grid_image()
    assert correct_perspective(image, 0, 0) is image


def test_correct_perspective_positive_vertical_narrows_the_top_more_than_the_bottom():
    """Empirically, a homography that only moves 2 of a quadrilateral's 4 corners (as
    correct_perspective's vertical>0 case does, keeping the bottom two fixed) still nudges the
    "fixed" edge a little too -- projective transforms don't leave a line pointwise-fixed just
    because both its endpoints are fixed. So this checks the intended *asymmetry* (top changes
    more than bottom) rather than asserting the bottom is untouched."""
    image = _grid_image((100, 100))
    result = correct_perspective(image, vertical=60, horizontal=0)
    before = np.asarray(image, dtype=np.float32)
    after = np.asarray(result, dtype=np.float32)
    top_change = np.abs(after[0:3, :] - before[0:3, :]).mean()
    bottom_change = np.abs(after[-3:, :] - before[-3:, :]).mean()
    assert top_change > bottom_change


def test_correct_perspective_negative_vertical_narrows_the_bottom_more_than_the_top():
    image = _grid_image((100, 100))
    result = correct_perspective(image, vertical=-60, horizontal=0)
    before = np.asarray(image, dtype=np.float32)
    after = np.asarray(result, dtype=np.float32)
    top_change = np.abs(after[0:3, :] - before[0:3, :]).mean()
    bottom_change = np.abs(after[-3:, :] - before[-3:, :]).mean()
    assert bottom_change > top_change


def test_correct_perspective_positive_horizontal_narrows_the_left_more_than_the_right():
    image = _grid_image((100, 100))
    result = correct_perspective(image, vertical=0, horizontal=60)
    before = np.asarray(image, dtype=np.float32)
    after = np.asarray(result, dtype=np.float32)
    left_change = np.abs(after[:, 0:3] - before[:, 0:3]).mean()
    right_change = np.abs(after[:, -3:] - before[:, -3:]).mean()
    assert left_change > right_change


def test_correct_perspective_negative_horizontal_narrows_the_right_more_than_the_left():
    image = _grid_image((100, 100))
    result = correct_perspective(image, vertical=0, horizontal=-60)
    before = np.asarray(image, dtype=np.float32)
    after = np.asarray(result, dtype=np.float32)
    left_change = np.abs(after[:, 0:3] - before[:, 0:3]).mean()
    right_change = np.abs(after[:, -3:] - before[:, -3:]).mean()
    assert right_change > left_change


def test_correct_perspective_combines_vertical_and_horizontal():
    image = _grid_image((100, 100))
    combined = correct_perspective(image, vertical=50, horizontal=50)
    only_vertical = correct_perspective(image, vertical=50, horizontal=0)
    assert not np.array_equal(np.asarray(combined), np.asarray(only_vertical))


def test_apply_lens_corrections_includes_perspective():
    image = _grid_image((60, 60))
    result = apply_lens_corrections(image, perspective_vertical=50)
    only_perspective = correct_perspective(image, 50, 0)
    assert np.array_equal(np.asarray(result), np.asarray(only_perspective))
