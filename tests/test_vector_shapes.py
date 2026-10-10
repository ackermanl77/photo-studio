import numpy as np

from app.vector_shapes import SHAPE_TYPES, path_outline_points, render_path_layer, render_shape_layer


def _point(x_pct, y_pct, handle_dx_pct=0.0, handle_dy_pct=0.0):
    return {"x_pct": x_pct, "y_pct": y_pct, "handle_dx_pct": handle_dx_pct, "handle_dy_pct": handle_dy_pct}


def test_rectangle_fills_its_bounding_box_and_leaves_the_rest_transparent():
    image = render_shape_layer((100, 100), {
        "shape_type": "Rectangle", "x_pct": 50, "y_pct": 50, "width_pct": 40, "height_pct": 40,
        "fill_enabled": True, "fill_color": (255, 0, 0), "fill_opacity": 1.0,
    })
    pixels = np.asarray(image)

    assert tuple(pixels[50, 50]) == (255, 0, 0, 255)  # center: filled
    assert pixels[5, 5, 3] == 0  # far corner: outside the 40%-wide box, transparent


def test_ellipse_fills_a_round_region():
    image = render_shape_layer((100, 100), {
        "shape_type": "Ellipse", "x_pct": 50, "y_pct": 50, "width_pct": 60, "height_pct": 60,
        "fill_enabled": True, "fill_color": (0, 255, 0), "fill_opacity": 1.0,
    })
    pixels = np.asarray(image)

    assert tuple(pixels[50, 50]) == (0, 255, 0, 255)  # center: filled
    assert pixels[2, 2, 3] == 0  # corner: outside any circle inscribed in the canvas


def test_line_draws_a_horizontal_bar_of_the_requested_thickness():
    image = render_shape_layer((100, 40), {
        "shape_type": "Line", "x_pct": 50, "y_pct": 50, "width_pct": 80, "rotation_deg": 0,
        "fill_color": (0, 0, 255), "fill_opacity": 1.0, "line_thickness": 6,
    })
    pixels = np.asarray(image)

    assert tuple(pixels[20, 50]) == (0, 0, 255, 255)  # on the line's own row, mid-length
    assert pixels[35, 50, 3] == 0  # far above the line: transparent


def test_polygon_and_star_are_registered_shape_types_and_render_without_error():
    for shape_type in SHAPE_TYPES:
        image = render_shape_layer((80, 80), {"shape_type": shape_type, "x_pct": 50, "y_pct": 50, "width_pct": 50, "height_pct": 50})
        assert image.size == (80, 80)
        assert image.mode == "RGBA"


def test_star_inner_ratio_pinches_the_shape_between_points():
    """A smaller inner_ratio pulls the star's inner vertices closer to the center, so a point
    roughly halfway out along an inward-pointing direction should be covered by a generous inner
    ratio but not by a sharply pinched one."""
    wide_waist = render_shape_layer((100, 100), {
        "shape_type": "Star", "x_pct": 50, "y_pct": 50, "width_pct": 80, "height_pct": 80,
        "sides": 5, "inner_ratio": 0.9, "rotation_deg": 0, "fill_color": (255, 255, 255), "fill_opacity": 1.0,
    })
    narrow_waist = render_shape_layer((100, 100), {
        "shape_type": "Star", "x_pct": 50, "y_pct": 50, "width_pct": 80, "height_pct": 80,
        "sides": 5, "inner_ratio": 0.1, "rotation_deg": 0, "fill_color": (255, 255, 255), "fill_opacity": 1.0,
    })

    assert np.asarray(wide_waist)[:, :, 3].sum() > np.asarray(narrow_waist)[:, :, 3].sum()


def test_rotation_moves_a_rectangles_corners():
    unrotated = render_shape_layer((100, 100), {
        "shape_type": "Rectangle", "x_pct": 50, "y_pct": 50, "width_pct": 20, "height_pct": 60,
        "rotation_deg": 0, "fill_color": (255, 255, 255), "fill_opacity": 1.0,
    })
    rotated = render_shape_layer((100, 100), {
        "shape_type": "Rectangle", "x_pct": 50, "y_pct": 50, "width_pct": 20, "height_pct": 60,
        "rotation_deg": 90, "fill_color": (255, 255, 255), "fill_opacity": 1.0,
    })

    assert not np.array_equal(np.asarray(unrotated), np.asarray(rotated))
    # A 90-degree rotation of a tall-narrow rectangle becomes short-wide -- its own alpha mask
    # should now resemble the UNROTATED shape's mask transposed.
    assert np.array_equal(np.asarray(unrotated)[:, :, 3], np.asarray(rotated)[:, :, 3].T)


def test_stroke_adds_an_outline_color_distinct_from_the_fill():
    image = render_shape_layer((100, 100), {
        "shape_type": "Rectangle", "x_pct": 50, "y_pct": 50, "width_pct": 40, "height_pct": 40,
        "fill_enabled": True, "fill_color": (255, 0, 0), "fill_opacity": 1.0,
        "stroke_enabled": True, "stroke_color": (0, 0, 255), "stroke_width": 3,
    })
    pixels = np.asarray(image)

    assert tuple(pixels[50, 50]) == (255, 0, 0, 255)  # interior: fill color
    edge_row = pixels[30, 30:70]  # the rectangle's top edge (y=30 for a box spanning 30..70)
    assert any(tuple(pixel) == (0, 0, 255, 255) for pixel in edge_row)  # stroke color appears somewhere on the edge


def test_disabled_fill_with_stroke_leaves_the_interior_transparent():
    image = render_shape_layer((100, 100), {
        "shape_type": "Rectangle", "x_pct": 50, "y_pct": 50, "width_pct": 40, "height_pct": 40,
        "fill_enabled": False, "stroke_enabled": True, "stroke_color": (0, 0, 255), "stroke_width": 3,
    })
    pixels = np.asarray(image)

    assert pixels[50, 50, 3] == 0  # interior: no fill, so transparent
    edge_row = pixels[30, 30:70]
    assert any(tuple(pixel) == (0, 0, 255, 255) for pixel in edge_row)  # but the stroke still shows


def test_path_outline_with_zero_handles_is_a_straight_polyline():
    points = [_point(10, 50), _point(50, 50), _point(90, 50)]

    outline = path_outline_points(points, (100, 100), closed=False)

    # Every sampled point along a zero-handle (corner-only) path must land exactly on y=50 --
    # a straight horizontal line from x=10 to x=90, since a zero-length handle degenerates the
    # cubic Bezier segment to a straight line between its two anchors.
    assert all(abs(y - 50.0) < 1e-6 for _x, y in outline)
    assert outline[0] == (10.0, 50.0)
    assert outline[-1] == (90.0, 50.0)


def test_path_outline_with_a_handle_bows_away_from_the_straight_line():
    # Only the first point gets a handle -- its OUTGOING control point (p0 + handle) pulls that
    # end of the curve upward (smaller y); the second point stays a plain corner (zero handle), so
    # there's only one handle's effect to reason about, not two interacting ones.
    straight = path_outline_points([_point(10, 50), _point(90, 50)], (100, 100), closed=False)
    curved = path_outline_points([_point(10, 50, handle_dx_pct=20, handle_dy_pct=-30), _point(90, 50)], (100, 100), closed=False)

    straight_mid_y = straight[len(straight) // 2][1]
    curved_mid_y = curved[len(curved) // 2][1]
    assert abs(straight_mid_y - 50.0) < 1e-6  # unchanged: still a flat line
    assert curved_mid_y < 50.0  # pulled upward (smaller y) by the first point's handle


def test_path_outline_closed_connects_the_last_point_back_to_the_first():
    points = [_point(20, 20), _point(80, 20), _point(50, 80)]

    open_outline = path_outline_points(points, (100, 100), closed=False)
    closed_outline = path_outline_points(points, (100, 100), closed=True)

    assert len(closed_outline) > len(open_outline)  # one more Bezier segment sampled
    # The closed outline's own last sampled point must land back on/near the first anchor.
    assert abs(closed_outline[-1][0] - 20.0) < 1e-6
    assert abs(closed_outline[-1][1] - 20.0) < 1e-6


def test_render_path_layer_with_fewer_than_two_points_is_a_transparent_no_op():
    image = render_path_layer((50, 50), {"points": [_point(50, 50)], "closed": False})
    assert np.asarray(image)[:, :, 3].max() == 0

    image_empty = render_path_layer((50, 50), {"points": [], "closed": False})
    assert np.asarray(image_empty)[:, :, 3].max() == 0


def test_render_path_layer_open_path_never_fills_even_if_fill_is_requested():
    image = render_path_layer((100, 100), {
        "points": [_point(20, 50), _point(50, 20), _point(80, 50)], "closed": False,
        "fill_enabled": True, "fill_color": (255, 0, 0), "fill_opacity": 1.0,
        "stroke_enabled": True, "stroke_color": (0, 255, 0), "stroke_width": 3,
    })
    pixels = np.asarray(image)

    # A point that would be "inside" the triangle-like curve if it were closed and filled --
    # an open path has no well-defined interior, so this must stay transparent (or at most catch
    # the stroke itself, never the requested red fill).
    assert tuple(pixels[45, 50][:3]) != (255, 0, 0)


def test_render_path_layer_closed_path_fills_its_enclosed_area():
    image = render_path_layer((100, 100), {
        "points": [_point(20, 20), _point(80, 20), _point(80, 80), _point(20, 80)], "closed": True,
        "fill_enabled": True, "fill_color": (0, 200, 0), "fill_opacity": 1.0,
        "stroke_enabled": False,
    })
    pixels = np.asarray(image)

    assert tuple(pixels[50, 50]) == (0, 200, 0, 255)  # well inside the closed square path
    assert pixels[5, 5, 3] == 0  # well outside it


def test_a_rotated_ellipse_turns_about_its_center():
    params = {"shape_type": "Ellipse", "x_pct": 50, "y_pct": 50, "width_pct": 80, "height_pct": 20,
              "fill_enabled": True, "fill_color": (255, 0, 0), "fill_opacity": 1.0}
    flat = np.asarray(render_shape_layer((100, 100), params))
    turned = np.asarray(render_shape_layer((100, 100), {**params, "rotation_deg": 90}))

    assert flat[50, 15, 3] == 255 and flat[15, 50, 3] == 0  # wide: left end filled, top empty
    assert turned[15, 50, 3] == 255 and turned[50, 15, 3] == 0  # tall after a quarter turn
    assert abs(int(flat[..., 3].astype(bool).sum()) - int(turned[..., 3].astype(bool).sum())) < 60  # same area


def test_an_unrotated_ellipse_is_unchanged_by_a_half_turn():
    params = {"shape_type": "Ellipse", "x_pct": 40, "y_pct": 50, "width_pct": 50, "height_pct": 30,
              "fill_enabled": True, "fill_color": (0, 0, 255), "fill_opacity": 1.0}
    assert np.array_equal(np.asarray(render_shape_layer((90, 70), params)),
                          np.asarray(render_shape_layer((90, 70), {**params, "rotation_deg": 180})))
