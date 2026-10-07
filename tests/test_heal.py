"""F13: Repair's "Heal" method (app.heal) -- patch-based spot healing with Poisson blending."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.heal import heal
from app.pipeline import run_pipeline


def _textured(seed: int = 0, size=(240, 320)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = size
    noise = rng.normal(128, 20, (height, width, 3)) + np.linspace(0, 60, width)[None, :, None]
    return np.clip(noise, 0, 255).astype(np.uint8)


def _spot(image: np.ndarray, center=(160, 120), radius=12):
    spotted = image.copy()
    cv2.circle(spotted, center, radius, (255, 0, 0), -1)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, center, radius + 2, 255, -1)
    return spotted, mask


def test_heal_removes_the_spot_and_keeps_the_texture():
    base = _textured()
    spotted, mask = _spot(base)
    healed = heal(spotted, mask)
    inpainted = cv2.inpaint(spotted, mask, 3, cv2.INPAINT_TELEA)
    selected = mask > 0
    assert np.abs(healed[selected].astype(int).mean(axis=0) - base[selected].mean(axis=0)).max() < 12  # the red is gone
    # Texture comes back (inpaint smooths it away).
    assert healed[selected].std() > 0.8 * base[selected].std() > inpainted[selected].std()


def test_heal_leaves_everything_away_from_the_spot_alone():
    spotted, mask = _spot(_textured(1))
    healed = heal(spotted, mask)
    far = cv2.dilate(mask, np.ones((15, 15), np.uint8)) == 0
    assert np.array_equal(healed[far], spotted[far])


def test_heal_falls_back_to_inpaint_at_the_border_and_for_huge_regions():
    base = _textured(2)
    edge_mask = np.zeros(base.shape[:2], np.uint8)
    edge_mask[:20, :20] = 255
    huge_mask = np.zeros(base.shape[:2], np.uint8)
    huge_mask[20:200, 20:300] = 255
    for mask in (edge_mask, huge_mask):
        out = heal(base, mask)
        assert out.shape == base.shape and not np.array_equal(out[mask > 0], base[mask > 0])


def test_heal_with_an_empty_mask_is_a_no_op():
    base = _textured(3)
    assert heal(base, np.zeros(base.shape[:2], np.uint8)) is base


def test_the_pipeline_uses_heal_when_asked():
    spotted, mask = _spot(_textured(4))
    image, mask_image = Image.fromarray(spotted), Image.fromarray(mask)
    fill = np.asarray(run_pipeline(image, {"removal_method": "Fill"}, removal_mask=mask_image))
    healed = np.asarray(run_pipeline(image, {"removal_method": "Heal"}, removal_mask=mask_image))
    selected = mask > 0
    assert healed[selected].std() > fill[selected].std()
