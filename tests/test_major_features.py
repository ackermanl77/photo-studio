import sqlite3

import numpy as np
import pytest
from PIL import Image

from app.major_features import Layer, LayerStack, PhotoCatalog, heal_patch, preview_image


def test_layer_stack_composites_opacity_and_blend_modes():
    base = Image.new("RGB", (8, 8), (100, 100, 100))
    overlay = Image.new("RGBA", (8, 8), (200, 0, 0, 255))

    result = LayerStack(base).add(Layer("Color", overlay, opacity=0.5, blend_mode="normal")).composite()

    pixels = np.asarray(result)
    assert pixels.shape == (8, 8, 4)
    assert 140 <= int(pixels[4, 4, 0]) <= 160


def test_layer_masks_and_preview():
    base = Image.new("RGB", (20, 10), (100, 100, 100))
    overlay = Image.new("RGBA", (20, 10), (255, 0, 0, 255))
    mask = Image.new("L", (20, 10), 0)
    mask.putpixel((10, 5), 255)

    result = LayerStack(base).add(Layer("Local", overlay, mask=mask)).composite()

    assert result.size == (20, 10)
    assert preview_image(result, (10, 10)).size == (10, 5)


def test_healing_patch_preserves_dimensions():
    image = Image.new("RGB", (40, 20), (120, 120, 120))

    result = heal_patch(image, (0, 0, 10, 10), (20, 5, 30, 15))

    assert result.size == image.size


def test_photo_catalog_supports_rating_tags_and_search(tmp_path):
    catalog = PhotoCatalog(tmp_path / "photos.sqlite")
    catalog.upsert("portraits/one.jpg", rating=5, tags=["portrait", "favorite"])
    catalog.upsert("travel/two.jpg", rating=3, tags=["travel"])

    results = catalog.search("portrait", minimum_rating=4)

    assert results == [{"path": "portraits/one.jpg", "rating": 5, "tags": ["favorite", "portrait"]}]


def test_photo_catalog_get_returns_a_default_for_an_unseen_path(tmp_path):
    catalog = PhotoCatalog(tmp_path / "photos.sqlite")
    catalog.upsert("portraits/one.jpg", rating=4, tags=["favorite"])

    assert catalog.get("portraits/one.jpg") == {"path": "portraits/one.jpg", "rating": 4, "tags": ["favorite"]}
    assert catalog.get("never/scanned.jpg") == {"path": "never/scanned.jpg", "rating": 0, "tags": []}


def test_photo_catalog_closes_every_connection_it_opens(tmp_path, monkeypatch):
    """Regression test: PhotoCatalog._connect() used to return a bare sqlite3.Connection used via
    `with self._connect() as connection:` -- a Connection's own context manager only commits or
    rolls back on exit, it does NOT close the connection, so every upsert()/get()/search() call
    leaked an open connection/file handle. Wraps sqlite3.connect() to track every connection
    opened through it and asserts each one was actually closed by the time this returns."""
    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)

    catalog = PhotoCatalog(tmp_path / "photos.sqlite")
    catalog.upsert("portraits/one.jpg", rating=5, tags=["favorite"])
    catalog.get("portraits/one.jpg")
    catalog.search("portrait")

    assert len(opened) >= 4  # __init__'s CREATE TABLE + upsert + get + search, each its own connection
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")  # a closed connection raises on any further use
