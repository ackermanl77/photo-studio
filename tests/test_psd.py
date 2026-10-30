"""Layered PSD import/export (F22): app.psd, the PSD composite as a photo, the Layers-tab import
and the layered export producer. The PSDs here are generated (a small builder below writes RLE/ZIP
channels, groups, masks, clipping and 16-bit files the way Photoshop lays them out); Pillow's own
PSD plugin is the independent reader for what app.psd writes."""
import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.pipeline import crop_rect, expected_output_size
from app.psd import PsdError, PsdLayer, read_psd, write_psd
from app.raw_import import SUPPORTED_EXTENSIONS, open_bytes


# --- a minimal PSD builder for the reader's tests --------------------------------------------------

def _packbits(row: bytes) -> bytes:
    out, pos = bytearray(), 0
    while pos < len(row):
        run = 1
        while pos + run < len(row) and run < 128 and row[pos + run] == row[pos]:
            run += 1
        if run > 2:
            out += bytes([257 - run, row[pos]])
            pos += run
        else:
            chunk = row[pos:pos + 128]
            out += bytes([len(chunk) - 1]) + chunk
            pos += len(chunk)
    return bytes(out)


def _channel(plane: np.ndarray, compression: int) -> bytes:
    height, width = plane.shape
    if compression == 0:
        return struct.pack(">H", 0) + plane.tobytes()
    if compression == 1:
        rows = [_packbits(plane[y].tobytes()) for y in range(height)]
        return struct.pack(">H", 1) + struct.pack(f">{height}H", *map(len, rows)) + b"".join(rows)
    if compression == 2:
        return struct.pack(">H", 2) + zlib.compress(plane.tobytes())
    deltas = np.diff(plane.astype(np.int32), axis=1, prepend=0).astype(plane.dtype)  # 3: ZIP with prediction
    return struct.pack(">H", 3) + zlib.compress(deltas.astype(plane.dtype.newbyteorder(">")).tobytes())


def _info(key: bytes, value: bytes) -> bytes:
    value += b"\0" * (len(value) % 2)
    return b"8BIM" + key + struct.pack(">I", len(value)) + value


def _layer(name, rect, rgba=None, compression=1, opacity=255, flags=8, blend=b"norm", section=None, mask=None, clipping=0, fill=None, depth=8):
    """One (record, channel data) pair. `rgba`: HxWx4 uint8/uint16 for the rect; `mask`: (rect, plane, default)."""
    top, left, bottom, right = rect
    channels = []
    if rgba is not None:
        for channel_id, index in ((-1, 3), (0, 0), (1, 1), (2, 2)):
            plane = np.ascontiguousarray(rgba[:, :, index])
            if depth == 16:
                plane = plane.astype(">u2")
            channels.append((channel_id, _channel(plane, compression)))
    mask_data = b""
    if mask is not None:
        (m_top, m_left, m_bottom, m_right), plane, default = mask
        mask_data = struct.pack(">iiiiBBxx", m_top, m_left, m_bottom, m_right, default, 0)
        channels.append((-2, _channel(np.ascontiguousarray(plane), compression)))
    if not channels:
        channels = [(channel_id, struct.pack(">H", 0)) for channel_id in (-1, 0, 1, 2)]
    encoded = name.encode("latin-1", "replace")
    pascal = bytes([len(encoded)]) + encoded
    pascal += b"\0" * ((-len(pascal)) % 4)
    extra = struct.pack(">I", len(mask_data)) + mask_data + struct.pack(">I", 0) + pascal
    extra += _info(b"luni", struct.pack(">I", len(name)) + name.encode("utf-16-be"))
    if section is not None:
        extra += _info(b"lsct", struct.pack(">I", section))
    if fill is not None:
        extra += _info(b"iOpa", bytes([fill, 0, 0, 0]))
    record = struct.pack(">iiiiH", top, left, bottom, right, len(channels))
    record += b"".join(struct.pack(">hI", channel_id, len(data)) for channel_id, data in channels)
    record += b"8BIM" + blend + struct.pack(">BBBB", opacity, clipping, flags, 0) + struct.pack(">I", len(extra)) + extra
    return record, b"".join(data for _id, data in channels)


def _document(width, height, layers, composite=None, depth=8, mode=3):
    records, data = zip(*layers) if layers else ((), ())
    info = struct.pack(">h", len(layers)) + b"".join(records) + b"".join(data)
    info += b"\0" * (len(info) % 2)
    section = struct.pack(">I", len(info)) + info + struct.pack(">I", 0)
    channels = 3 if mode == 3 else 1
    composite = np.zeros((height, width, channels), dtype=np.uint8) if composite is None else composite
    planes = b"".join(np.ascontiguousarray(composite[:, :, index]).astype(">u2" if depth == 16 else np.uint8).tobytes() for index in range(channels))
    header = b"8BPS" + struct.pack(">H6xHIIHH", 1, channels, height, width, depth, mode)
    return header + struct.pack(">I", 0) * 2 + struct.pack(">I", len(section)) + section + struct.pack(">H", 0) + planes


def _solid(height, width, rgba):
    return np.broadcast_to(np.array(rgba, dtype=np.uint8), (height, width, 4)).copy()


# --- writing ---------------------------------------------------------------------------------------

def test_written_psd_round_trips_and_pillow_reads_its_layers():
    background = Image.new("RGB", (64, 48), (200, 30, 30))
    blue = Image.new("RGBA", (20, 10), (0, 0, 255, 128))
    green = Image.new("RGBA", (64, 48), (0, 0, 0, 0))
    green.paste((0, 255, 0, 255), (30, 20, 50, 40))
    content = write_psd(background, [
        PsdLayer("Fondo ñ 日本", blue, 5, 6, 0.5, "multiply", True),
        PsdLayer("Top", green, 0, 0, 1.0, "screen", False),
    ])
    document = read_psd(content)
    assert (document.width, document.height) == (64, 48)
    low, top = document.layers
    assert (low.name, low.left, low.top, low.image.size, low.blend_mode, low.visible) == ("Fondo ñ 日本", 5, 6, (20, 10), "multiply", True)
    assert low.opacity == pytest.approx(0.5, abs=0.01) and low.image.getpixel((0, 0)) == (0, 0, 255, 128)
    # Written cropped to its visible pixels.
    assert (top.left, top.top, top.image.size, top.visible, top.blend_mode) == (30, 20, (20, 20), False, "screen")
    pillow = Image.open(io.BytesIO(content))
    assert pillow.size == (64, 48) and pillow.getpixel((0, 0)) == (200, 30, 30)
    assert [bbox for _name, _mode, bbox, _tiles in pillow.layers] == [(5, 6, 25, 16), (30, 20, 50, 40)]


def test_written_psd_preserves_user_masks_and_nested_groups():
    masked = Image.new("RGBA", (8, 6), (0, 255, 0, 255))
    mask = Image.new("L", masked.size, 255)
    mask.paste(0, (0, 0, 4, 6))
    content = write_psd(
        Image.new("RGB", masked.size, (20, 30, 40)),
        [PsdLayer("Masked", masked, group="Outer / Inner", mask=mask)],
    )
    document = read_psd(content)
    assert len(document.layers) == 1
    layer = document.layers[0]
    assert layer.name == "Masked" and layer.group == "Outer / Inner"
    alpha = np.asarray(layer.image.getchannel("A"))
    assert alpha[:, :4].max() == 0 and alpha[:, 4:].min() == 255


def test_every_blend_mode_survives_a_round_trip():
    from app.major_features import BLEND_MODE_LABELS

    layers = [PsdLayer(mode, Image.new("RGBA", (2, 2), (9, 9, 9, 255)), blend_mode=mode) for mode in BLEND_MODE_LABELS]
    assert [layer.blend_mode for layer in read_psd(write_psd(Image.new("RGB", (4, 4)), layers)).layers] == list(BLEND_MODE_LABELS)


# --- reading ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("compression", [0, 1, 2, 3])
def test_reader_decodes_every_channel_compression(compression):
    pixels = np.random.default_rng(compression).integers(0, 256, (7, 9, 4), dtype=np.uint8)
    pixels[:, :4] = (10, 20, 30, 255)  # a run for PackBits' repeat path
    document = read_psd(_document(12, 10, [_layer("L", (2, 1, 9, 10), pixels, compression)]))
    assert np.array_equal(np.asarray(document.layers[0].image), pixels)


def test_groups_masks_clipping_fill_and_empty_layers():
    layers = [
        _layer("Background", (0, 0, 20, 20), _solid(20, 20, (255, 255, 255, 255))),
        _layer("</Layer group>", (0, 0, 0, 0), section=3),  # bounding divider below the group's children
        _layer("in hidden group", (0, 0, 10, 10), _solid(10, 10, (255, 0, 0, 255))),
        _layer("Group", (0, 0, 0, 0), section=1, flags=8 | 2),  # the folder itself, hidden
        _layer("masked", (0, 0, 10, 10), _solid(10, 10, (0, 255, 0, 255)),
               mask=((0, 0, 10, 5), np.zeros((10, 5), dtype=np.uint8), 255)),  # left half masked out
        _layer("base", (10, 10, 20, 15), _solid(10, 5, (0, 0, 255, 255))),
        _layer("clipped", (10, 10, 20, 20), _solid(10, 10, (255, 255, 0, 255)), clipping=1),
        _layer("half fill", (0, 10, 5, 20), _solid(5, 10, (9, 9, 9, 255)), fill=128),
        _layer("adjustment", (0, 0, 0, 0)),
    ]
    document = read_psd(_document(20, 20, layers))
    names = [layer.name for layer in document.layers]
    assert names == ["Background", "in hidden group", "masked", "base", "clipped", "half fill"] and document.skipped == 1
    by_name = {layer.name: layer for layer in document.layers}
    assert not by_name["in hidden group"].visible and by_name["masked"].visible
    masked = np.asarray(by_name["masked"].image)[:, :, 3]
    assert masked[:, :5].max() == 0 and masked[:, 5:].min() == 255
    clipped = np.asarray(by_name["clipped"].image)[:, :, 3]
    assert clipped[:, :5].min() == 255 and clipped[:, 5:].max() == 0  # only over the base's 5 columns
    assert np.asarray(by_name["half fill"].image)[0, 0, 3] == 128


def test_16_bit_and_grayscale_documents():
    pixels16 = np.zeros((4, 4, 4), dtype=np.uint16)
    pixels16[..., 0], pixels16[..., 3] = 65535, 65535
    document = read_psd(_document(4, 4, [_layer("L", (0, 0, 4, 4), pixels16, 1, depth=16)], depth=16))
    assert document.layers[0].image.getpixel((0, 0)) == (255, 0, 0, 255)
    gray = np.full((3, 5, 1), 77, dtype=np.uint8)
    assert read_psd(_document(5, 3, [], composite=gray, mode=1)).composite.getpixel((0, 0)) == (77, 77, 77)


def test_unsupported_files_are_refused_cleanly():
    with pytest.raises(PsdError):
        read_psd(b"not a psd")
    unknown_version = bytearray(_document(2, 2, []))
    unknown_version[4:6] = b"\0\3"
    with pytest.raises(PsdError, match="version"):
        read_psd(bytes(unknown_version))
    base = _document(2, 2, [], composite=np.zeros((2, 2, 1), dtype=np.uint8), mode=1)
    for mode, name in ((2, "Indexed"), (0, "Bitmap"), (7, "Multichannel")):
        with pytest.raises(PsdError, match=name):
            read_psd(base[:12] + struct.pack(">HIIHH", 1, 2, 2, 8, mode) + base[26:])
    with pytest.raises(PsdError, match="32-bit"):
        read_psd(base[:12] + struct.pack(">HIIHH", 1, 2, 2, 32, 1) + base[26:])
    with pytest.raises(PsdError, match="truncated"):
        read_psd(_document(12, 10, [_layer("L", (0, 0, 9, 9), _solid(9, 9, (1, 2, 3, 4)), 0)])[:200])


# --- real files saved by Photoshop 2026 (tests/assets/psd/make_psd_fixtures.jsx) ------------------------

REAL_PSDS = Path(__file__).parent / "assets" / "psd"
# Each holds, bottom to top: Background (white), "Red box" (multiply, 50 %), "Inside group" in
# "My group" (blue), "Masked green" (its mask reveals the left half), a "Hello PSD" text layer
# (MyriadPro, 10 px, RGB 10/20/30) and a Hue/Saturation adjustment layer. Converting the mode
# made Photoshop drop the adjustment layer in the CMYK, Lab and grey copies.
REAL_FILES = {
    "rgb8_layers.psd": dict(adjustment=True, exact=True),
    "rgb16_layers.psd": dict(adjustment=True, exact=True),
    "rgb8_layers.psb": dict(adjustment=True, exact=True, large=True),
    "gray16_layers.psd": dict(adjustment=False, grey=True),
    "cmyk8_layers.psd": dict(adjustment=False),
    "lab8_layers.psd": dict(adjustment=False),
}


@pytest.mark.parametrize("name", sorted(REAL_FILES))
def test_real_photoshop_files_keep_their_layers_groups_text_and_colours(name):
    expected = REAL_FILES[name]
    document = read_psd((REAL_PSDS / name).read_bytes())
    assert (document.width, document.height) == (64, 48) and document.composite.size == (64, 48)
    assert document.large == expected.get("large", False)
    layers = {layer.name: layer for layer in document.layers}
    assert list(layers)[1:] == ["Red box", "Inside group", "Masked green", "Hello PSD"]  # [0]: the localized Background
    assert document.skipped_adjustments == (["Hue/Saturation"] if expected["adjustment"] else [])
    red, inside, green, text = layers["Red box"], layers["Inside group"], layers["Masked green"], layers["Hello PSD"]
    assert (red.left, red.top, red.image.size) == (4, 4, (26, 20))
    assert red.blend_mode == "multiply" and red.opacity == pytest.approx(0.5, abs=0.01)
    assert inside.group == "My group" and red.group == "" and (inside.left, inside.top) == (34, 4)
    # The mask reveals x 4..32 of the green layer: opaque on the left, transparent on the right.
    assert green.image.getpixel((5, 8))[3] == 255 and green.image.getpixel((40, 8))[3] == 0
    assert text.text["contents"] == "Hello PSD" and text.text["font"] == "MyriadPro-Regular"
    assert text.text["size_px"] == pytest.approx(10.0)
    red_pixel, blue_pixel = red.image.getpixel((10, 10))[:3], inside.image.getpixel((10, 8))[:3]
    if expected.get("exact"):
        assert red_pixel == (255, 0, 0) and blue_pixel == (0, 0, 255) and text.text["color"] == (10, 20, 30)
    elif expected.get("grey"):
        assert len(set(red_pixel)) == 1 and red_pixel[0] < 160 and len(set(text.text["color"])) == 1
    else:  # CMYK and Lab go through the plain formulas, so close to the RGB values, not equal
        assert red_pixel[0] > 230 and max(red_pixel[1:]) < 60 and blue_pixel[2] > 230 and max(blue_pixel[:2]) < 60
        assert max(text.text["color"]) < 40


def test_a_16_bit_photoshop_file_keeps_its_layers():
    """Photoshop stores a 16-bit document's layers in an "Lr16" block, not the layer info section:
    reading only the latter imported no layers at all."""
    document = read_psd((REAL_PSDS / "rgb16_layers.psd").read_bytes())
    assert len(document.layers) == 5


def test_a_psd_opens_as_its_composite_photo():
    assert ".psd" in SUPPORTED_EXTENSIONS
    composite = np.zeros((6, 8, 3), dtype=np.uint8)
    composite[..., 1] = 180
    image = open_bytes(_document(8, 6, [], composite=composite), "art.psd")
    assert image.size == (8, 6) and image.getpixel((3, 3)) == (0, 180, 0)


# --- the app side ----------------------------------------------------------------------------------

def test_crop_rect_matches_the_pipeline_crop():
    assert crop_rect((400, 300), {}) is None
    for options in ({"crop_box": [10, 20, 210, 170]}, {"aspect": "1:1"}, {"aspect": "16:9"}, {"aspect": "9:16"}):
        left, top, right, bottom = crop_rect((400, 300), options)
        assert (right - left, bottom - top) == expected_output_size((400, 300), options)
    assert crop_rect((400, 300), {"aspect": "1:1"}) == (50, 0, 350, 300)


def test_psd_layers_become_image_layers_on_the_canvas():
    from app.ui.layers import layer_dict_to_image, psd_layer_dicts

    layers = [
        _layer("Photo", (0, 0, 10, 20), _solid(10, 20, (50, 50, 50, 255))),
        _layer("Logo", (2, 4, 6, 8), _solid(4, 4, (255, 0, 0, 255)), opacity=128, blend=b"scrn", flags=8 | 2),
    ]
    content = _document(20, 10, layers)
    dicts, skipped, adjustments = psd_layer_dicts(content, skip_bottom=True)
    assert adjustments == []
    assert skipped == 0 and len(dicts) == 1
    logo = dicts[0]
    assert (logo["name"], logo["blend_mode"], logo["visible"], logo["stretch"]) == ("Logo", "screen", False, True)
    assert logo["opacity"] == pytest.approx(0.502, abs=0.01)
    on_photo = layer_dict_to_image(logo, (40, 20))  # stretched from the 20x10 document to a 40x20 frame
    assert on_photo.getpixel((12, 6))[3] == 255 and on_photo.getpixel((2, 2))[3] == 0
    assert len(psd_layer_dicts(content, skip_bottom=False)[0]) == 2


def test_a_real_psd_imports_text_as_an_editable_text_layer_and_names_groups():
    from app.ui.layers import layer_dict_to_image, psd_layer_dicts

    content = (REAL_PSDS / "rgb8_layers.psd").read_bytes()
    dicts, skipped, adjustments = psd_layer_dicts(content, skip_bottom=True, editable_text=True)
    assert adjustments == ["Hue/Saturation"] and skipped == 1
    by_name = {layer["name"]: layer for layer in dicts}
    assert "My group / Inside group" in by_name
    text = by_name["Hello PSD"]
    assert text["kind"] == "text" and text["text"] == "Hello PSD" and text["font"] == "MyriadPro-Regular"
    assert text["color"] == "#0A141E" and text["font_size_pct"] == pytest.approx(10 / 48 * 100, abs=0.01)
    # Centred where Photoshop drew it: the layer's pixels span x 6..47, y 8..16 of the 64x48 document.
    assert text["x_pct"] == pytest.approx((6 + 41 / 2) / 64 * 100, abs=0.01)
    assert text["y_pct"] == pytest.approx((8 + 8 / 2) / 48 * 100, abs=0.01)
    assert layer_dict_to_image(text, (64, 48)).getchannel("A").getbbox() is not None  # it renders
    rasterized, _skipped, _adjustments = psd_layer_dicts(content, skip_bottom=True, editable_text=False)
    assert {layer["kind"] for layer in rasterized} == {"image"}


def test_a_psb_opens_as_its_composite_photo():
    image = open_bytes((REAL_PSDS / "rgb8_layers.psb").read_bytes(), "art.psb")
    assert image.size == (64, 48) and ".psb" in SUPPORTED_EXTENSIONS


def test_layered_export_has_the_background_and_each_layer(monkeypatch):
    from app import backend_client
    from app.layer_workspace import WorkspaceLayer, layers_to_json
    from app.ui import processing

    calls = []

    def fake_export(content, filename, options, settings, local=None, layers_bytes=None, removal=None, **extra):
        calls.append(layers_bytes is not None)
        image = Image.new("RGB", (100, 50), (0, 0, 200) if layers_bytes else (10, 10, 10))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    star = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    star.paste((255, 255, 0, 255), (0, 0, 100, 100))
    payload = layers_to_json([
        WorkspaceLayer("Star", star, 0.75, "overlay", True, Image.new("L", star.size, 255),
                       group="Imported / Group")
    ]).encode("utf-8")
    monkeypatch.setattr(backend_client, "export_via_backend", fake_export)
    monkeypatch.setattr(processing, "_producer_layers", lambda *args: payload)
    content = processing.render_layered_psd(b"src", "photo.jpg", {"aspect": "2:1"}, Image.new("RGB", (200, 100)), None, None, "[]", None, (200, 100), {})
    assert calls == [False, True]  # the Background without the layers, then the composite with them
    document = read_psd(content)
    assert [layer.name for layer in document.layers] == ["Background", "Star"]
    background, star_layer = document.layers
    assert background.image.getpixel((0, 0)) == (10, 10, 10, 255)
    assert star_layer.blend_mode == "overlay" and star_layer.opacity == pytest.approx(0.75, abs=0.01)
    assert star_layer.group == "Imported / Group"
    # Rasterized at the 200x100 frame, then scaled to the 100x50 export: the left half.
    assert (star_layer.left, star_layer.top, star_layer.image.height) == (0, 0, 50)
    assert 50 <= star_layer.image.width <= 54  # plus the resampling filter's soft edge
    assert document.composite.getpixel((0, 0)) == (0, 0, 200)


@pytest.mark.slow
def test_the_layers_tab_imports_a_psd(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import backend_client
    from tests.test_frontend_unified_app import APP_PATH, _fake_process_via_backend, image_bytes, open_tab

    monkeypatch.setattr(backend_client, "ensure_backend_running", lambda: None)
    monkeypatch.setattr(backend_client, "process_via_backend", _fake_process_via_backend)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.file_uploader[0].set_value(("photo.png", image_bytes(size=(40, 20)), "image/png"))
    at.run(timeout=30)
    open_tab(at, "◈ Layers")
    at.radio(key="l-new-kind").set_value("PSD")
    at.run(timeout=30)
    psd = _document(40, 20, [
        _layer("Photo", (0, 0, 20, 40), _solid(20, 40, (50, 50, 50, 255))),
        _layer("Logo", (2, 4, 6, 8), _solid(4, 4, (255, 0, 0, 255)), blend=b"mul "),
    ])
    uploader = next(widget for widget in at.file_uploader if widget.key == "l-new-psd")
    uploader.set_value(("art.psd", psd, "image/vnd.adobe.photoshop"))
    at.run(timeout=30)
    at.run(timeout=30)
    assert not at.exception
    layers = at.session_state["image_layers"]
    assert [(layer["name"], layer["blend_mode"]) for layer in layers] == [("Logo", "multiply")]
    assert json.loads(json.dumps(layers))  # plain data: Undo, projects
