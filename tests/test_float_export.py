"""EXR/PFM float exports (F24): app.float_export, encode_export/write_export and the CLI."""
import struct
import sys
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image

from app import high_bit
from app.export import FLOAT_FORMATS, encode_export, validate_export_settings, write_export
from app.float_export import CHROMATICITIES, EXR_FLOAT, EXR_NO_COMPRESSION, encode_exr, encode_pfm, srgb_to_linear


def _read_exr(content: bytes) -> tuple[dict, np.ndarray]:
    """A minimal reader of the subset encode_exr writes (scanline, HALF/FLOAT, NO/ZIP)."""
    magic, version = struct.unpack_from("<ii", content, 0)
    assert magic == 20000630 and version == 2
    pos, attrs = 8, {}
    while content[pos] != 0:
        name_end = content.index(b"\0", pos)
        kind_end = content.index(b"\0", name_end + 1)
        (size,) = struct.unpack_from("<i", content, kind_end + 1)
        attrs[content[pos:name_end].decode()] = (content[name_end + 1:kind_end].decode(), content[kind_end + 5:kind_end + 5 + size])
        pos = kind_end + 5 + size
    pos += 1
    xmin, ymin, xmax, ymax = struct.unpack("<4i", attrs["dataWindow"][1])
    width, height = xmax - xmin + 1, ymax - ymin + 1
    compression = attrs["compression"][1][0]
    chlist = attrs["channels"][1]
    names, types, cursor = [], [], 0
    while chlist[cursor] != 0:
        end = chlist.index(b"\0", cursor)
        names.append(chlist[cursor:end].decode())
        types.append(struct.unpack_from("<i", chlist, end + 1)[0])
        cursor = end + 17
    dtype = np.dtype("<f2") if types[0] == 1 else np.dtype("<f4")
    lines = 16 if compression == 3 else 1
    blocks = (height + lines - 1) // lines
    offsets = struct.unpack_from(f"<{blocks}Q", content, pos)
    rows = []
    for offset in offsets:
        y, size = struct.unpack_from("<ii", content, offset)
        data = content[offset + 8:offset + 8 + size]
        count = min(lines, height - y)
        expected = count * width * len(names) * dtype.itemsize
        if compression == 3 and size < expected:
            predicted = np.frombuffer(zlib.decompress(data), dtype=np.uint8).astype(np.int32)
            undone = (np.cumsum(predicted - 128 * (np.arange(predicted.size) > 0)) & 0xFF).astype(np.uint8)
            half = (undone.size + 1) // 2
            raw = np.empty_like(undone)
            raw[0::2], raw[1::2] = undone[:half], undone[half:]
            data = raw.tobytes()
        rows.append(np.frombuffer(data, dtype=dtype).reshape(count, len(names), width))
    planar = np.concatenate(rows).astype(np.float32)  # H x C x W, channels in list order
    image = {name: planar[:, index, :] for index, name in enumerate(names)}
    return attrs, np.dstack([image["R"], image["G"], image["B"]])


def _gradient(height=37, width=50) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.dstack([x / (width - 1), y / (height - 1), (x + y) / (width + height - 2)])


def test_exr_round_trips_through_zip_and_raw_blocks():
    linear = _gradient()
    attrs, decoded = _read_exr(encode_exr(linear))
    assert attrs["compression"][1] == b"\x03"
    np.testing.assert_allclose(decoded, linear, atol=1e-3)  # half precision
    _attrs, decoded = _read_exr(encode_exr(linear, pixel_type=EXR_FLOAT, compression=EXR_NO_COMPRESSION))
    np.testing.assert_array_equal(decoded, linear)
    noise = np.random.default_rng(1).random((20, 33, 3), dtype=np.float32)  # incompressible -> raw blocks
    _attrs, decoded = _read_exr(encode_exr(noise))
    np.testing.assert_allclose(decoded, noise, atol=1e-3)


def test_exr_records_its_primaries():
    attrs, _ = _read_exr(encode_exr(_gradient(4, 4), "Display P3"))
    assert attrs["chromaticities"][0] == "chromaticities"
    np.testing.assert_allclose(struct.unpack("<8f", attrs["chromaticities"][1]), CHROMATICITIES["Display P3"], atol=1e-6)


def test_pfm_matches_opencv_decoder():
    linear = _gradient()
    decoded = cv2.imdecode(np.frombuffer(encode_pfm(linear), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(decoded[:, :, ::-1], linear)  # OpenCV returns BGR


def test_encode_export_writes_linear_float_from_the_attached_pixels():
    encoded = _gradient(30, 40)
    image = high_bit.attach(Image.fromarray(high_bit.quantize8(encoded), mode="RGB"), encoded)
    for format_name in FLOAT_FORMATS:
        assert validate_export_settings({"format": format_name})["bit_depth"] == 16
    content, extension, mime = encode_export(image, {"format": "PFM"})
    assert (extension, mime) == (".pfm", "image/x-portable-floatmap")
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    np.testing.assert_allclose(decoded, srgb_to_linear(encoded), atol=1e-6)  # the float pixels, not the 8-bit image
    content, extension, _mime = encode_export(image, {"format": "EXR", "long_edge": 20})
    _attrs, decoded = _read_exr(content)
    assert extension == ".exr" and decoded.shape == (15, 20, 3)
    assert 0.0 <= decoded.min() and decoded.max() <= 1.0


def test_write_export_and_cli_pick_float_by_extension(tmp_path, monkeypatch):
    source = tmp_path / "in.png"
    Image.fromarray(high_bit.quantize8(_gradient(24, 32)), mode="RGB").save(source)
    path = write_export(Image.open(source), tmp_path / "out.exr", {"bit_depth": 8})
    assert path.read_bytes()[:4] == struct.pack("<i", 20000630)
    from app import cli

    output = tmp_path / "cli.pfm"
    monkeypatch.setattr(sys, "argv", ["photo", "--input", str(source), "--output", str(output), "--exposure-ev", "0.5"])
    cli.main()
    decoded = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert decoded.dtype == np.float32 and decoded.shape == (24, 32, 3)


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.zeros((4, 4, 4))])
def test_float_encoders_reject_non_rgb(bad):
    with pytest.raises(ValueError):
        encode_exr(bad)
    with pytest.raises(ValueError):
        encode_pfm(bad)
