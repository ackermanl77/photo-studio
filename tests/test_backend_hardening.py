import io
import json

from PIL import Image

from app.backend import app


def valid_image():
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def test_process_adds_request_id_header():
    response = app.test_client().post(
        "/process",
        headers={"X-Request-ID": "test-request-123"},
        data={"image": (io.BytesIO(valid_image()), "photo.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-123"


def test_process_rejects_invalid_image_bytes():
    response = app.test_client().post(
        "/process",
        data={"image": (io.BytesIO(b"not-an-image"), "photo.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Invalid image content" in response.get_json()["error"]
