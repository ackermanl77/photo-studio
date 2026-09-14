import io
import json

from PIL import Image

from app.backend import app, backend_logger


def test_backend_process_emits_request_audit(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image, format="PNG")
    response = app.test_client().post(
        "/process",
        headers={"X-Request-ID": "audit-test"},
        data={"image": (io.BytesIO(image.getvalue()), "audit.png"), "options": json.dumps({})},
        content_type="multipart/form-data",
    )
    for handler in backend_logger.handlers:
        handler.flush()
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "audit-test"
