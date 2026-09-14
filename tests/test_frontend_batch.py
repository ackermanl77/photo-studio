from pathlib import Path

from PIL import Image

from app.batch_release import process_files


def test_batch_frontend_backend_contract(tmp_path):
    first = tmp_path / "one.jpg"
    second = tmp_path / "two.jpg"
    Image.new("RGB", (10, 10), (20, 30, 40)).save(first)
    Image.new("RGB", (10, 10), (40, 30, 20)).save(second)

    outputs, report = process_files([first, second], profile="Web")

    assert len(outputs) == 2
    assert all(item.status == "ok" for item in report)
    assert all(Path(name).suffix == ".jpg" for name, _ in outputs)
