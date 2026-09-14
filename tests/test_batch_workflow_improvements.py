from pathlib import Path

from PIL import Image

from app.batch_release import process_files, report_csv


def test_batch_progress_and_csv_report(tmp_path):
    first = tmp_path / "one.jpg"
    second = tmp_path / "two.jpg"
    Image.new("RGB", (10, 10), (20, 30, 40)).save(first)
    Image.new("RGB", (10, 10), (40, 30, 20)).save(second)
    progress = []

    outputs, report = process_files([first, second], progress=lambda index, total, item: progress.append((index, total, item.status)))

    assert len(outputs) == 2
    assert progress[-1][:2] == (2, 2)
    assert b"source,output,status,error" in report_csv(report)


def test_batch_filename_collisions_are_disambiguated(tmp_path):
    one = tmp_path / "one.jpg"
    nested = tmp_path / "nested"
    nested.mkdir()
    two = nested / "one.jpg"
    Image.new("RGB", (10, 10), (20, 30, 40)).save(one)
    Image.new("RGB", (10, 10), (40, 30, 20)).save(two)

    outputs, report = process_files([one, two])

    assert len(outputs) == 2
    assert len({name for name, _ in outputs}) == 2
    assert all(item.status == "ok" for item in report)
