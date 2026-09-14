import json
import logging

from app.observability import JsonFormatter, ProcessingAudit, configure_json_logger, log_audit


def test_processing_audit_records_completion():
    audit, started = ProcessingAudit.start("req-1", "photo.jpg", "preview")
    audit.finish(started)
    payload = audit.as_dict()
    assert payload["request_id"] == "req-1"
    assert payload["status"] == "ok"
    assert payload["elapsed_ms"] >= 0


def test_json_logger_writes_request_event(tmp_path):
    logger = configure_json_logger("test-observability", tmp_path / "events.log")
    audit, started = ProcessingAudit.start("req-2", "photo.jpg", "export")
    log_audit(logger, audit.finish(started))
    for handler in logger.handlers:
        handler.flush()
    line = (tmp_path / "events.log").read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(line)
    assert record["request_id"] == "req-2"
    assert record["event"] == "processing"
