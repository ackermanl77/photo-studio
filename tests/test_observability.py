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


def test_importing_the_app_modules_configures_no_logging(tmp_path):
    """R2: a module import must never attach a handler or create a log file -- only entry points
    (backend serve(), desktop, CLI, the Streamlit app) call configure_logging()."""
    import os
    import subprocess
    import sys

    code = (
        "import logging, app.backend, app.pipeline, app.cli, app.desktop;"
        "print(len(logging.getLogger('photo-studio').handlers), len(logging.getLogger('photo-studio.backend').handlers))"
    )
    env = {**os.environ, "PHOTO_STUDIO_DATA_DIR": str(tmp_path / "data")}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=os.path.dirname(os.path.dirname(__file__)), timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["0", "0"]
    assert not (tmp_path / "data" / "logs").exists()


def test_configure_logging_routes_backend_audits_to_a_per_component_file(tmp_path):
    from app.observability import LOGGER_NAME, configure_logging, get_logger

    root = logging.getLogger(LOGGER_NAME)
    saved = list(root.handlers)
    for handler in saved:
        root.removeHandler(handler)
    try:
        configure_logging("backend")
        configure_logging("backend")  # idempotent
        assert len(root.handlers) == 1
        audit, started = ProcessingAudit.start("req-3", "photo.jpg", "preview")
        log_audit(get_logger("backend"), audit.finish(started))
        root.handlers[0].flush()
        log_file = tmp_path / ".photo-studio" / "logs" / "backend.log"
        record = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
        assert record["request_id"] == "req-3" and record["logger"] == "photo-studio.backend"
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)
