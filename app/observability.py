from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "photo-studio"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        return json.dumps(payload, sort_keys=True)


def configure_json_logger(name: str = LOGGER_NAME, path: str | Path | None = None) -> logging.Logger:
    """Attach a rotating JSON file handler to logger `name` (once). `path` defaults to
    logs/photo-studio.log under app.config.user_data_dir() -- the one per-user data folder, never a
    path relative to the working directory. Rotates at 2 MB, keeping 5 old files."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    if path is None:
        from app.config import user_data_dir

        path = user_data_dir() / "logs" / "photo-studio.log"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(destination, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# Every module logs through a child of LOGGER_NAME (get_logger("backend") -> "photo-studio.backend")
# and attaches no handler itself: importing a module must never open a log file (R2). Only an entry
# point -- the backend server, the desktop launcher, the CLI, the Streamlit app -- calls
# configure_logging() once, which writes that process's records to logs/<component>.log.
def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{component}")


def configure_logging(component: str, path: str | Path | None = None) -> logging.Logger:
    """Entry points only: route every "photo-studio.*" record of this process to a rotating JSON
    file, logs/<component>.log under app.config.user_data_dir() (one file per process kind, so the
    backend and the Streamlit process never rotate the same file under each other). Idempotent --
    a second call in the same process (a Streamlit rerun) keeps the first handler."""
    if path is None:
        from app.config import user_data_dir

        path = user_data_dir() / "logs" / f"{component}.log"
    return configure_json_logger(LOGGER_NAME, path)


@dataclass
class ProcessingAudit:
    request_id: str
    source_name: str
    operation: str
    status: str
    elapsed_ms: float
    cache_hit: bool = False
    error: str | None = None

    @classmethod
    def start(cls, request_id: str, source_name: str, operation: str) -> tuple["ProcessingAudit", float]:
        return cls(request_id, source_name, operation, "started", 0.0), time.perf_counter()

    def finish(self, started: float, status: str = "ok", error: str | None = None) -> "ProcessingAudit":
        self.status = status
        self.error = error
        self.elapsed_ms = (time.perf_counter() - started) * 1000
        return self

    def as_dict(self) -> dict:
        return asdict(self)


def log_audit(logger: logging.Logger, audit: ProcessingAudit) -> None:
    extra = {"event": "processing", "request_id": audit.request_id}
    logger.info(json.dumps(audit.as_dict(), sort_keys=True), extra=extra)
