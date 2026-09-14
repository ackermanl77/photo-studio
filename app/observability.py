from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path


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


def configure_json_logger(name: str = "photo-studio", path: str | Path = ".photo-studio/photo-studio.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(destination, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


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
