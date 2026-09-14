# Observability and Audit Workflow

## Goal

Trace processing behavior without storing image pixels or sensitive image content in logs.

## Events

`ProcessingAudit` records:

- Request ID
- Source filename
- Operation name
- Status
- Elapsed milliseconds
- Cache-hit status
- Optional error text

## Logging

```python
from app.observability import configure_json_logger, log_audit

logger = configure_json_logger()
log_audit(logger, audit)
```

Logs are JSON lines and default to `.photo-studio/photo-studio.log`.

## Privacy rule

Do not log image bytes, full source paths when they contain sensitive information, or image metadata unless explicitly needed. Use request IDs and source names for correlation.
