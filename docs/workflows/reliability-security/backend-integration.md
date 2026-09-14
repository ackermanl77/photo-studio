# Backend Reliability Integration

The Flask processing endpoint now actively uses the reliability layer.

## Runtime protections

- Uploaded image bytes are validated before decoding.
- Invalid content returns HTTP 400 with a clear error.
- Every response receives an `X-Request-ID` header.
- Caller-provided request IDs are preserved for tracing.
- Generated request IDs are used when callers do not provide one.

## Testing

Use the request ID from an error response to correlate it with application logs when structured logging is added.
