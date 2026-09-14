# Reliability and Security Workflow

## Image validation

Use `validate_image_bytes()` before processing upload content. It checks that the bytes are a real readable image rather than trusting the filename extension.

## Project validation

Use `validate_project_archive()` before loading `.photoproject` content. It checks:

- ZIP validity
- Maximum file count
- Maximum expanded size
- Path traversal attempts
- Required manifest
- Manifest version and structure

## Request reporting

Use `new_request_id()` for a stable identifier in logs and `image_digest()` to identify source content without storing the image itself.

## Operational rule

Keep Flask bound to localhost, enforce upload and pixel limits, validate content before decoding, and never extract project archive paths directly to disk without validation.
