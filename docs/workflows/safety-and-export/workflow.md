# Safety and Export Workflow

## Upload limits

The backend protects the local service with configurable limits:

- `PHOTO_MAX_UPLOAD_BYTES`, default 50 MB
- `PHOTO_MAX_IMAGE_PIXELS`, default 50,000,000 pixels

Set these before starting the app when working with a different hardware profile:

```powershell
$env:PHOTO_MAX_UPLOAD_BYTES = "104857600"
$env:PHOTO_MAX_IMAGE_PIXELS = "80000000"
```

## Export controls

The frontend supports:

- JPEG or PNG output
- JPEG quality selection
- Maximum width and height resizing
- Optional metadata stripping

Use maximum dimensions for web or social exports, and keep original dimensions for archival delivery.

## Operational checks

1. Keep source images outside output folders.
2. Review a full-size export after resizing.
3. Strip metadata when privacy requires it.
4. Keep the Flask service bound to localhost unless authentication is added.
5. Treat upload failures over the configured limit as expected protection, not processing errors.
