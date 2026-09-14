# Preview Cache Integration Workflow

## Goal

Avoid reprocessing the same source image and recipe repeatedly during interactive editing.

## API

```python
from app.cached_processing import cached_preview

preview, report = cached_preview(
    source_bytes,
    source_name,
    options,
    process_full_image,
    ".photo-studio/previews",
)
```

## Behavior

- First request processes and stores a bounded preview.
- Same source bytes plus same options returns a cache hit.
- Changing the recipe creates a new cache key.
- Changing the source bytes creates a new cache key.
- Cache reports include elapsed time, dimensions, and hit status.
- Full-resolution delivery exports should bypass this cache.
