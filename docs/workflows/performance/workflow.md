# Performance and Reliability Workflow

## Goal

Keep interactive previews responsive on CPU-only systems and make repeated processing observable.

## Preview cache

Use `PreviewCache` with source bytes and an options dictionary. The cache key combines both, so changing either the source or recipe produces a new preview.

```python
from app.performance import PreviewCache

cache = PreviewCache(".photo-studio/previews", max_items=100)
key = cache.key(source_bytes, options)
preview = cache.get(key)
if preview is None:
    preview = process_image()
    cache.put(key, preview)
```

## Reliability rules

- Cache only bounded JPEG previews, never original files.
- Keep a maximum item count so cache growth is bounded.
- Delete unreadable cache entries automatically.
- Clear the cache when changing processing semantics.
- Use `timed_report()` to record cache hits, processing time, and output dimensions.

The full-resolution image should still be generated only for final export.
