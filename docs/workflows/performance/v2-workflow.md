# Performance Batch V2

## Current implementation

`app/performance_v2.py` provides a self-contained bounded preview cache and stable recipe hashing. It does not depend on optional helpers from other modules.

## Use

```python
from app.performance_v2 import PreviewCache

cache = PreviewCache(".photo-studio/previews", max_items=100)
key = cache.key(source_bytes, options)
preview = cache.get(key)
if preview is None:
    preview = process_image()
    cache.put(key, preview)
```

## Integration note

The existing `app/performance.py` currently imports a missing `preview_image` symbol from `app.professional_tools`. It needs an in-place compatibility patch to delegate to `performance_v2` before the original full-suite performance test can pass.
