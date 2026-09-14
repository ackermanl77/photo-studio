# Non-Destructive Edit Stack Workflow

## Goal

Manage edits as named operations that can be disabled, reordered, removed, saved, and reapplied.

## Current API

```python
from app.edit_stack import EditStack

stack = (
    EditStack()
    .add("Exposure", "apply_exposure", {"value": 1.1})
    .add("Contrast", "apply_contrast", {"value": 1.15})
)
stack.set_enabled("Contrast", False)
stack.move("Exposure", -1)
stack.apply(processor)
```

## Steps

1. Add each adjustment with a readable name.
2. Disable an operation to compare the result without deleting it.
3. Move operations when processing order matters.
4. Remove unwanted operations.
5. Save the stack as JSON for a reproducible recipe.
6. Reload it and apply it to the original image.

## Design rule

The stack stores operations and parameters, not rendered pixels. This keeps edits reversible and makes the same recipe usable on multiple images.
