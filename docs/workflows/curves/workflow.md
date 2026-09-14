# Curves Workflow

## Goal

Shape the tonal response with explicit input/output points.

## Steps

1. Begin with the neutral curve `0:0,255:255`.
2. Add a midpoint to lift or lower midtones.
3. Keep input values unique and within `0` to `255`.
4. Compare skin, highlights, and shadows before exporting.

## CLI

```powershell
python -m app --input input.jpg --output output.jpg --curve "0:8,64:58,128:140,192:202,255:250"
```

The current implementation applies one RGB curve to all channels.