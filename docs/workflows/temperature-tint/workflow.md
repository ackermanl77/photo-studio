# Temperature and Tint Workflow

## Goal

Correct the warm/cool balance and green/magenta cast of an image.

## Steps

1. Start with temperature and tint at `0`.
2. Increase temperature for a warmer image; decrease it for a cooler image.
3. Increase tint toward magenta; decrease it toward green.
4. Make small adjustments and inspect neutral objects and skin tones.
5. Apply curves only after the color balance is acceptable.

## CLI

```powershell
python -m app --input input.jpg --output output.jpg --temperature 12 --tint -4
```

Values are clamped to `-100` through `100` and run on the CPU.