# Object Removal Workflow

## Goal

Remove a small unwanted object using a manually prepared grayscale mask.

## Steps

1. Create a grayscale mask with black for protected pixels and white over the object.
2. Keep the mask the same dimensions as the source image.
3. Use the removal command on a copy of the source.
4. Inspect the repaired area at 100% zoom.
5. Use a smaller mask or a second pass if the repair is too broad.

## CLI

```powershell
python -m app --input input.jpg --output cleaned.jpg --remove-mask object-mask.png
```

Removal uses OpenCV Telea inpainting and is intended for small objects or blemishes.