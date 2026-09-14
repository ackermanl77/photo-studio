# Edit Stack Frontend Workflow

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\python.exe -m streamlit run app/frontend_stack.py
```

## Steps

1. Upload a photo.
2. Adjust values in Look, Tone, Color, or Detail.
3. Open `Edit stack` to inspect named operations.
4. Use `On/Off` to compare an operation without deleting it.
5. Use `Up` and `Down` to change processing order.
6. Use `Remove` to delete an operation from the current recipe.
7. Use `Reset stack` to return to a clean recipe.
8. Download the edit-stack JSON for reuse.

The image is always rebuilt from the original upload using the current enabled operation order.
