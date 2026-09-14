# Photo Library Workflow

## Goal

Manage a local collection of photos before editing: scan folders, view thumbnails, rate images, add tags, and search.

## Launch

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\python.exe -m streamlit run app/frontend_library.py
```

## Steps

1. Enter a local photo folder.
2. Choose whether subfolders should be included.
3. Click `Scan folder`.
4. Review the thumbnail grid.
5. Set a rating and add comma-separated tags.
6. Click `Save` for an image.
7. Search by filename, path, or tag.
8. Use the catalog result to choose photos for the main editor.

The catalog database and generated thumbnails are stored under `%USERPROFILE%\\.photo-studio` and do not modify source images.
