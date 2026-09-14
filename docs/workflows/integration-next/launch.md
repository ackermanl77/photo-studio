# Integrated App Launch

## Browser mode

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\python.exe -m streamlit run app/frontend_integrated.py
```

Open the URL Streamlit prints, normally `http://localhost:8501`.

## Native desktop mode

```powershell
.\.venv\Scripts\python.exe app/desktop_integrated.py
```

This opens the integrated frontend in a PyWebView window on port 8506.

The integrated entry combines the mature editor with layers, blend modes, healing, previews, catalog ratings, tags, and search.
