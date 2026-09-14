# Flask Backend Workflow

## Architecture

The project now has two local services:

- Streamlit frontend: `http://localhost:8501`
- Flask processing backend: `http://127.0.0.1:8502`

The frontend starts the Flask backend automatically through `python -m app.backend` and reuses it across Streamlit reruns. The backend calls the existing `PhotoProcessor` code for every preview and export request.

## Normal launch

Start only the frontend:

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
.\.venv\Scripts\Activate.ps1
streamlit run app/frontend.py
```

The frontend checks `/health`, starts Flask if needed, and sends image data to `/process`.

## Manual backend launch

For API development or troubleshooting, run the backend separately:

```powershell
python -m app.backend
```

## API contract

- `GET /health` returns service status.
- `POST /process` accepts an `image`, optional `removal_mask`, optional `local_mask`, and an `options` JSON form field.
- The response is the edited JPEG or PNG binary.

## Troubleshooting

If port `8502` is occupied, stop the existing process before launching the frontend. The frontend currently expects the fixed local backend address `127.0.0.1:8502`.