# Global Python Environment Workflow

This project can run from the system Python installation instead of `.venv`.

## Interpreter

```text
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

## Run the complete frontend

```powershell
cd "C:\Users\MELEJ\OneDrive\photo_professional_tool"
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app/frontend_complete.py
```

## Run the editor

```powershell
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app/frontend_pro.py
```

## Run tests

```powershell
C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest -q
```

The global environment currently has all core, RAW, desktop, packaging, and test dependencies installed. The existing `.venv` folder is not required for these commands.
