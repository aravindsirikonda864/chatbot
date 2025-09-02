
@echo off
REM Start server using venv python (no need to activate manually)
if exist venv\Scripts\python.exe (
    venv\Scripts\python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
) else (
    echo Virtual environment not found. Run setup.ps1 first to create venv and install dependencies.
)
