@echo off
chcp 65001 >nul
echo ========================================
echo   Starting Python RAG Service (8001)
echo ========================================
echo.
cd /d "%~dp0rag-service"

if not exist ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    echo [INFO] Installing dependencies...
    .venv\Scripts\pip.exe install -r requirements.txt
)

echo [INFO] Starting uvicorn on port 8001...
.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8001 --reload
pause
