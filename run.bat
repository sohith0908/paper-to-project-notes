@echo off
setlocal
cd /d "%~dp0"

echo.
echo  Paper-to-Project Notes
echo  ----------------------
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
  echo Then tick "Add Python to PATH" and run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create .venv
    pause
    exit /b 1
  )
)

echo Installing / updating dependencies...
".venv\Scripts\python.exe" -m pip install -q --upgrade pip
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  pause
  exit /b 1
)

echo.
echo Starting app... Browser should open at http://localhost:8501
echo Close this window to stop the server.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py

pause
