@echo off
title Fusion-JEPA Studio Launcher
cd /d "%~dp0"

echo ==============================================================================
echo             Fusion-JEPA Studio -- Interactive Web Interface
echo ==============================================================================
echo.

:: 1. Check for Virtual Environment Python
if exist "venv\Scripts\python.exe" (
    set "PY_EXE=venv\Scripts\python.exe"
    echo [OK] Using virtual environment Python: venv\Scripts\python.exe
) else (
    set "PY_EXE=python"
    echo [INFO] Virtual environment not detected. Using system Python.
)

:: 2. Ensure Streamlit is installed
echo [INFO] Checking Streamlit installation...
"%PY_EXE%" -c "import streamlit" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INSTALL] Streamlit not found. Installing streamlit...
    "%PY_EXE%" -m pip install streamlit
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install streamlit. Please check your internet connection.
        pause
        exit /b 1
    )
)

echo.
echo ==============================================================================
echo [LAUNCH] Starting Streamlit server...
echo [INFO] Opening browser at http://localhost:8501
echo ==============================================================================
echo.

:: 3. Launch Streamlit
"%PY_EXE%" -m streamlit run app.py --server.port 8501 --server.headless false

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Streamlit exited with an error.
    pause
)
