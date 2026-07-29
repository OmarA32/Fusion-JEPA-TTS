@echo off
cd /d "%~dp0\.."
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo =========================================================
    echo   No arguments provided.
    echo.
    echo   Available Arguments:
    echo     --lang           [arabic, english] (Optional: Language of the model to download)
    echo.
    echo   Defaults: --lang arabic
    echo =========================================================
    set /p args="Enter arguments (or press Enter to use defaults): "
)

python download_from_hf.py %args%
pause
