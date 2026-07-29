@echo off
cd /d "%~dp0\.."
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo =========================================================
    echo   No arguments provided.
    echo.
    echo   Available Arguments:
    echo     --lang           [arabic, english] (Optional: Language of the model to upload)
    echo     --token          (Optional: Hugging Face Write Token, leave blank if setup ran)
    echo.
    echo   Defaults: --lang arabic
    echo =========================================================
    set /p args="Enter arguments (or press Enter to use defaults): "
)

python upload_to_hf.py %args%
pause
