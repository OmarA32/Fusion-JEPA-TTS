@echo off
cd /d "%~dp0\.."
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo ==============================================================================
    echo   Hugging Face Checkpoint Uploader
    echo ==============================================================================
    echo.
    echo   Available Arguments:
    echo     --lang      [arabic, english] Language model checkpoint to upload
    echo     --token     Hugging Face user access token (write permissions)
    echo     --repo      Hugging Face repository ID
    echo     --epoch     Specific epoch number to upload (optional: uploads latest)
    echo.
    echo   Examples:
    echo     launchers\upload_to_hf.bat --lang arabic --token "hf_..."
    echo     launchers\upload_to_hf.bat --lang english --token "hf_..."
    echo ==============================================================================
    set /p args="Enter arguments (or press Enter for Arabic): "
)

python upload_to_hf.py %args%
pause
