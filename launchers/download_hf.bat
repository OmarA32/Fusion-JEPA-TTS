@echo off
cd /d "%~dp0\.."
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo ==============================================================================
    echo   Hugging Face Model Downloader
    echo ==============================================================================
    echo.
    echo   Available Arguments:
    echo     --lang      [arabic, english] Model language checkpoint to download
    echo     --token     Hugging Face user access token (optional if set in hf_config.json)
    echo     --repo      Hugging Face repository ID (default: KAST-JEPA-QUANTIZED/Arabic or English)
    echo.
    echo   Examples:
    echo     launchers\download_hf.bat --lang arabic
    echo     launchers\download_hf.bat --lang english
    echo ==============================================================================
    set /p args="Enter arguments (or press Enter for Arabic): "
)

python download_from_hf.py %args%
pause
