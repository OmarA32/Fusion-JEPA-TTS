@echo off
cd /d "%~dp0\.."
setlocal

:: If no arguments are provided, switch to interactive mode
if "%~1"=="" (
    echo ==============================================
    echo  JEPA-TTS Hugging Face Model Uploader
    echo ==============================================
    echo.
    set /p LANG="1. Enter the language model (arabic or english): "
    set /p TOKEN="2. (Optional) Enter your Hugging Face WRITE token (leave blank if you ran setup): "
    echo.
    echo Starting Upload...
    if "%TOKEN%"=="" (
        venv\Scripts\python upload_to_hf.py --lang "%LANG%"
    ) else (
        venv\Scripts\python upload_to_hf.py --lang "%LANG%" --token "%TOKEN%"
    )
    echo.
    pause
) else (
    :: If arguments are provided, just pass them straight to the python script
    venv\Scripts\python upload_to_hf.py %*
)
