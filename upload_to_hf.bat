@echo off
setlocal

:: If no arguments are provided, switch to interactive mode
if "%~1"=="" (
    echo ==============================================
    echo  JEPA-TTS Hugging Face Model Uploader
    echo ==============================================
    echo.
    set /p CKPT="1. Enter the path to your .ckpt file (e.g. training_logs\arabic\nawar_halabi\best-epoch=010.ckpt): "
    set /p REPO="2. Enter your Hugging Face Repo ID (e.g. KASP-JEPA/Project-Arabic): "
    set /p TOKEN="3. Enter your Hugging Face WRITE token: "
    echo.
    echo Starting Upload...
    venv\Scripts\python upload_to_hf.py --ckpt "%CKPT%" --repo "%REPO%" --token "%TOKEN%"
    echo.
    pause
) else (
    :: If arguments are provided, just pass them straight to the python script
    venv\Scripts\python upload_to_hf.py %*
)
