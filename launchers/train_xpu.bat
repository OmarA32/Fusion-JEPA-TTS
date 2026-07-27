@echo off
cd /d "%~dp0\.."
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo =========================================================
    echo   No arguments provided.
    echo.
    echo   Available Arguments:
    echo     --lang           [arabic, english]
    echo     --db             [common_voice, clartts, nawar_halabi, libritts, ljspeech]
    echo     --val            (add this flag to enable validation loops)
    echo     --resume         (add this flag to resume from latest checkpoint)
    echo     --checkpointnum  (number of epochs between auto-uploads to HF)
    echo.
    echo   Defaults: --lang arabic --db nawar_halabi
    echo =========================================================
    set /p args="Enter arguments (e.g., --lang english --resume) or press Enter for defaults: "
)

python train_xpu.py %args%
pause
