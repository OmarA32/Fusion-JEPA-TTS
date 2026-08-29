@echo off
cd /d "%~dp0\..\.."
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo =========================================================
    echo   No arguments provided.
    echo.
    echo   Available Arguments:
    echo     --lang           [arabic, english] (Optional: Language of the model)
    echo     --db             [common_voice, nawar_halabi, ljspeech, clartts, libritts] (Optional: Dataset database)
    echo     --index          (Optional: Index of the test dataset item to synthesize)
    echo.
    echo   Defaults: --lang arabic --db nawar_halabi
    echo =========================================================
    set /p args="Enter arguments (or press Enter to use defaults): "
)

python tools/test_vocoder_ground_truth.py %args%
pause
