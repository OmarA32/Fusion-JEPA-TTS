@echo off
cd /d "%~dp0\.."
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo =========================================================
    echo   No arguments provided.
    echo.
    echo   Available Arguments:
    echo     --lang    [arabic, english]
    echo     --db      [common_voice, clartts, nawar_halabi, libritts, ljspeech]
    echo     --index   [any number, e.g., 15] (Automatically pulls text from dataset)
    echo.
    echo   Defaults: --lang arabic --db common_voice
    echo =========================================================
    set /p args="Enter arguments (e.g., --lang english) or press Enter for defaults: "
)

python inference.py %args%
pause
