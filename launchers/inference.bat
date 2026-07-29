@echo off
cd /d "%~dp0\.."
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo =========================================================
    echo   No arguments provided.
    echo.
    echo   Available Arguments:
    echo     --lang           [arabic, english] (Optional: Language of the model)
    echo     --db             [common_voice, nawar_halabi, ljspeech, clartts, libritts] (Optional: Dataset database)
    echo     --text           (Optional: Custom text string to synthesize into speech)
    echo     --index          (Optional: Dataset item index to fetch the exact text and ground truth mel)
    echo     --output         (Optional: Custom output WAV file path)
    echo     --vocoder        [vocos, bigvgan] (Optional: Vocoder to use)
    echo     --cfg-scale      (Optional: Classifier-Free Guidance scale, default 1.0)
    echo     --save-mel       (Optional: Save an image of the mel spectrogram)
    echo.
    echo   Defaults: --lang arabic --db nawar_halabi
    echo =========================================================
    set /p args="Enter arguments (or press Enter to use defaults): "
)

python inference.py %args%
pause
