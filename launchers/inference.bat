@echo off
cd /d "%~dp0\.."
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo ==============================================================================
    echo   Fusion-JEPA TTS Inference Runner
    echo ==============================================================================
    echo.
    echo   Available Arguments:
    echo     --lang           [arabic, english] Language of the model (default: arabic)
    echo     --text           Custom text string to synthesize into speech
    echo     --output         Custom output WAV file path or filename (default: output_test.wav)
    echo     --file_name      Alias for --output
    echo     --db             [nawar_halabi, common_voice, clartts, ljspeech, libritts] (Only used with --index)
    echo     --index          Dataset item index to fetch exact text and ground truth mel
    echo     --ckpt           Explicit path to a .ckpt or .pt model checkpoint
    echo     --cfg-scale      Classifier-Free Guidance scale (default: 7.0)
    echo     --steps          Number of ODE flow matching steps (default: 60)
    echo     --save-mel       Flag to save an image of the mel spectrogram (.png)
    echo     --no-trim        Flag to disable automatic post-speech silence trimming
    echo.
    echo   Examples:
    echo     launchers\inference.bat --lang english --text "Hello world from Fusion-JEPA"
    echo     launchers\inference.bat --lang arabic --text "مرحبا بكم" --save-mel
    echo     launchers\inference.bat --lang english --db ljspeech --index 108 --save-mel
    echo ==============================================================================
    set /p args="Enter arguments (or press Enter for default Arabic synthesis): "
)

python inference.py %args%
pause
