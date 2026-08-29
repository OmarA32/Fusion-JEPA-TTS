@echo off
cd /d "%~dp0\.."
if not exist "BigVGAN\meldataset.py" (
    echo Initializing BigVGAN neural vocoder submodule...
    git submodule update --init --recursive
)
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo ==============================================================================
    echo   Fusion-JEPA Overfitting Training Runner (Single Sample Convergence Test)
    echo ==============================================================================
    echo.
    echo   Available Arguments:
    echo     --lang             [arabic, english] Language of the model (default: english)
    echo     --db               [nawar_halabi, common_voice, clartts, ljspeech, libritts] Dataset
    echo     --index            Sample index to overfit on (default: 108 for English, 107 for Arabic)
    echo     --epochs           Total number of epochs (default: 5000)
    echo     --lr               Learning rate for AdamW (default: 1e-4)
    echo     --resume           Flag to resume from existing overfit checkpoint
    echo     --freeze_jepa      Flag to freeze ViT backbone
    echo     --freeze_diffuser  Flag to freeze DiT diffuser
    echo     --save-mel         Flag to save generated Mel comparison during evaluation
    echo.
    echo   Examples:
    echo     launchers\overfit_train.bat --lang english --db ljspeech --index 108 --epochs 5000
    echo     launchers\overfit_train.bat --lang arabic --db nawar_halabi --index 107 --epochs 5000
    echo ==============================================================================
    set /p args="Enter arguments (or press Enter for default English overfitting): "
)

python overfit_train.py %args%
pause
