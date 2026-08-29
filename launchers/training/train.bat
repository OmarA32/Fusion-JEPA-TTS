@echo off
cd /d "%~dp0\..\.."
if not exist "BigVGAN\meldataset.py" (
    echo Initializing BigVGAN neural vocoder submodule...
    git submodule update --init --recursive
)
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set "args=%*"

if "%args%"=="" (
    echo ==============================================================================
    echo   Fusion-JEPA Full Distributed Training Runner
    echo ==============================================================================
    echo.
    echo   Available Arguments:
    echo     --lang             [arabic, english] Language of the model (default: arabic)
    echo     --db               [nawar_halabi, common_voice, clartts, ljspeech, libritts] Dataset to train on
    echo     --epochs           Total number of epochs to train (default: 2600)
    echo     --batch_size       Batch size per GPU (default: 16)
    echo     --lr               Learning rate for AdamW (default: 1e-4)
    echo     --resume           Flag to automatically resume from latest checkpoint
    echo     --download_latest  Flag to download latest checkpoint from Hugging Face if missing
    echo     --val              Flag to enable validation evaluation loop
    echo     --freeze_jepa      Flag to freeze ViT backbone and train only Flow Matching DiT
    echo     --freeze_diffuser  Flag to freeze DiT diffuser and train only JEPA backbone
    echo     --checkpointnum    Epoch interval between Hugging Face auto-uploads (e.g. 80 or 150)
    echo     --hf_token         Hugging Face user access token for automated sync
    echo.
    echo   Examples:
    echo     launchers\training\train.bat --lang arabic --db nawar_halabi --resume --checkpointnum 150
    echo     launchers\training\train.bat --lang english --db ljspeech --resume --checkpointnum 80
    echo ==============================================================================
    set /p args="Enter arguments (or press Enter for default Arabic training): "
)

python training/train.py %args%
pause
