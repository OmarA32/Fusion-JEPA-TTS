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
    echo     --val            (Optional: Flag to enable validation loops)
    echo     --resume         (Optional: Flag to resume from the latest checkpoint)
    echo     --freeze_jepa    (Optional: Freeze the ViT backbone and train only the Diffusion MLP)
    echo     --freeze_diffuser(Optional: Freeze the SpatialDiT diffuser and train only the JEPA backbone)
    echo     --epochs         (Optional: Maximum number of epochs to train, e.g. 2600)
    echo     --checkpointnum  (Optional: Number of epochs between auto-uploads to HF)
    echo.
    echo   Defaults: --lang arabic --db nawar_halabi
    echo =========================================================
    set /p args="Enter arguments (or press Enter to use defaults): "
)

python train.py %args%
pause
