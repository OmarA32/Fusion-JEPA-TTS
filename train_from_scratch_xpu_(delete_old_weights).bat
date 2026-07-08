@echo off
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat

echo Deleting old training weights...
if exist "training_logs" rmdir /s /q "training_logs"

echo Starting Pure PyTorch JEPA-T training on XPU...
python -u train_xpu.py
pause
