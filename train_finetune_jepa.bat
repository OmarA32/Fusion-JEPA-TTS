@echo off
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
echo Starting JEPA-T training (resuming from latest checkpoint)...
python train.py --resume
pause
