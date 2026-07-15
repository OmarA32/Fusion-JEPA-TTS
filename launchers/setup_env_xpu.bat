@echo off
cd /d "%~dp0"

echo Creating virtual environment (venv)...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel

echo Installing PyTorch with Intel XPU support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

echo Installing remaining requirements...
pip install -r requirements.txt

echo.
echo Environment setup is complete!
pause
