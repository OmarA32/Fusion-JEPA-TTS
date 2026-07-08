@echo off
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
echo Testing index 10 with HiFi-GAN vocoder...
python test_split.py --vocoder hifigan --index 10
pause
