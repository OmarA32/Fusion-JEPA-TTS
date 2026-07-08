@echo off
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
echo Testing index 10 with Vocos vocoder...
python test_split.py --vocoder vocos --index 10
pause
