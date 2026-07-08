@echo off
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
echo Running full project test with Vocos vocoder...
python test_split.py --vocoder vocos
pause
