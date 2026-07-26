@echo off
:: Hugging Face Weights Downloader Launcher
:: Usage: download_hf.bat [arabic|english]
:: Example: download_hf.bat arabic

if "%~1"=="" (
    echo Error: You must specify a language.
    echo Usage: download_hf.bat [arabic^|english]
    exit /b 1
)

set LANG=%~1

python download_from_hf.py --lang %LANG%
