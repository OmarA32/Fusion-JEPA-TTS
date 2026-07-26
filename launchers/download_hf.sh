#!/bin/bash
# Hugging Face Weights Downloader Launcher
# Usage: ./download_hf.sh [arabic|english]
# Example: ./download_hf.sh arabic

if [ -z "$1" ]; then
    echo "Error: You must specify a language."
    echo "Usage: ./download_hf.sh [arabic|english]"
    exit 1
fi

LANG=$1

python download_from_hf.py --lang "$LANG"
