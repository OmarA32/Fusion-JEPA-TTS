#!/bin/bash
cd "$(dirname "$0")/.."
[ -f "venv/bin/activate" ] && source venv/bin/activate

ARGS="$@"

if [ -z "$ARGS" ]; then
    echo "========================================================="
    echo "  No arguments provided."
    echo ""
    echo "  Available Arguments:"
    echo "    --lang    [arabic, english]"
    echo "    --db      [common_voice, nawar_halabi, libritts, ljspeech]"
    echo "    --index   [any number, e.g., 15] (Automatically pulls text from dataset)"
    echo ""
    echo "  Defaults: --lang arabic --db common_voice"
    echo "========================================================="
    read -p "Enter arguments (e.g., --lang english) or press Enter for defaults: " ARGS
fi

python inference.py $ARGS
