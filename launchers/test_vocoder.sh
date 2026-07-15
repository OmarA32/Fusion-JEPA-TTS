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
    echo ""
    echo "  Defaults: --lang arabic --db common_voice"
    echo "========================================================="
    read -p "Enter arguments (e.g., --lang english) or press Enter for defaults: " ARGS
fi

python test_vocoder_ground_truth.py $ARGS
