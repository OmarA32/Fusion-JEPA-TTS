#!/bin/bash
cd "$(dirname "$0")/.."
[ -f "venv/bin/activate" ] && source venv/bin/activate

ARGS="$@"

if [ -z "$ARGS" ]; then
    echo "========================================================="
    echo "  No arguments provided."
    echo ""
    echo "  Available Arguments:"
    echo "    --lang           [arabic, english]"
    echo "    --db             [common_voice, nawar_halabi, libritts, ljspeech]"
    echo "    --resume         (add this flag to resume from latest checkpoint)"
    echo "    --checkpointnum  (number of epochs between auto-uploads to HF)"
    echo ""
    echo "  Defaults: --lang arabic --db nawar_halabi"
    echo "========================================================="
    read -p "Enter arguments (e.g., --lang english --resume) or press Enter for defaults: " ARGS
fi

python train.py $ARGS
