#!/bin/bash
cd "$(dirname "$0")/.."
[ -f "venv/bin/activate" ] && source venv/bin/activate

ARGS="$@"

if [ -z "$ARGS" ]; then
    echo "========================================================="
    echo "  No arguments provided."
    echo ""
    echo "  Available Arguments:"
    echo "    --lang           [arabic, english] (Optional: Language of the model to download)"
    echo ""
    echo "  Defaults: --lang arabic"
    echo "========================================================="
    read -p "Enter arguments (or press Enter to use defaults): " ARGS
fi

python download_from_hf.py $ARGS
