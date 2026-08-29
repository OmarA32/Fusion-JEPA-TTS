#!/bin/bash
cd "$(dirname "$0")/../.."
[ -f "venv/bin/activate" ] && source venv/bin/activate

ARGS="$@"

if [ -z "$ARGS" ]; then
    echo "========================================================="
    echo "  No arguments provided."
    echo ""
    echo "  Available Arguments:"
    echo "    --lang           [arabic, english] (Optional: Language of the model)"
    echo "    --db             [common_voice, nawar_halabi, ljspeech, clartts, libritts] (Optional: Dataset database)"
    echo "    --index          (Optional: Index of the test dataset item to synthesize)"
    echo ""
    echo "  Defaults: --lang arabic --db nawar_halabi"
    echo "========================================================="
    read -p "Enter arguments (or press Enter to use defaults): " ARGS
fi

python tools/test_vocoder_ground_truth.py $ARGS
