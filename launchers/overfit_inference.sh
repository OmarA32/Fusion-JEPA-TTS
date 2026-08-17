#!/bin/bash
cd "$(dirname "$0")/.."
[ -f "venv/bin/activate" ] && source venv/bin/activate

ARGS="$@"

if [ -z "$ARGS" ]; then
    echo "========================================================="
    echo "  No arguments provided."
    echo ""
    echo "  Available Arguments:"
    echo "    --lang           [arabic, english] (Optional: Language of the model)"
    echo "    --db             [common_voice, nawar_halabi, ljspeech, clartts, libritts] (Optional: Dataset database)"
    echo "    --text           (Optional: Custom text string to synthesize into speech)"
    echo "    --index          (Optional: Dataset item index to fetch the exact text and ground truth mel)"
    echo "    --output         (Optional: Custom output WAV file path)"
    echo "    --vocoder        [vocos, bigvgan] (Optional: Vocoder to use)"
    echo "    --cfg-scale      (Optional: Classifier-Free Guidance scale, default 1.0)"
    echo "    --steps          (Optional: Number of ODE diffusion steps, default 60)"
    echo "    --save-mel       (Optional: Save an image of the mel spectrogram)"
    echo ""
    echo "  Defaults: --lang arabic --db nawar_halabi"
    echo "========================================================="
    read -p "Enter arguments (or press Enter to use defaults): " ARGS
fi

python overfit_inference.py $ARGS
