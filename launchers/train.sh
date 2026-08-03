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
    echo "    --val            (Optional: Flag to enable validation loops)"
    echo "    --resume         (Optional: Flag to resume from the latest checkpoint)"
    echo "    --freeze_jepa    (Optional: Freeze the ViT backbone and train only the Diffusion MLP)"
    echo "    --checkpointnum  (Optional: Number of epochs between auto-uploads to HF)"
    echo "    --hf_token       (Optional: Save a Hugging Face token to hf_config.json automatically)"
    echo "    --download_latest(Optional: Download the latest checkpoint from HF before starting)"
    echo ""
    echo "  Defaults: --lang arabic --db nawar_halabi"
    echo "========================================================="
    read -p "Enter arguments (or press Enter to use defaults): " ARGS
fi

python train.py $ARGS
