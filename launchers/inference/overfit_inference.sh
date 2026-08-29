#!/bin/bash
cd "$(dirname "$0")/../.."

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

if [ $# -eq 0 ]; then
    echo "=============================================================================="
    echo "  Fusion-JEPA Overfit Inference Runner"
    echo "=============================================================================="
    echo ""
    echo "  Available Arguments:"
    echo "    --lang           [arabic, english] Language of the model (default: arabic)"
    echo "    --text           Custom text string to synthesize into speech"
    echo "    --output         Custom output WAV file path or filename (default: output_test.wav)"
    echo "    --file_name      Alias for --output"
    echo "    --db             [nawar_halabi, common_voice, clartts, ljspeech, libritts] (Only used with --index)"
    echo "    --index          Dataset item index to fetch exact text and ground truth mel"
    echo "    --ckpt           Explicit path to an overfit .ckpt or .pt model checkpoint"
    echo "    --cfg-scale      Classifier-Free Guidance scale (default: 7.0)"
    echo "    --steps          Number of ODE flow matching steps (default: 60)"
    echo "    --save-mel       Flag to save an image of the mel spectrogram (.png)"
    echo "    --no-trim        Flag to disable automatic post-speech silence trimming"
    echo ""
    echo "  Examples:"
    echo "    bash launchers/overfit_inference.sh --lang english --db ljspeech --index 108 --save-mel"
    echo "    bash launchers/overfit_inference.sh --lang arabic --db nawar_halabi --index 107 --save-mel"
    echo "=============================================================================="
    read -p "Enter arguments (or press Enter to use defaults): " -r user_args
    python pipelines/overfit_inference.py $user_args
else
    python pipelines/overfit_inference.py "$@"
fi
