#!/bin/bash
cd "$(dirname "$0")/../.."

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

if [ $# -eq 0 ]; then
    echo "=============================================================================="
    echo "  Fusion-JEPA Long-Form Speech Synthesis Runner"
    echo "=============================================================================="
    echo ""
    echo "  Available Arguments:"
    echo "    --lang           [arabic, english] Language of the model (default: arabic)"
    echo "    --text           Long text string or paragraph to synthesize into speech"
    echo "    --file           Path to a .txt file containing the long paragraph"
    echo "    --output         Custom output WAV file path (default: test_results/longform_output.wav)"
    echo "    --file_name      Alias for --output"
    echo "    --ckpt           Explicit path to a .ckpt or .pt model checkpoint"
    echo "    --cfg-scale      Classifier-Free Guidance scale (default: 7.0)"
    echo "    --steps          Number of ODE flow matching steps per clause (default: 60)"
    echo "    --pause-ms       Pause duration between clauses in milliseconds (default: 100)"
    echo "    --save-mel       Flag to save stitched Mel-spectrogram comparison image"
    echo "    --no-trim        Flag to disable automatic per-chunk silence trimming"
    echo ""
    echo "  Examples:"
    echo "    bash launchers/longform_inference.sh --lang english --text \"Fusion-JEPA is a deep multimodal architecture designed for expressive text-to-speech synthesis.\""
    echo "    bash launchers/longform_inference.sh --lang arabic --text \"يعتمد نظام فيوجن جيبا على التعلم الذاتي لتوليد صوت عالي الجودة\""
    echo "=============================================================================="
    read -p "Enter arguments (or press Enter for default synthesis): " -r user_args
    python pipelines/longform_inference.py $user_args
else
    python pipelines/longform_inference.py "$@"
fi
