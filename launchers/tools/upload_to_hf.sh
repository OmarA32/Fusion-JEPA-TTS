#!/bin/bash
cd "$(dirname "$0")/../.."

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

if [ $# -eq 0 ]; then
    echo "=============================================================================="
    echo "  Hugging Face Checkpoint Uploader"
    echo "=============================================================================="
    echo ""
    echo "  Available Arguments:"
    echo "    --lang      [arabic, english] Language model checkpoint to upload"
    echo "    --token     Hugging Face user access token (write permissions)"
    echo "    --repo      Hugging Face repository ID"
    echo "    --epoch     Specific epoch number to upload (optional: uploads latest)"
    echo ""
    echo "  Examples:"
    echo "    bash launchers/upload_to_hf.sh --lang arabic --token \"hf_...\""
    echo "    bash launchers/upload_to_hf.sh --lang english --token \"hf_...\""
    echo "=============================================================================="
    read -p "Enter arguments (or press Enter for Arabic): " -r user_args
    python tools/upload_to_hf.py $user_args
else
    python tools/upload_to_hf.py "$@"
fi
