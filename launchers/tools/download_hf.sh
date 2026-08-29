#!/bin/bash
cd "$(dirname "$0")/../.."

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

if [ $# -eq 0 ]; then
    echo "=============================================================================="
    echo "  Hugging Face Model Downloader"
    echo "=============================================================================="
    echo ""
    echo "  Available Arguments:"
    echo "    --lang      [arabic, english] Model language checkpoint to download"
    echo "    --token     Hugging Face user access token (optional if set in hf_config.json)"
    echo "    --repo      Hugging Face repository ID (default: KAST-JEPA-QUANTIZED/Arabic or English)"
    echo ""
    echo "  Examples:"
    echo "    bash launchers/download_hf.sh --lang arabic"
    echo "    bash launchers/download_hf.sh --lang english"
    echo "=============================================================================="
    read -p "Enter arguments (or press Enter for Arabic): " -r user_args
    python tools/download_from_hf.py $user_args
else
    python tools/download_from_hf.py "$@"
fi
