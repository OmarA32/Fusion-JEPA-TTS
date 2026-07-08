#!/bin/bash
cd "$(dirname "$0")"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
echo "Running full project test with BigVGAN vocoder..."
python test_split.py --vocoder bigvgan
