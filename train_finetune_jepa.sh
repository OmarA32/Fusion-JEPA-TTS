#!/bin/bash
cd "$(dirname "$0")"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
echo "Starting JEPA-T training (resuming from latest checkpoint)..."
python train.py --resume
