#!/bin/bash
cd "$(dirname "$0")"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
echo "Deleting old training weights..."
rm -rf training_logs
echo "Starting JEPA-T training from scratch..."
python train.py
