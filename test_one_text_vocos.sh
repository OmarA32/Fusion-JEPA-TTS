#!/bin/bash
cd "$(dirname "$0")"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
echo "Testing index 10 with Vocos vocoder..."
python test_split.py --vocoder vocos --index 10
