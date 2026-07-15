#!/bin/bash
cd "$(dirname "$0")/.."

echo "Creating virtual environment (venv)..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip and build tools..."
python3 -m pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Environment setup is complete!"
