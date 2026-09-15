#!/bin/bash
# ==============================================================================
# SLURM submission script for KAUST IBEX Supercomputer -- Download English Weights
# ==============================================================================
# HOW TO DEPLOY ON IBEX:
#   sbatch launchers/tools/download_latest_weights_english.sh "YOUR_HF_TOKEN_HERE"
# ==============================================================================

#SBATCH --job-name=dl_weights_en
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1         # Request 1x NVIDIA A100 GPU (Fast queue scheduling!)
#SBATCH --cpus-per-task=12        # Request 12 CPU cores for fast data loading
#SBATCH --mem=64G                 # Request 64GB of RAM
#SBATCH --time=04:00:00           # 4-hour time limit
#SBATCH --output=training_logs/ibex_download_en_%j.txt
#SBATCH --error=training_logs/ibex_download_en_%j.txt

echo "=========================================================="
echo "Starting Download of Latest English Weights on IBEX"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "=========================================================="

# Ensure we are in the repository root directory
cd "$(dirname "$0")/../.."
mkdir -p training_logs/english

# 1. Initialize Git Submodules (Crucial for BigVGAN!)
echo "Ensuring Git submodules (BigVGAN) are initialized and updated..."
git submodule update --init --recursive

# 2. Load the optimized Ibex Machine Learning Environment
echo "Loading Ibex machine_learning module..."
module purge
module load machine_learning/2024.01

# 3. Print GPU Status
nvidia-smi

# 4. Create and activate a 100% isolated local virtual environment
echo "Creating isolated local environment to prevent user conflicts..."
python -m venv ibex_tts_env --system-site-packages
source ibex_tts_env/bin/activate
pip install -r requirements.txt

# 5. Save HF Token if provided
if [ -n "$1" ]; then
    echo "Saving HF token to hf_config.json..."
    echo "{\"HF_TOKEN\": \"$1\"}" > hf_config.json
fi

# 6. Download latest English weights
echo "Downloading latest English checkpoint..."
python tools/download_from_hf.py --lang english

echo "Job Completed!"
