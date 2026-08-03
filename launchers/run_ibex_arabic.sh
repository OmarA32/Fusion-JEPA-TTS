#!/bin/bash
# ==============================================================================
# SLURM submission script for KAUST IBEX Supercomputer -- Audio-JEPA-Arabic-TTS
# ==============================================================================
# HOW TO DEPLOY ON IBEX:
#   1. SSH into the GPU login node: ssh -XY username@glogin.ibex.kaust.edu.sa
#   2. Clone the repo (WITH SUBMODULES!): git clone --recursive https://github.com/OmarA32/Audio-JEPA-Arabic-TTS.git
#   3. cd Audio-JEPA-Arabic-TTS
#   4. Submit this script to the queue: sbatch launchers/run_ibex_arabic.sh "YOUR_HF_TOKEN_HERE"
# ==============================================================================

#SBATCH --job-name=jepa_tts_ar
#SBATCH --partition=batch
#SBATCH --gres=gpu:1              # Request 1x generic GPU (Any available architecture)
#SBATCH --cpus-per-task=8         # Request 8 CPU cores for fast data loading
#SBATCH --mem=64G                 # Request 64GB of RAM
#SBATCH --time=24:00:00           # 24-hour strict time limit
#SBATCH --output=training_logs/ibex_output_%j.txt
#SBATCH --error=training_logs/ibex_error_%j.txt

echo "=========================================================="
echo "Starting JEPA TTS Training (Arabic) on KAUST Ibex Supercomputer"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "=========================================================="

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
python -m venv ibex_tts_env
source ibex_tts_env/bin/activate
pip install -r requirements.txt

# 5. Launch PyTorch Lightning Training
# We use the new flags to automatically download the latest checkpoint and inject your HF Token!
echo "Booting up the PyTorch Lightning Trainer..."
python train.py --resume --download_latest --lang arabic --db nawar_halabi --hf_token "$1" --checkpointnum 60

echo "Job Completed!"
