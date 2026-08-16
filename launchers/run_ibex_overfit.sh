#!/bin/bash
# ==============================================================================
# SLURM submission script for KAUST IBEX Supercomputer -- Audio-JEPA-Arabic-TTS
# ==============================================================================
# HOW TO DEPLOY ON IBEX:
#   1. SSH into the GPU login node: ssh -XY username@glogin.ibex.kaust.edu.sa
#   2. Clone the repo (WITH SUBMODULES!): git clone --recursive https://github.com/OmarA32/Audio-JEPA-Arabic-TTS.git
#   3. cd Audio-JEPA-Arabic-TTS
#   4. Submit this script to the queue: sbatch launchers/run_ibex_overfit.sh
# ==============================================================================

#SBATCH --job-name=jepa_tts_overfit
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1         # Request 1x NVIDIA A100 GPU
#SBATCH --cpus-per-task=4         # Request 4 CPU cores
#SBATCH --mem=32G                 # Request 32GB of RAM
#SBATCH --time=04:00:00           # 4-hour strict time limit (overfitting is fast)
#SBATCH --output=training_logs/ibex_output_%j.txt
#SBATCH --error=training_logs/ibex_error_%j.txt

echo "=========================================================="
echo "Starting JEPA TTS Training (SCORCHED EARTH OVERFITTING) on A100"
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
python -m venv ibex_tts_env --system-site-packages
source ibex_tts_env/bin/activate
pip install -r requirements.txt

# 5. Launch Overfitting Script
echo "Booting up the Overfitting Script on exactly 8 audio files..."
python overfit_train.py --lang english --db ljspeech --resume

echo "Job Completed!"
