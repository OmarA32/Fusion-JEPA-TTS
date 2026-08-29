#!/bin/bash
# ==============================================================================
# SLURM submission script for KAUST IBEX Supercomputer -- Audio-JEPA English TTS
# (1x A100 GPU, 2600 Epochs, LJSpeech)
# ==============================================================================
# HOW TO DEPLOY ON IBEX:
#   1. SSH into the GPU login node: ssh -XY username@glogin.ibex.kaust.edu.sa
#   2. cd Fusion-JEPA-TTS
#   3. Submit this script: sbatch launchers/run_ibex_english_1gpu.sh "YOUR_HF_TOKEN_HERE"
# ==============================================================================

#SBATCH --job-name=jepa_tts_en_1gpu
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1         # Request 1x NVIDIA A100 GPU (Fast queue scheduling!)
#SBATCH --cpus-per-task=12        # Request 12 CPU cores for fast data loading
#SBATCH --mem=64G                 # Request 64GB of RAM
#SBATCH --time=24:00:00           # 24-hour time limit
#SBATCH --output=training_logs/ibex_output_%j.txt
#SBATCH --error=training_logs/ibex_error_%j.txt

echo "=========================================================="
echo "Starting JEPA TTS Training on 1x A100 GPU (2,600 Epochs)"
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

# 5. Launch PyTorch Lightning Training on 1x GPU
echo "Booting up Trainer on 1x A100 GPU for 2,600 Epochs..."
python training/train.py --resume --download_latest --lang english --db ljspeech --epochs 2600 --hf_token "$1" --checkpointnum 80

echo "Job Completed!"
