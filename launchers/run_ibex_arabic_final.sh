#!/bin/bash
# ==============================================================================
# SLURM submission script for KAUST IBEX Supercomputer -- Audio-JEPA Arabic TTS
# Matches D-JEPA paper scale (4x A100 GPUs, 2600 Epochs, Nawar Halabi)
# ==============================================================================
# HOW TO DEPLOY ON IBEX:
#   1. SSH into the GPU login node: ssh -XY username@glogin.ibex.kaust.edu.sa
#   2. Clone the repo: git clone --recursive https://github.com/OmarA32/Audio-JEPA-Arabic-TTS.git
#   3. cd Audio-JEPA-Arabic-TTS
#   4. Submit this script: sbatch launchers/run_ibex_arabic_final.sh "YOUR_HF_TOKEN_HERE"
# ==============================================================================

#SBATCH --job-name=jepa_tts_arabic_2600ep
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:4         # Request 4x NVIDIA A100 GPUs (Paper configuration)
#SBATCH --cpus-per-task=32        # Request 32 CPU cores for fast multi-GPU data loading
#SBATCH --mem=128G                # Request 128GB of RAM
#SBATCH --time=24:00:00           # 24-hour time limit
#SBATCH --output=training_logs/ibex_output_%j.txt
#SBATCH --error=training_logs/ibex_error_%j.txt

echo "=========================================================="
echo "Starting JEPA TTS Arabic Training on 4x A100 GPUs (2,600 Epochs)"
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

# 5. Launch PyTorch Lightning Distributed Training
# 4x A100 GPUs, 2600 Epochs on Nawar Halabi Arabic Dataset
echo "Booting up the Trainer across 4x A100 GPUs for 2,0000 Epochs..."
python train.py --resume --download_latest --lang arabic --db nawar_halabi --epochs 20000 --hf_token "$1" --checkpointnum 80

echo "Job Completed!"
