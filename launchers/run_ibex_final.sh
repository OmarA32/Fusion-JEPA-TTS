#!/bin/bash
# ==============================================================================
# SLURM submission script for KAUST IBEX Supercomputer -- Audio-JEPA English TTS
# (1x A100 GPU, 2600 Epochs, LJSpeech)
# ==============================================================================
# HOW TO DEPLOY ON IBEX:
#   1. SSH into the GPU login node: ssh -XY username@glogin.ibex.kaust.edu.sa
#   2. Clone the repo: git clone --recursive https://github.com/OmarA32/Fusion-JEPA-TTS.git
#   3. cd Fusion-JEPA-TTS
#   4. Submit this script: sbatch launchers/run_ibex_final.sh "YOUR_HF_TOKEN_HERE"
# ==============================================================================

#SBATCH --job-name=jepa_tts_english_2600ep
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1         # Request 1x NVIDIA A100 GPU
#SBATCH --cpus-per-task=8         # Request 8 CPU cores for fast data loading
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

# 5. Launch PyTorch Lightning Training
# 1x A100 GPU, 2600 Epochs on LJSpeech English Benchmark
echo "Booting up the Trainer on 1x A100 GPU for 2,600 Epochs..."
python train.py --resume --download_latest --lang english --db ljspeech --epochs 2600 --hf_token "$1" --checkpointnum 40

echo "Job Completed!"
