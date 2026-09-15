#!/bin/bash
# ==============================================================================
# SLURM script for KAUST IBEX Supercomputer -- Upload English Weights (CPU)
# ==============================================================================
# HOW TO RUN ON IBEX:
#   sbatch launchers/tools/upload_latest_weights_english.sh "YOUR_HF_TOKEN"
# ==============================================================================

#SBATCH --job-name=ul_weights_en
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4          # 4 CPU cores for transfer & checksums
#SBATCH --mem=16G                 # 16GB of RAM
#SBATCH --time=02:00:00           # 2-hour time limit
#SBATCH --output=training_logs/ibex_output_%j.txt
#SBATCH --error=training_logs/ibex_error_%j.txt

echo "=========================================================="
echo "Starting Upload of Latest English Weights on IBEX (CPU)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "Working Directory: $(pwd)"
echo "=========================================================="

# Ensure we are in the repository directory where sbatch was executed
cd "${SLURM_SUBMIT_DIR:-.}"

# 1. Initialize Git Submodules (Crucial for BigVGAN!)
echo "Ensuring Git submodules are initialized..."
git submodule update --init --recursive

# 2. Load the optimized Ibex Machine Learning Environment
echo "Loading Ibex machine_learning module..."
module purge
module load machine_learning/2024.01

# 3. Create and activate a 100% isolated local virtual environment
echo "Creating isolated local environment to prevent user conflicts..."
python -m venv ibex_tts_env --system-site-packages
source ibex_tts_env/bin/activate
pip install -r requirements.txt

# 4. Save HF Token if provided
if [ -n "$1" ]; then
    echo "Saving HF token to hf_config.json..."
    echo "{\"HF_TOKEN\": \"$1\"}" > hf_config.json
fi

# 5. Upload latest English weights to Hugging Face
echo "Uploading latest English checkpoint to Hugging Face..."
if [ -n "$1" ]; then
    python tools/upload_to_hf.py --lang english --token "$1"
else
    python tools/upload_to_hf.py --lang english
fi

echo "English upload job completed!"
