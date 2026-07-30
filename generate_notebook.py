import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# JEPA-TTS Supercomputer Pipeline\n",
                "This notebook automates the entire pipeline: downloading the fixed codebase, installing dependencies, training from scratch, and running inference/testing on the supercomputer."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Clone Codebase & Checkout Prototype Branch"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!git clone https://github.com/OmarA32/Audio-JEPA-Arabic-TTS.git\n",
                "%cd Audio-JEPA-Arabic-TTS\n",
                "!git checkout prototype/v6.0.0\n",
                "!git submodule update --init --recursive"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Install Dependencies\n",
                "Installs all libraries directly into the notebook kernel (including `vocos`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -r requirements.txt"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Clear Old Weights (Safety)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!rm -rf training_logs"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4A. Run Training (From Scratch)\\n",
                "This deletes any existing weights and trains from scratch at Epoch 0. The dataset (`MohamedRashad/common-voice-18-arabic`) is public and will download automatically during the first epoch.\n",
                "You can customize training with these optional arguments:\n",
                "- `--lang english` (to train the english dataset)\n",
                "- `--db ljspeech` (to switch database to LJSpeech or LibriTTS)\n",
                "- `--val` (to enable the validation loop for graphing)\n",
                "- `--freeze_jepa` (to freeze the ViT backbone and only train the Diffusion MLP)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python train.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4B. Resume Training / Fine-Tune\n",
                "If you downloaded your saved weights from Hugging Face into the `training_logs` folder, run this cell instead! It will seamlessly resume training from the latest epoch.\n",
                "You can also pass arguments here e.g., `!python train.py --resume --val --freeze_jepa`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python train.py --resume"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Generate Inference Audio\n",
                "Once training finishes (or if you manually stop the cell above after some epochs), run this cell to generate audio from text using your newly trained model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python inference.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Test Vocoder (Ground Truth Quality)\n",
                "If you want to test the raw quality of the vocoder against the dataset (without the neural network's influence), run this test."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python test_vocoder_ground_truth.py --vocoder vocos"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Test TTS model with new text.\n",
                "You can pass any custom text to generate here:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python inference.py --text \"أي نص عربي تريد\" --output \"my_custom_audio.wav\""
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8. Upload Weights to Hugging Face\n",
                "Since supercomputers delete data when shut down, run this cell to push your saved epochs (100, 200, etc.) to your HF account. \n",
                "First, get an Access Token from your Hugging Face settings (make sure it is a **WRITE** token)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from huggingface_hub import HfApi, login\n",
                "\n",
                "# 1. Login (Paste your token below)\n",
                "hf_token = \"YOUR_HUGGINGFACE_WRITE_TOKEN\" \n",
                "login(token=hf_token)\n",
                "\n",
                "# 2. Push the entire training_logs folder to a new or existing repository\n",
                "repo_id = \"your-username/JEPA-Arabic-TTS-Checkpoints\" # Change this!\n",
                "\n",
                "api = HfApi()\n",
                "api.create_repo(repo_id=repo_id, exist_ok=True)\n",
                "api.upload_folder(\n",
                "    folder_path=\"training_logs\",\n",
                "    repo_id=repo_id,\n",
                "    repo_type=\"model\",\n",
                "    commit_message=\"Bulk upload of all training epochs\"\n",
                ")\n",
                "print(\"Upload complete! All epochs are now safely stored in Hugging Face.\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(r"C:\Users\g3m43\.gemini\antigravity\scratch\JEPA-TTS v2\supercomputer_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
