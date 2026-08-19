# Fusion-JEPA: Expressive, Low-Resource Text-to-Speech (TTS)

[![Demo](https://img.shields.io/badge/Demo-Interactive_Showcase-teal)](https://omara32.github.io/Fusion-JEPA-TTS/)
[![Paper](https://img.shields.io/badge/Paper-IEEE_Format-red)](https://omara32.github.io/Fusion-JEPA-TTS/reports/Fusion_JEPA_IEEE_Conference.pdf)
[![Report](https://img.shields.io/badge/Report-KAUST_Lab_Report-blue)](https://omara32.github.io/Fusion-JEPA-TTS/reports/Fusion_JEPA_KAUST_Report.pdf)

An open-source, dual-language Text-to-Speech (TTS) pipeline leveraging a Joint-Embedding Predictive Architecture (JEPA), Multimodal Diffusion Transformer (MM-DiT), and Continuous Flow Matching. This repository contains the complete infrastructure for training, inference, and cloud synchronization for both English and Arabic audio generation.

## Features
- **Dual-Language End-to-End TTS**: Full support for both Arabic and English phonetic modeling.
- **Multiple Database Integration**: Natively trains on `nawar_halabi`, `common_voice`, `clartts` (Arabic) and `ljspeech`, `libritts` (English).
- **Advanced Vocoders**: Uses NVIDIA's [BigVGAN](https://github.com/NVIDIA/BigVGAN) as the default high-fidelity vocoder, with fallback support for Vocos. 
- **Automated Supercomputer Pipeline**: Includes a fully-automated Kaggle/Colab Jupyter Notebook (`supercomputer_training.ipynb`) designed for distributed multi-GPU (DDP) training.
- **Cloud Checkpoint Syncing**: Built-in PyTorch Lightning callback automatically pushes updated weights to `KASP-JEPA` Hugging Face repositories seamlessly during training.

## Installation & Setup

### Supercomputer / Cloud (Kaggle/Colab)
For cloud platforms, simply upload and run `supercomputer_training.ipynb`. It automatically handles environment setup, multi-GPU configuration, dataset downloading, and Hugging Face authentication.

### Local Installation
To run the project locally, execute the setup scripts to automatically securely configure your Python virtual environment and install all dependencies (PyTorch, Lightning, Transformers, BigVGAN).

**Windows:**
Double-click `launchers/setup_env.bat` or `launchers/setup_env_xpu.bat`

**Linux/Mac:**
Run `bash launchers/setup_env.sh` or `bash launchers/setup_env_xpu.sh`

## Pipeline Usage

All core logic is contained within the root Python files. You can execute them directly, or use the convenient `launchers/` wrapper scripts to automatically hook into your virtual environment.

### 1. Training (`train.py`)
Trains the JEPA-T model. Automatically scales to multiple GPUs using PyTorch Lightning DDP strategy.
```bash
python train.py --lang english --db ljspeech --checkpointnum 2
```
*Arguments:*
- `--lang`: `english` or `arabic`
- `--db`: Dataset to use (must match language)
- `--resume`: Resumes from the last saved checkpoint
- `--val`: Enables validation (Note: turning validation off drastically saves disk space)
- `--checkpointnum`: Syncs `last.ckpt` to Hugging Face every N epochs

### 2. Inference (`inference.py`)
Generates audio from text using the latest trained model weights.
```bash
python inference.py --lang english --text "Hello world"
```
*Arguments:*
- `--lang`: Language model to load
- `--text`: The string of text to synthesize
- `--index`: Alternative to `--text`. Provide an integer index to grab the corresponding ground-truth transcript from the dataset for testing.
- `--db`: (Required if using `--index`) The dataset to read from.

### 3. Vocoder Testing (`test_vocoder_ground_truth.py`)
Extracts raw ground-truth audio from the dataset, converts it to mel-spectrograms, and pushes it through the BigVGAN vocoder to isolate and test the absolute maximum audio quality achievable by the vocoder alone (ignoring the JEPA model).
```bash
python test_vocoder_ground_truth.py --lang english --db ljspeech --index 25
```

### 4. Cloud Synchronization 
Manually interact with the Hugging Face hub outside of the training loop.
- `download_from_hf.py`: Downloads the latest pre-trained model weights.
- `upload_to_hf.py`: Manually pushes local weights to the cloud.

## Repository Structure
- `data/`: Contains `dataset.py` logic for downloading datasets, tokenizing text, and extracting mel-spectrograms.
- `launchers/`: Contains `.bat` and `.sh` wrapper scripts for all major operations.
- `BigVGAN/`: Git Submodule linked directly to NVIDIA's source code for mel-spectrogram extraction and audio reconstruction.
