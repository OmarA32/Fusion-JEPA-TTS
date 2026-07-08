# Audio-JEPA Arabic TTS

An open-source Text-to-Speech (TTS) pipeline for Arabic, leveraging a modified Joint-Embedding Predictive Architecture (JEPA-T).

## Features
- End-to-end Arabic TTS processing utilizing `MohamedRashad/common-voice-18-arabic` from HuggingFace.
- Multiple Vocoder support (HiFi-GAN, Vocos, BigVGAN).
- Robust PyTorch Lightning training and inference pipeline.
- Cross-platform automated environment handling.

## Automated Installation
No manual virtual environment configuration is needed.

**Windows:**
Double-click `setup_env.bat`

**Linux/Mac:**
Run `bash setup_env.sh`

This script will automatically create an isolated Python virtual environment (`venv`), securely install build tools, and download all core dependencies (PyTorch, PyTorch Lightning, HuggingFace Datasets, CLIP, timm) seamlessly.

## Usage
Simply execute any of the automated `.bat` or `.sh` scripts to test or train the models. They automatically connect to the internal virtual environment for you.

**Training:**
- `train_from_scratch_jepa.[bat/sh]`
- `train_finetune_jepa.[bat/sh]`

**Testing:**
- `test_one_text_hifigan.[bat/sh]`
- `test_one_text_vocos.[bat/sh]`
- `test_one_text_bigvgan.[bat/sh]`
