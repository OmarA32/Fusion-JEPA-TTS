# Fusion-JEPA: Expressive, Low-Resource Text-to-Speech (TTS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-Interactive_Audio_Showcase-00A896?style=for-the-badge&logo=google-chrome&logoColor=white)](https://omara32.github.io/Fusion-JEPA-TTS/)
[![Research Paper](https://img.shields.io/badge/Paper-Research_Manuscript_(PDF)-E63946?style=for-the-badge&logo=adobe-acrobat-reader&logoColor=white)](https://omara32.github.io/Fusion-JEPA-TTS/reports/Fusion_JEPA_IEEE_Conference.pdf)
[![KAUST Report](https://img.shields.io/badge/Report-KAUST_Lab_Report_(14p)-1D3557?style=for-the-badge&logo=overleaf&logoColor=white)](https://omara32.github.io/Fusion-JEPA-TTS/reports/Fusion_JEPA_KAUST_Report.pdf)
[![Presentation](https://img.shields.io/badge/Slides-Beamer_Presentation-EE964B?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white)](https://omara32.github.io/Fusion-JEPA-TTS/reports/Fusion_JEPA_Presentation.pdf)

> **Fusion-JEPA** is an open-source, bilingual Text-to-Speech (TTS) architecture that decouples abstract phonetic representation learning from continuous acoustic distribution modeling via **Joint-Embedding Predictive Architecture (JEPA)**, **Multimodal Diffusion Transformer (MM-DiT)**, and **Continuous Flow Matching**.

---

## Key Innovations

1. **Self-Supervised JEPA Latent Space ($\mathcal{L}_p$):**  
   An Exponential Moving Average (EMA, $\alpha=0.9999$) Target Teacher network encodes full acoustic spectrograms to provide stable target representations for the Predictor.
2. **Continuous Conditional Flow Matching ($\mathcal{L}_v$):**  
   Replaces blurry pixel-level $L_1/L_2$ regression with vector velocity field modeling, eliminating robotic buzzing and preserving sharp high-frequency harmonic formants.
3. **Multimodal Diffusion Transformer (MM-DiT) with 1D RoPE:**  
   Concatenates discrete phonemic text tokens and continuous acoustic patches into unified self-attention layers with 1D Rotary Position Embeddings for length-invariant temporal conditioning.
4. **Studio-Grade 44.1 kHz Vocoding:**  
   Generates 128-band Mel-spectrograms natively aligned with BigVGAN v2 neural vocoding (anti-aliased periodic Snake activations) for broadcast-quality waveform synthesis.
5. **Low-Resource Modern Standard Arabic (MSA):**  
   Learns complex Arabic phonetics, geminate consonants (*Shaddah*), and emphatic sounds with $< 4$ hours of studio data.

---

## Architecture

<p align="center">
  <img src="assets/model_architecture.png" alt="Fusion-JEPA Architecture Diagram" width="92%">
</p>

---

## Interactive Audio Demos

Listen to side-by-side comparisons of **Ground Truth studio recordings** vs. **Fusion-JEPA generated speech** on our interactive GitHub Pages showcase:

<p align="center">
  <a href="https://omara32.github.io/Fusion-JEPA-TTS/">
    <img src="assets/demo_preview.png" alt="Fusion-JEPA Interactive Audio Showcase" width="92%">
  </a>
</p>

**[Explore Live Speech Showcase → https://omara32.github.io/Fusion-JEPA-TTS/](https://omara32.github.io/Fusion-JEPA-TTS/)**

---

## Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/OmarA32/Fusion-JEPA-TTS.git
cd Fusion-JEPA-TTS
```

**Windows:**
```bat
launchers\setup_env.bat
```

**Linux / Mac:**
```bash
bash launchers/setup_env.sh
```

---

## 💻 Web Interface, CLI & Launcher Reference

Fusion-JEPA provides an interactive Streamlit Web UI, Python pipelines, and categorized batch/shell launchers in `launchers/`.

### 1. Interactive Web Studio (`app.py` / `launchers/ui/`)

Launch the bilingual real-time studio interface with audio playback, spectrogram visualization, and dataset indexing:

```bash
# Windows:
launchers\ui\run_interface.bat

# Linux / Mac:
bash launchers/ui/run_interface.sh
```

---

### 2. Speech Synthesis (`pipelines/inference.py` / `launchers/inference/`)

Synthesizes high-fidelity 44.1 kHz audio from text prompts or dataset indices using BigVGAN v2 vocoding and adaptive silence truncation.

```bash
# Basic English Synthesis (with automatic silence trimming)
python pipelines/inference.py --lang english --text "This is Fusion JEPA text to speech synthesis."

# Arabic Synthesis with Mel-Spectrogram saving
python pipelines/inference.py --lang arabic --text "وَتَتَضَمَّنُ حَفَلَاتٍ لِمُوسِيقَى الْجَازِ" --save-mel

# Synthesize directly by dataset test index (LJSpeech or Nawar Halabi)
python pipelines/inference.py --lang english --db ljspeech --index 108 --save-mel

# Longform Multi-Paragraph Synthesis (Arbitrary Length)
python pipelines/longform_inference.py --lang arabic --text "..." --pause-ms 100

# Using Launcher Wrappers:
# Windows:
launchers\inference\inference.bat --lang english --text "Hello world"
# Linux/Mac:
bash launchers/inference/inference.sh --lang english --text "Hello world"
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--text` | `str` | Arabic Greeting | Custom text string or diacritized Arabic/English to synthesize |
| `--output`, `--file_name` | `str` | `output_test.wav` | Output WAV audio destination path or filename |
| `--lang` | `str` | `arabic` | Model language: `arabic` or `english` |
| `--db` | `str` | Auto | Dataset to fetch index from (only used with `--index`): `nawar_halabi`, `common_voice`, `clartts`, `ljspeech`, `libritts` |
| `--index` | `int` | `None` | Sample index to fetch ground-truth text and reference Mel from test split |
| `--ckpt` | `str` | `None` | Explicit path to a `.ckpt` (Lightning) or `.pt` checkpoint file |
| `--cfg-scale` | `float` | `7.0` | Classifier-Free Guidance scale ($1.0 = \text{unconditional}$) |
| `--steps` | `int` | `60` | Number of Euler ODE Flow Matching diffusion integration steps |
| `--save-mel` | `flag` | `False` | Save generated Mel-spectrogram comparison plot (`.png`) |
| `--no-trim` | `flag` | `False` | Disable automatic 4-stage post-speech silence and hallucination trimming |

---

### 3. Model Training (`training/train.py` / `launchers/training/`)

Full distributed training of the dual-objective Fusion-JEPA architecture ($\mathcal{L}_v + \mathcal{L}_p$) with PyTorch Lightning.

```bash
# Train on Arabic Speech Corpus (Nawar Halabi) with checkpoint syncing
python training/train.py --lang arabic --db nawar_halabi --epochs 2600 --checkpointnum 150 --hf_token "hf_..."

# Train on LJSpeech (English)
python training/train.py --lang english --db ljspeech --epochs 2600 --checkpointnum 80 --resume

# Using Launcher Wrappers:
# Windows:
launchers\training\train.bat --lang arabic --db nawar_halabi --resume
# Linux/Mac:
bash launchers/training/train.sh --lang arabic --db nawar_halabi --resume
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--lang` | `str` | `arabic` | Training language: `arabic` or `english` |
| `--db` | `str` | `nawar_halabi` | Dataset: `nawar_halabi`, `common_voice`, `clartts`, `ljspeech`, `libritts` |
| `--epochs` | `int` | `2600` | Maximum number of training epochs |
| `--batch_size` | `int` | `16` | Per-GPU batch size |
| `--lr` | `float` | `1e-4` | Learning rate for AdamW optimizer |
| `--resume` | `flag` | `False` | Automatically resume from latest checkpoint in `training_logs/<lang>/` |
| `--download_latest` | `flag` | `False` | Auto-download latest Hugging Face checkpoint if missing locally |
| `--val` | `flag` | `False` | Enable periodic validation evaluation loops |
| `--freeze_jepa` | `flag` | `False` | Freeze ViT encoder backbone and train only Flow Matching MM-DiT |
| `--freeze_diffuser` | `flag` | `False` | Freeze SpatialDiT diffuser and train only JEPA backbone |
| `--checkpointnum` | `int` | `0` | Epoch interval for automatic model checkpoint upload to Hugging Face |
| `--hf_token` | `str` | `None` | Hugging Face access token for automated model synchronization |

---

### 4. Single-Sample Overfitting Verification (`training/overfit_train.py` & `pipelines/overfit_inference.py`)

Verifies architecture convergence, flow velocity alignment, and eliminates representation collapse on a single sample:

```bash
# 1. Overfit Train on English LJSpeech Sample (Index 108)
python training/overfit_train.py --lang english --db ljspeech --index 108 --epochs 5000

# 2. Overfit Train on Arabic Nawar Halabi Sample (Index 107)
python training/overfit_train.py --lang arabic --db nawar_halabi --index 107 --epochs 5000

# 3. Evaluate Overfit Synthesis & Compare Spectrograms
python pipelines/overfit_inference.py --lang english --db ljspeech --index 108 --save-mel
python pipelines/overfit_inference.py --lang arabic --db nawar_halabi --index 107 --save-mel
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--index` | `int` | `108` / `107` | Fixed dataset sample index to overfit |
| `--epochs` | `int` | `5000` | Total overfitting epochs |
| `--lr` | `float` | `1e-4` | Learning rate |
| `--save-mel` | `flag` | `False` | Render and save side-by-side ground truth vs generated spectrograms |

---

### 5. Hugging Face Checkpoint Sync (`tools/download_from_hf.py` & `tools/upload_to_hf.py`)

```bash
# Download latest English or Arabic model weights
python tools/download_from_hf.py --lang english
python tools/download_from_hf.py --lang arabic

# Upload local checkpoint
python tools/upload_to_hf.py --lang arabic --token "YOUR_HF_TOKEN"
```

---

## Repository Structure

```text
├── app.py                      # Interactive Streamlit Web Studio
├── supercomputer_training.ipynb# Distributed HPC, Colab & Kaggle Training Notebook
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
├── LICENSE                     # MIT License
│
├── pipelines/                  # Inference & Speech Generation Engines
│   ├── inference.py            # Single-sentence synthesis CLI
│   ├── longform_inference.py   # Multi-paragraph chunking & cross-fading
│   └── overfit_inference.py    # Overfit evaluation script
│
├── training/                   # Model Training Scripts
│   ├── train.py                # Multi-GPU training script (PyTorch Lightning)
│   ├── train_xpu.py            # Intel XPU accelerator training script
│   └── overfit_train.py        # Single-sample convergence tester
│
├── tools/                      # Cloud Synchronization & Evaluation Tools
│   ├── download_from_hf.py     # Checkpoint downloader
│   ├── upload_to_hf.py         # Checkpoint uploader
│   ├── evaluate_metrics.py     # Objective audio quality benchmarks (RTF, MCD)
│   └── test_vocoder_ground_truth.py # BigVGAN fidelity verification
│
├── launchers/                  # Categorized Environment & Execution Launchers
│   ├── ui/                     # 1-Click Web Studio Launchers (.bat, .sh)
│   ├── inference/              # Inference CLI Wrappers
│   ├── training/               # Local Training Wrappers
│   ├── tools/                  # Hugging Face & Vocoder Test Wrappers
│   ├── setup/                  # Environment Installation Wrappers
│   └── hpc_ibex/               # Slurm Supercomputer Scripts
│
├── models/                     # Fusion-JEPA Neural Implementations
│   ├── block.py                # MM-DiT & 1D RoPE Attention Blocks
│   ├── jepa.py                 # Core Fusion-JEPA Architecture
│   ├── jepa_lightning.py       # PyTorch Lightning Module with Lv + Lp loss
│   └── vocoder_manager.py      # BigVGAN Vocoder Manager & Wrapper
│
├── data/                       # Dataset Loaders & Speech Corpora
├── text/                       # Multilingual G2P & Phonetic Tokenizers
└── BigVGAN/                    # NVIDIA BigVGAN v2 Vocoder (44.1kHz Universal)
```

---

## Authors

- **Omar Alkhammash** — King Khalid University / KAUST Academy
- **Abdulrahman Soliman** — KAUST Academy
- **Hassan Alahmed** — KAUST Academy
- **Mentor:** **Dr. Kerven Durdymyradov** — KAUST Academy AI Program

*Developed as part of the **KAUST Academy Artificial Intelligence Program** (August 2026).*

---

## Acknowledgments & Prior Work

We gratefully acknowledge the foundational open-source models, reference codebases, and datasets that made this work possible:

- **D-JEPA Codebase & Formulation:** [D-JEPA/djepa-imagenet](https://github.com/D-JEPA/djepa-imagenet) (Hao Chen et al., *D-JEPA: Denoising with a Joint-Embedding Predictive Architecture*, 2024) — which established the generative JEPA paradigm with dual latent prediction and flow/diffusion objectives.
- **JEPA-T Architecture:** [justin-herry/JEPA-T](https://github.com/justin-herry/JEPA-T) (Siheng Wan, Jifeng Shen, Justin Herry et al., *JEPA-T: Joint-Embedding Predictive Architecture with Text Fusion*, 2025) — which provided the core architectural foundations for text-conditioned multimodal transformer prediction.
- **JEPA Theory & Self-Supervised Learning:** Yann LeCun (*A Path Towards Autonomous Machine Intelligence*, 2022) and Meta AI (*I-JEPA / V-JEPA*, Assran et al., 2023; Bardes et al., 2024).
- **Continuous Flow Matching:** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matthew Le (*Flow Matching for Generative Modeling*, ICLR 2023).
- **BigVGAN Neural Vocoder:** NVIDIA Corporation ([NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN), Sang-gil Lee et al., *BigVGAN-v2: Universal Neural Vocoder with Anti-Aliased Snake Activations*, 2024).
- **Arabic Speech Corpus:** Dr. Nawar Halabi ([Arabic Speech Corpus](http://en.arabicspeechcorpus.com/), University of Southampton, 2016).
- **English Benchmark:** Keith Ito and Linda Johnson ([The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/), 2017).

---

## Publications & Documentation

- **[Research Manuscript (PDF)](https://omara32.github.io/Fusion-JEPA-TTS/reports/Fusion_JEPA_IEEE_Conference.pdf)**: 6-page IEEE-format conference paper with mathematical derivations and benchmark evaluations.
- **[KAUST Lab Report (PDF)](https://omara32.github.io/Fusion-JEPA-TTS/reports/Fusion_JEPA_KAUST_Report.pdf)**: 14-page comprehensive technical report with implementation details and Slurm recipes.
- **[Presentation Slides (PDF)](https://omara32.github.io/Fusion-JEPA-TTS/reports/Fusion_JEPA_Presentation.pdf)**: 15-slide defense presentation.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
