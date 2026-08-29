# 🎓 Fusion-JEPA: Complete Master Study Guide & Presentation Companion
**Bilingual Generative Speech Synthesis via Decoupled Multimodal Joint-Embedding Predictive Architectures and Continuous Flow Matching**

*Authoritative Reference Manual for Project Presentation, Oral Defense, and Poster Exhibition*
*KAUST Academy AI Program — August 2026*

---

# Table of Contents
1. [Chapter 1: Comprehensive Glossary of Terms & Core Definitions](#chapter-1-comprehensive-glossary-of-terms--core-definitions)
   - [1.1 Self-Supervised Learning & JEPA Terminology](#11-self-supervised-learning--jepa-terminology)
   - [1.2 Generative Flow Matching & Diffusion Physics](#12-generative-flow-matching--diffusion-physics)
   - [1.3 Transformer & Attention Mechanics](#13-transformer--attention-mechanics)
   - [1.4 Audio, DSP & Mel-Spectrogram Terminology](#14-audio-dsp--mel-spectrogram-terminology)
   - [1.5 Neural Vocoding & BigVGAN v2 Architecture](#15-neural-vocoding--bigvgan-v2-architecture)
   - [1.6 Linguistics & Bilingual Phonetization (Arabic & English)](#16-linguistics--bilingual-phonetization-arabic--english)
   - [1.7 Inference, Sampling & Post-Processing Terminology](#17-inference-sampling--post-processing-terminology)
2. [Chapter 2: Deep Architectural Blueprint, Training Dynamics & Inference Mechanics](#chapter-2-deep-architectural-blueprint-training-dynamics--inference-mechanics)
   - [2.1 Core Problem Formulation & The Decoupled Paradigm](#21-core-problem-formulation--the-decoupled-paradigm)
   - [2.2 Step-by-Step Tensor Dimensionality & Data Flow](#22-step-by-step-tensor-dimensionality--data-flow)
   - [2.3 The 1D RoPE (Rotary Position Embedding) Deep Dive](#23-the-1d-rope-rotary-position-embedding-deep-dive)
   - [2.4 Training Mechanics & The Dual-Loss Objective](#24-training-mechanics--the-dual-loss-objective)
   - [2.5 Inference Pipeline & Mathematical ODE Flow Integration](#25-inference-pipeline--mathematical-ode-flow-integration)
   - [2.6 Acoustic Boundary Detection & Adaptive Silence Elimination](#26-acoustic-boundary-detection--adaptive-silence-elimination)
3. [Chapter 3: Defense & Poster Day Q&A Bank](#chapter-3-defense--poster-day-qa-bank)
   - [3.1 High-Priority Architectural & Theoretical Questions](#31-high-priority-architectural--theoretical-questions)
   - [3.2 Comparative Analysis vs Prior SOTA (FastPitch, VITS, F5-TTS)](#32-comparative-analysis-vs-prior-sota-fastpitch-vits-f5-tts)
   - [3.3 Low-Resource Arabic Challenges & Linguistic Defenses](#33-low-resource-arabic-challenges--linguistic-defenses)

---

# Chapter 1: Comprehensive Glossary of Terms & Core Definitions

This chapter provides rigorous definitions, mathematical intuitions, and real-world project context for every technical term employed in Fusion-JEPA.

---

### 1.1 Self-Supervised Learning & JEPA Terminology

#### 🔹 Joint-Embedding Predictive Architecture (JEPA)
A non-generative self-supervised learning paradigm introduced by Yann LeCun. Instead of predicting raw, high-dimensional pixels or waveforms (which waste network capacity predicting irrelevant micro-noise), a JEPA predicts representations in an abstract, semantic latent space:
$$\hat{s}_y = \text{Predictor}(s_x, z)$$
In our system, the Student Predictor learns to predict the semantic acoustic latent representations produced by the Target Teacher, forcing the model to understand phonetic-acoustic structure before generating audio.

#### 🔹 Target Teacher Encoder
The branch of the JEPA network that observes complete, unmasked Mel-spectrogram inputs and maps them into target latent representations. The teacher's weights are not updated via gradient descent; instead, they are updated smoothly using an Exponential Moving Average (EMA) of the student's weights.

#### 🔹 Context / Student Encoder
The active branch of the JEPA network that observes masked or text-conditioned representations and is trained via backpropagation to predict the teacher's latent embeddings.

#### 🔹 Exponential Moving Average (EMA) Update Rule
A temporal momentum smoothing technique used to stabilize the teacher network. At each training step $k$, teacher weights $\theta_{\text{target}}$ are updated according to:
$$\theta_{\text{target}}^{(k)} \leftarrow \alpha \theta_{\text{target}}^{(k-1)} + (1 - \alpha) \theta_{\text{student}}^{(k)}$$
In Fusion-JEPA, we set $\alpha = 0.9999$. This high momentum prevents the teacher from drifting rapidly, ensuring consistent learning targets for the student.

#### 🔹 Stop-Gradient Operator ($\text{sg}[\cdot]$)
A mathematical operation that treats its argument as a constant during backpropagation. In JEPA:
$$\mathcal{L}_p = \mathcal{D}\left(Z_{\text{student}}, \text{sg}[Z_{\text{teacher}}]\right)$$
Gradients flow strictly into the student branch, preventing the teacher from collapsing towards trivial solutions.

#### 🔹 Representation Collapse
A pathological failure mode in self-supervised learning where the encoder maps all inputs to a single constant vector or trivial subspace (e.g., $f(x) = \mathbf{0}$). JEPA eliminates collapse through asymmetric architectures, EMA updates, and stop-gradient operations without requiring negative samples or contrastive pairs.

---

### 1.2 Generative Flow Matching & Diffusion Physics

#### 🔹 Continuous Flow Matching (CFM)
A simulation-free generative paradigm that trains continuous vector fields to transport a simple base distribution (e.g., standard Gaussian noise $p_0 = \mathcal{N}(0, \mathbf{I})$) to a complex target data distribution $p_1$ (e.g., natural speech Mel-spectrograms).

#### 🔹 Probability Density Path ($p_t$)
A time-dependent family of probability distributions that interpolates smoothly between noise at $t = 0$ and clean data at $t = 1$.

#### 🔹 Optimal Transport (OT) Displacement Interpolant
The straight-line path connecting a noise sample $x_0 \sim \mathcal{N}(0, \mathbf{I})$ and a clean target Mel sample $x_1 \sim q(x)$:
$$x_t = (1 - t)x_0 + t x_1, \quad t \in [0, 1]$$
The constant target velocity along this path is simply the time derivative:
$$u_t(x_t | x_0, x_1) = \frac{d x_t}{dt} = x_1 - x_0$$

#### 🔹 Vector Velocity Field ($v_\theta(x_t, t, c)$)
The neural network parameterized by $\theta$ that takes noisy state $x_t$, timestep $t$, and conditioning context $c$ (text + JEPA latents) to predict the direction and speed pointing toward clean speech.

#### 🔹 Velocity Matching Loss ($\mathcal{L}_v$)
The mean squared error objective used to train the Flow Matching vector field:
$$\mathcal{L}_v(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1], \, x_0 \sim \mathcal{N}(0, \mathbf{I}), \, x_1 \sim q(x)}\left[ \| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2 \right]$$

#### 🔹 Ordinary Differential Equation (ODE) Sampling
During inference, speech generation is formulated as solving the initial value problem:
$$\frac{dx}{dt} = v_\theta(x_t, t, c), \quad x(0) \sim \mathcal{N}(0, \mathbf{I})$$
Solving this ODE from $t = 0$ to $t = 1$ reconstructs the clean Mel-spectrogram $x(1)$.

#### 🔹 Euler Integration Method
The first-order numerical ODE solver used for fast inference over $N$ discrete steps with step size $\Delta t = \frac{1}{N}$:
$$x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c)$$
In Fusion-JEPA, $N = 60$ steps provides high acoustic fidelity.

---

### 1.3 Transformer & Attention Mechanics

#### 🔹 Multimodal Diffusion Transformer (MM-DiT)
An advanced DiT architecture featuring dual-stream attention blocks where text tokens and acoustic patch tokens maintain dedicated representations while exchanging cross-modal information through a shared attention matrix.

#### 🔹 Patchification & Unpatchification
* **Patchify:** Slicing a continuous 2D Mel-spectrogram $[1, 128, 512]$ into non-overlapping $16 \times 16$ square patches, flattening each into a $256$-dimensional vector, resulting in an $8 \times 32 = 256$ token sequence.
* **Unpatchify:** The reverse operation that rearranges $256$ token vectors back into the original $[1, 128, 512]$ 2D frequency-time grid.

#### 🔹 1D Rotary Position Embedding (1D RoPE)
A position encoding method that injects relative positional information by rotating Query and Key vectors in the 2D complex plane:
$$\mathbf{q}_m^{(i)} = \mathbf{R}_{\Theta, m}^{(i)} \mathbf{q}^{(i)}, \quad \mathbf{k}_n^{(i)} = \mathbf{R}_{\Theta, n}^{(i)} \mathbf{k}^{(i)}$$
Because $\langle \mathbf{q}_m, \mathbf{k}_n \rangle$ depends only on the relative offset $(m - n)$, RoPE enables sequence length extrapolation without requiring learned positional lookup tables.

#### 🔹 Adaptive Layer Normalization (AdaLN-Zero)
A conditioning mechanism where timestep embeddings modulate transformer layer activations via scale ($\gamma$), shift ($\beta$), and gating ($\alpha$) parameters initialized to zero:
$$\mathbf{h}' = \mathbf{h} + \alpha \cdot \text{Block}\left( (1 + \gamma) \odot \text{LayerNorm}(\mathbf{h}) + \beta \right)$$
Zero initialization ensures the network begins training as an identity mapping, stabilizing deep diffusion convergence.

---

### 1.4 Audio, DSP & Mel-Spectrogram Terminology

#### 🔹 Mel-Spectrogram
A time-frequency visual representation of sound where frequency bands are warped onto the non-linear Mel scale to match human auditory perception (higher resolution at low frequencies, lower resolution at high frequencies).

#### 🔹 Short-Time Fourier Transform (STFT)
The mathematical transform that segments continuous audio waveforms into overlapping windowed frames and calculates the discrete Fourier transform for each frame:
$$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j \omega n}$$

#### 🔹 Key Audio Parameters in Fusion-JEPA:
* **Sampling Rate ($f_s$):** $44,100\text{ Hz}$ ($44.1\text{ kHz}$) — Studio-grade audio capturing frequencies up to the Nyquist limit of $22.05\text{ kHz}$.
* **FFT / Window Size ($N_{\text{fft}}$):** $2048\text{ samples}$ ($\approx 46.4\text{ ms}$) — Determines spectral frequency resolution.
* **Hop Size ($H$):** $512\text{ samples}$ ($\approx 11.6\text{ ms}$) — Frame step size. At $44.1\text{ kHz}$, 512 frames equal $5.944\text{ seconds}$ of audio ($512 \times 512 = 262,144\text{ samples}$).
* **Mel Filterbank Bins ($M$):** $128\text{ frequency bands}$ (spanning $0\text{ Hz}$ to $22,050\text{ Hz}$).

---

### 1.5 Neural Vocoding & BigVGAN v2 Architecture

#### 🔹 Neural Vocoder
A deep neural network that inverts lossy 2D Mel-spectrograms back into time-domain 1D acoustic pressure waveforms $x(t)$.

#### 🔹 BigVGAN v2
A state-of-the-art GAN-based universal neural vocoder developed by NVIDIA, trained on large multi-speaker corpuses to synthesize high-frequency harmonics and eliminate robotic artifacts.

#### 🔹 Anti-Aliased Periodic Snake Activation
A specialized periodic activation function designed for raw audio synthesis:
$$\text{Snake}_\alpha(x) = x + \frac{1}{\alpha} \sin^2(\alpha x) = x + \frac{1 - \cos(2\alpha x)}{2\alpha}$$
The learnable parameter $\alpha$ controls periodic frequency scaling, providing inductive bias for modeling harmonic audio waveforms while anti-aliasing low-pass filters prevent high-frequency distortion.

---

### 1.6 Linguistics & Bilingual Phonetization (Arabic & English)

#### 🔹 Grapheme vs. Phoneme
* **Grapheme:** The smallest unit of written text (e.g., letters like 'b', 'ت', 'k').
* **Phoneme:** The smallest distinct unit of speech sound (e.g., ARPAbet `/B/`, `/T/`, `/K/`).

#### 🔹 G2P (Grapheme-to-Phoneme Converter)
An automated pipeline that translates orthographic text into phonemic pronunciation sequences:
* **English:** Uses `g2p_en` based on the Carnegie Mellon University Pronouncing Dictionary (CMUdict) with 39 ARPAbet phonemes and lexical stress markers (`0`, `1`, `2`).
* **Arabic:** Uses phonetic transliteration and rule-based phonetization supporting Modern Standard Arabic (MSA) diacritics.

#### 🔹 Modern Standard Arabic (MSA) Specifics:
* **Tashkeel / Harakat (Diacritics):** Short vowels (*Fatha* `/a/`, *Damma* `/u/`, *Kasra* `/i/`, *Sukun* `//`, *Tanwin* `/an, un, in/`). Diacritization is essential because unvocalized Arabic lacks explicit short vowel letters.
* **Shaddah (Geminates):** Consonant doubling diacritic ($^\sim$). In our tokenizer, geminated consonants are handled via a dedicated `_dbl_` doubling token to sustain acoustic energy.
* **Emphatic Consonants:** Velarized pharyngeal consonants ($ص, ض, ط, ظ$) that lower adjacent vowel formants ($F_2$).

#### 🔹 Special Token Vocabulary in Fusion-JEPA:
* `_pad_` (ID: 0): Padding token for batch alignment.
* `_eos_` (ID: 1): End-of-Sequence delimiter.
* `_sil_` (ID: 2): Explicit acoustic pause/silence.
* `_dbl_` (ID: 3): Arabic Shaddah consonant multiplier.
* `_+_` (ID: 4): Inter-word boundary whitespace separator.

---

### 1.7 Inference, Sampling & Post-Processing Terminology

#### 🔹 Classifier-Free Guidance (CFG)
A sampling technique that improves adherence to conditioning text by interpolating between unconditional ($\varnothing$) and conditioned ($c$) vector field estimates:
$$\tilde{v}_\theta(x_t, t, c) = v_\theta(x_t, t, \varnothing) + w \cdot \left( v_\theta(x_t, t, c) - v_\theta(x_t, t, \varnothing) \right)$$
In Fusion-JEPA, setting guidance scale $w = 7.0$ sharpens acoustic formant boundaries and eliminates muffled speech.

#### 🔹 4-Stage Adaptive Speech Boundary Truncator
An automated post-processing algorithm that detects where active human speech ends and trailing canvas silence begins:
1. **Dynamic Dynamic Range Gating:** Calculates acoustic noise floor across linear Mel energy.
2. **Adaptive Energy Thresholding:** Identifies speech cessation point $t_{\text{end}}$.
3. **Natural Acoustic Ring-Out Decay:** Appends $+80\text{ms}$ buffer to preserve natural room reverberation.
4. **Anti-Click Raised-Cosine Fade-Out:** Applies a gentle $10\text{ms}$ smoothing curve to prevent speaker popping.

---

# Chapter 2: Deep Architectural Blueprint, Training Dynamics & Inference Mechanics

---

### 2.1 Core Problem Formulation & The Decoupled Paradigm

Traditional single-stage TTS models suffer from **Representation Entanglement**:
* When a model is trained directly from text to Mel-spectrograms using pixel regression ($L_1$ or $L_2$ loss), it tries to predict the average of all possible acoustic realizations, causing **Spectral Oversmoothing** (robotic, muffled voice).
* In low-resource scenarios ($< 4\text{ hours}$ of data), standard diffusion models collapse because learning abstract phonetics and high-resolution spectrogram synthesis simultaneously requires massive datasets.

#### 💡 The Fusion-JEPA Solution:
Fusion-JEPA decouples the problem into two complementary sub-tasks:
1. **Phonetic Latent Representation Learning (JEPA):** Learns semantic alignment in a smooth latent space $\mathcal{L}_p$ free from pixel noise.
2. **Acoustic Flow Generation (CFM):** Operates on vector velocity fields $v_\theta$ to generate crisp 128-band Mel-spectrograms from noise.

```
       ┌───────────────────────────────┐
       │   Phonetic Text Embedding     │
       └──────────────┬────────────────┘
                      │  Cross-Modal Conditioning (Z)
                      ▼
 ┌─────────────┐   ┌───────────────────────────┐   ┌─────────────────────────┐
 │ Pure Noise  ├──►│ MM-DiT Flow Matching ODE  ├──►│ Clean Mel-Spectrogram   │
 │   x(0) ~ N  │   │   (N = 60 Euler Steps)    │   │      [128 x 512]        │
 └─────────────┘   └───────────────────────────┘   └────────────┬────────────┘
                                                                │
                                                                ▼
                                                   ┌─────────────────────────┐
                                                   │  BigVGAN v2 Vocoder     │
                                                   │   (44.1 kHz Waveform)   │
                                                   └─────────────────────────┘
```

---

### 2.2 Step-by-Step Tensor Dimensionality & Data Flow

Below is the complete tensor transformation pipeline through the architecture:

| Stage | Operation / Module | Input Tensor Shape | Output Tensor Shape | Mathematical Operation / Description |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Raw Audio Loading | Discrete samples | $[B, 1, 262144]$ | $5.944\text{s}$ at $44.1\text{kHz}$ |
| **2** | STFT + Mel Filterbank | Audio Waveform | $[B, 1, 128, 512]$ | $128\text{ Mel bins}, 512\text{ frames}, H=512, N_{\text{fft}}=2048$ |
| **3** | Mel Patchification | Mel-Spectrogram | $[B, 256, 256]$ | Grid of $8 \text{ rows} \times 32 \text{ cols} = 256 \text{ patches}$ ($16 \times 16 = 256\text{ dims}$) |
| **4** | Linear Patch Projection | Patch Vectors | $[B, 256, 768]$ | Linear projection to transformer hidden dim $D=768$ |
| **5** | Text Phonetization & Embedding | Text string $\to$ Token IDs | $[B, L_{\text{text}}, 768]$ | Phoneme vocab size $V=100$, lookup table to $D=768$ |
| **6** | Joint Dual Sequence | Concat $[Z_{\text{mel}}, Z_{\text{text}}]$ | $[B, 256 + L_{\text{text}}, 768]$ | Combined multimodal sequence fed to MM-DiT backbone |
| **7** | Target Teacher Latents | Complete Mel Patches | $[B, 256, 768]$ | Evaluated via EMA Teacher network $\text{sg}[E_{\text{EMA}}(X)]$ |
| **8** | Student JEPA Prediction | Masked / Conditioned sequence | $[B, 256, 768]$ | Student predictor output matched via $\mathcal{L}_p$ |
| **9** | Flow Matching Velocity Head | Noisy patches $x_t \in [B, 256, 256]$ | $[B, 256, 256]$ | Vector velocity field $v_\theta(x_t, t, Z)$ |
| **10**| Unpatchify | Reconstructed patches | $[B, 1, 128, 512]$ | Fold $256$ patches of $16 \times 16$ back into 2D grid |
| **11**| BigVGAN Vocoder | Inverted Mel-Spectrogram | $[B, 262144]$ | Upsampled $\times 2 \times 2 \times 8 \times 8 = \times 512$ via Snake activations |

---

### 2.3 The 1D RoPE (Rotary Position Embedding) Deep Dive

Unlike vision transformers that use 2D learned spatial grids, Fusion-JEPA employs **1D Rotary Position Embeddings (1D RoPE)** across both text and acoustic tokens.

#### Mathematical Formulation:
For a 2D subspace of the attention head with rotation angle $\theta_i = 10000^{-2(i-1)/d}$, the Query vector at sequence position $m$ is transformed by orthogonal rotation matrix $\mathbf{R}_m$:
$$\mathbf{R}_m = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$
When computing the attention dot-product between Query at position $m$ and Key at position $n$:
$$\mathbf{q}_m^\top \mathbf{k}_n = (\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_m^\top \mathbf{R}_n \mathbf{k} = \mathbf{q}^\top \mathbf{R}_{n - m} \mathbf{k}$$

#### 🌟 Why is 1D RoPE Crucial for Speech?
1. **Relative Temporal Distance:** Attention weights naturally decay as the temporal distance $|m - n|$ between phonemes increases.
2. **Length Invariance & Extrapolation:** The model can synthesize arbitrary-length sentences without encountering out-of-distribution positional indices.

---

### 2.4 Training Mechanics & The Dual-Loss Objective

During training, every batch undergoes simultaneous optimization of both representation learning and flow matching.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Ground Truth Mel [128, 512]            │
                  └───────────┬────────────────────────────────┬───────────┘
                              │                                │
                 (Clean Patches)                  (Noise Interpolation)
                              ▼                                ▼
                  ┌──────────────────────┐         ┌───────────────────────┐
                  │ Target Teacher (EMA) │         │ Noisy Sample: x(t)    │
                  └───────────┬──────────┘         │ x(t) = (1-t)x0 + t x1 │
                              │                    └───────────┬───────────┘
                     Latent Target Z_target                    │
                              │                                │
  ┌──────────────┐            │                                │
  │ Input Text   ├────────────┼────────────────►┌──────────────┴──────────┐
  └──────────────┘            │                 │   MM-DiT Student Model  │
                              │                 └──────────────┬──────────┘
                              │                                │
                              ▼                                ▼
                     ┌──────────────────┐             ┌───────────────────┐
                     │   L_p Latent     │             │    L_v Velocity   │
                     │  Prediction Loss │             │   Matching Loss   │
                     └────────┬─────────┘             └────────┬──────────┘
                              │                                │
                              └────────────────┬───────────────┘
                                               ▼
                                      Total Loss: L_v + L_p
```

#### 1. Latent Prediction Loss ($\mathcal{L}_p$):
$$\mathcal{L}_p = \frac{1}{K} \sum_{i=1}^K \text{Smooth}_{L_1}\left( Z_{\text{student}}^{(i)}, \, \text{sg}\left[Z_{\text{teacher}}^{(i)}\right] \right)$$
Where $\text{Smooth}_{L_1}(u) = 0.5 u^2$ for $|u| < 1$, and $|u| - 0.5$ otherwise.

#### 2. Flow Matching Velocity Loss ($\mathcal{L}_v$):
Sample random timestep $t \sim \mathcal{U}[0, 1]$ and noise $x_0 \sim \mathcal{N}(0, \mathbf{I})$. Compute target displacement $u_t = x_1 - x_0$.
$$\mathcal{L}_v = \frac{1}{D_{\text{patch}}} \sum_{j=1}^{256} \| v_\theta(x_t, t, Z_{\text{student}}) - (x_1 - x_0) \|^2$$

#### 3. Total Loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_v + \mathcal{L}_p$$

---

### 2.5 Inference Pipeline & Mathematical ODE Flow Integration

At inference time, the Target Teacher is completely discarded. The generation follows a deterministic 5-stage algorithm:

```
[ Input Text ]
     │
     ▼
[ Phonetizer & Tokenizer ] ──► Token IDs [1, L_text]
     │
     ▼
[ MM-DiT Context Forward ] ──► Conditioning Latents Z [1, 256, 768] (with 100% masked canvas)
     │
     ▼
[ Euler ODE Solver ] ◄──────── Gaussian Noise x(0) ~ N(0, I) [1, 256, 256]
     │
     │  For step k = 0 ... 59 (N = 60 steps, dt = 1/60):
     │    1. t = k / 60
     │    2. v_cond = Model(x_t, t, Z)
     │    3. v_uncond = Model(x_t, t, ZeroContext)
     │    4. v_guided = v_uncond + 7.0 * (v_cond - v_uncond)  <-- CFG Guidance
     │    5. x_{t+dt} = x_t + dt * v_guided
     ▼
[ Unpatchify ] ──────────────► Reconstructed Mel-Spectrogram [1, 128, 512]
     │
     ▼
[ BigVGAN Vocoder ] ─────────► High-Fidelity 44.1 kHz Waveform [1, 262144]
     │
     ▼
[ 4-Stage Adaptive Truncator ]► Clean Final Speech WAV
```

---

### 2.6 Acoustic Boundary Detection & Adaptive Silence Elimination

Because the MM-DiT operates on a fixed canvas of 512 frames ($5.94\text{s}$), shorter utterances finish before the canvas ends. If unmanaged, flow matching can hallucinate low-level background breathing noise in empty canvas regions.

Our **4-Stage Adaptive Speech Boundary Truncator** eliminates trailing artifacts:

$$\text{Energy}(f) = \frac{1}{128} \sum_{m=1}^{128} 10^{\frac{\text{Mel}(m, f)}{20}}$$

1. **Noise Floor Calibration:** Calculates median background energy $E_{\text{floor}}$ across the final 30 frames.
2. **Threshold Scan:** Locates the final active speech frame $f_{\text{speech}}$ where $\text{Energy}(f) > 2.5 \times E_{\text{floor}}$.
3. **Reverberation Ring-Out Buffer:** Adds $80\text{ms}$ ($\approx 7\text{ frames}$) after $f_{\text{speech}}$ to avoid cutting natural acoustic decay.
4. **Anti-Click Cosine Window:** Multiplies the final $10\text{ms}$ of samples by $w[n] = 0.5 \left(1 + \cos\left(\frac{\pi n}{M}\right)\right)$, ensuring zero DC offset and preventing speaker pops.

---

# Chapter 3: Defense & Poster Day Q&A Bank

*(Structured Q&A framework ready for defense rehearsals and reviewer questions)*

---

### 3.1 High-Priority Architectural & Theoretical Questions

#### Q1: "Why did you use Flow Matching instead of standard Diffusion (DDPM / Score-based SDE)?"
> **Answer:** Standard DDPM relies on curved Brownian motion paths, requiring hundreds of small denoising steps ($N = 250 - 1000$) or complex stochastic sampling schedules. Continuous Flow Matching (CFM) with Optimal Transport creates **straight-line probability trajectories** between noise and data ($x_t = (1-t)x_0 + tx_1$). Because the trajectories are straight, a simple first-order Euler ODE solver can traverse the path in only **$N = 60$ steps** with minimal discretization error, achieving faster inference and sharper acoustic formants.

#### Q2: "What is the exact purpose of the JEPA component if Flow Matching is already generating the Mel-spectrogram?"
> **Answer:** Flow Matching models are powerful density estimators, but in low-resource settings ($< 4\text{ hours}$ of speech), they struggle to learn robust phonetic-acoustic alignments from scratch. The JEPA objective acts as a self-supervised regularizer: it forces the encoder to predict invariant semantic representations of phonemes in an abstract latent space $\mathcal{L}_p$ before velocity field generation. This prevents representation collapse and stabilizes multi-modal attention.

#### Q3: "How does the Exponential Moving Average (EMA) prevent representation collapse in the JEPA teacher?"
> **Answer:** In non-contrastive self-supervised architectures, if the teacher and student are updated simultaneously via gradient descent, they quickly learn to output a constant vector ($Z = \mathbf{0}$) to minimize loss trivially. The EMA teacher ($\alpha = 0.9999$) updates on a much slower timescale and gradients are blocked via $\text{sg}[\cdot]$. The teacher provides a moving, non-stationary target that prevents the student from converging to trivial constants.

---

### 3.2 Comparative Analysis vs Prior SOTA (FastPitch, VITS, F5-TTS)

#### Q4: "How does Fusion-JEPA compare against FastPitch and VITS?"
> **Answer:**
> * **vs. FastPitch:** FastPitch relies on explicit duration and pitch predictors trained with pixel-level $L_1$ loss, leading to oversmoothed spectra and robotic buzz. Fusion-JEPA eliminates explicit duration/pitch predictors, allowing natural prosodic pitch curves to emerge through continuous vector field integration.
> * **vs. VITS:** VITS uses complex Monotonic Alignment Search (MAS) and normalizing flows with adversarial training, which is notoriously unstable to train on small single-speaker datasets. Fusion-JEPA is simulation-free, highly stable, and synthesizes full $44.1\text{ kHz}$ studio audio directly with BigVGAN v2.

---

### 3.3 Low-Resource Arabic Challenges & Linguistic Defenses

#### Q5: "How does Fusion-JEPA handle unvocalized Arabic text and geminate consonants (Shaddah)?"
> **Answer:** Unvocalized Arabic creates severe phonetic ambiguity because short vowels are omitted in everyday writing. Fusion-JEPA requires diacritized text (*Tashkeel*) and maps short vowels (*Fatha, Damma, Kasra*) to dedicated phonetic tokens. For geminate consonants (*Shaddah*), our tokenizer inserts a special doubling token (`_dbl_`), allowing the transformer self-attention to allocate sustained acoustic duration and closure energy to doubled consonants.

---

### 🎯 Key Summary Statistics for Presentation Slides:
* **Audio Sampling Rate:** $44.1\text{ kHz}$ (Studio-Grade)
* **Mel-Spectrogram Resolution:** $128\text{ bins} \times 512\text{ frames}$
* **Backbone Hidden Dimension:** $D = 768$, Multi-Head Self-Attention ($H = 12$)
* **Total ODE Sampling Steps:** $N = 60\text{ Euler steps}$
* **Classifier-Free Guidance Scale:** $w = 7.0$
* **Teacher EMA Momentum:** $\alpha = 0.9999$
* **Bilingual Datasets:** Modern Standard Arabic (*Nawar Halabi Corpus*) & English (*LJSpeech-1.1*)
