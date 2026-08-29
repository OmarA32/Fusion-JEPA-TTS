import os
import sys
import time
import re
import io
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import streamlit as st

# Configure stdout encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import core Fusion-JEPA components
from models.jepa import JEPA_base
from vocoder_manager import VocoderManager
from longform_inference import (
    split_into_prosodic_chunks,
    truncate_trailing_silence,
    stitch_audio_segments
)
from text import (
    arabic_to_tokens,
    english_to_tokens,
    tokens_to_ids,
    phon_to_id_
)

# ----------------------------------------------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Fusion-JEPA Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------------------------------------
# High-End Custom CSS (Modern Glassmorphism & Sleek Dark Styling)
# ----------------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Theme Core Styles */
    .stApp {
        background-color: #0b0d13;
        color: #e2e8f0;
    }
    
    /* Header Gradient & Typography */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #c084fc 0%, #38bdf8 50%, #fb923c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 1.0rem;
    }
    
    /* System Status Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .badge-gpu { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }
    .badge-arch { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); }
    .badge-sr { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.35); }
    .badge-ckpt { background: rgba(251, 146, 60, 0.15); color: #fb923c; border: 1px solid rgba(251, 146, 60, 0.35); }

    /* Card Panels */
    .ui-card {
        background: #131620;
        border: 1px solid #232736;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Primary Generate Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #9333ea 0%, #d97706 100%) !important;
        color: #ffffff !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(147, 51, 234, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(147, 51, 234, 0.55) !important;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        font-size: 1.15rem !important;
        line-height: 1.6 !important;
        border-radius: 10px !important;
        background-color: #0f121a !important;
        border: 1px solid #282d3f !important;
        color: #f8fafc !important;
    }
    .stTextArea textarea:focus {
        border-color: #9333ea !important;
        box-shadow: 0 0 0 1px #9333ea !important;
    }

    /* Metric Boxes */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------
# Benchmark Bank (Numbered Samples: Index 1 to 10 for Arabic & English)
# ----------------------------------------------------------------------------------------
BENCHMARKS = {
    "arabic": {
        1: {
            "title": "Standard MSA Overview (With Full Tashkeel)",
            "text": "يَعْتَمِدُ نِظَامُ فُيُوجِن جِيبَا عَلَى التَّعَلُّمِ الذَّاتِيِّ لِتَوْلِيدِ صَوْتٍ عَالِي الْجَوْدَةِ، وَيَتَمَيَّزُ بِقُدْرَتِهِ عَلَى مُعَالَجَةِ النُّصُوصِ الْعَرَبِيَّةِ الْمُعَقَّدَةِ بِكُلِّ دِقَّةٍ وَوُضُوحٍ."
        },
        2: {
            "title": "Scientific & Technological News",
            "text": "أَعْلَنَتْ مَدِينَةُ الْمَلِكِ عَبْدِاللَّهِ لِلْعُلُومِ وَالتَّقْنِيَّةِ عَنْ إِطْلَاقِ حُزْمَةٍ جَدِيدَةٍ مِنْ نَمَاذِجِ الذَّكَاءِ الاصْطِنَاعِيِّ الْمُتَقَدِّمَةِ لِدَعْمِ اللُّغَةِ الْعَرَبِيَّةِ فِي شَتَّى الْمَجَالَاتِ."
        },
        3: {
            "title": "Quranic / Classical Cadence (Surat Al-Infitar)",
            "text": "وَإِذَا السَّمَاءُ انْفَطَرَتْ، وَإِذَا الْكَوَاكِبُ انْتَثَرَتْ، وَإِذَا الْبِحَارُ فُجِّرَتْ، وَإِذَا الْقُبُورُ بُعْثِرَتْ، عَلِمَتْ نَفْسٌ مَا قَدَّمَتْ وَأَخَّرَتْ."
        },
        4: {
            "title": "Expressive Narrative & Atmosphere",
            "text": "كَانَ الصَّبَاحُ هَادِئًا فِي تِلْكَ الْقَرْيَةِ الْجَمِيلَةِ، حَيْثُ تَتَصَاعَدُ أَلْحَانُ الطُّيُورِ مَعَ إِشْرَاقَةِ الشَّمْسِ الذَّهَبِيَّةِ لِتَبْعَثَ الْأَمَلَ فِي نُفُوسِ الْجَمِيعِ."
        },
        5: {
            "title": "Phonetic Gemination (Shaddah) & Articulation Test",
            "text": "تَقَدَّمَ الْمُتَحَدِّثُ الرَّسْمِيُّ لِيُؤَكِّدَ تَطَوُّرَ الصِّنَاعَاتِ التِّقْنِيَّةِ الْمُتَقَدِّمَةِ وَتَفَوُّقَهَا الْمُسْتَمِرَّ فِي كَافَّةِ الْأَسْوَاقِ الْعَالَمِيَّةِ."
        },
        6: {
            "title": "Short Conversational Prompt",
            "text": "مَرْحَبًا بِكُمْ جَمِيعًا فِي هَذَا الْعَرْضِ التَّوْضِيحِيِّ لِمَشْرُوعِ فُيُوجِن جِيبَا لِتَوْلِيدِ الْكَلَامِ."
        },
        7: {
            "title": "Philosophical / Literary Prose",
            "text": "إِنَّ الْمَعْرِفَةَ نُورٌ يُضِيءُ دُرُوبَ الْحَيَاةِ، وَبِهَا تَرْتَقِي الْأُمَمُ وَتَتَحَقَّقُ أَعْظَمُ الإِنْجَازَاتِ الإِنْسَانِيَّةِ عَلَى مَرِّ الْعُصُورِ."
        },
        8: {
            "title": "Formal Diplomatic Announcement",
            "text": "أَكَّدَتِ الدُّوَلُ الْمُشَارِكَةُ فِي الْقِمَّةِ عَلَى أَهَمِّيَّةِ التَّعَاوُنِ الْمُشْتَرَكِ لِمُوَاجَهَةِ التَّحَدِّيَاتِ وَتَعْزِيزِ الأَمْنِ وَالاسْتِقْرَارِ فِي الْمِنْطَقَةِ."
        },
        9: {
            "title": "Multi-Paragraph Longform Article (Part 1)",
            "text": "يَشْهَدُ الْعَالَمُ الْيَوْمَ ثَوْرَةً تِقْنِيَّةً غَيْرَ مَسْبُوقَةٍ فِي مَجَالَاتِ الذَّكَاءِ الاصْطِنَاعِيِّ وَمُعَالَجَةِ اللُّغَاتِ الطَّبِيعِيَّةِ. وَقَدْ أَسْهَمَتْ هَذِهِ الاِبْتِكَارَاتُ فِي تَحْسِينِ جَوْدَةِ الْحَيَاةِ وَتَسْهِيلِ التَّوَاصُلِ بَيْنَ الشُّعُوبِ."
        },
        10: {
            "title": "Vision 2030 Transformation (Longform Paragraph)",
            "text": "تَسْعَى الْمَمْلَكَةُ الْعَرَبِيَّةُ السَّعُودِيَّةُ بِخُطًى حَثِيثَةٍ نَحْوَ بِنَاءِ مُسْتَقْبَلٍ رَقْمِيٍّ رَائِدٍ، يُرَكِّزُ عَلَى تَطْوِيرِ الْكِفَاءَاتِ الْوَطَنِيَّةِ وَتَوْطِينِ أَحْدَثِ التِّقْنِيَّاتِ الْعَالَمِيَّةِ لِتَحْقِيقِ رُؤْيَةِ عِشْرِينَ ثَلَاثِينَ."
        }
    },
    "english": {
        1: {
            "title": "Fusion-JEPA Technical Overview",
            "text": "Fusion-JEPA is a deep multimodal architecture designed for expressive text-to-speech synthesis, achieving studio-quality audio through continuous flow matching and joint-embedding representations."
        },
        2: {
            "title": "LJSpeech Benchmark Reference (Sample #001)",
            "text": "Printing, in the only sense with which we are at present concerned, differs from most if not from all other arts and crafts represented in the Exhibition."
        },
        3: {
            "title": "Pangram & Consonant Clarity Test",
            "text": "The quick brown fox jumps over the lazy dog near the vibrant riverbank under the blazing golden sunset."
        },
        4: {
            "title": "Expressive Storytelling & Cadence",
            "text": "The journey of artificial intelligence has reached an exciting milestone. Today, neural networks can understand complex linguistic patterns and generate human-like speech with remarkable naturalness."
        },
        5: {
            "title": "Scientific Abstract (Diffusion vs. Flow Matching)",
            "text": "Recent breakthroughs in continuous normalizing flows and joint-embedding predictive architectures have enabled highly efficient generative modeling without autoregressive bottlenecks."
        },
        6: {
            "title": "Conversational Greeting",
            "text": "Welcome to the live interactive demonstration of Fusion-JEPA bilingual text to speech synthesis."
        },
        7: {
            "title": "Literature & Philosophical Thought",
            "text": "In a world driven by constant innovation, the ability to communicate with clarity and emotional depth remains our greatest human achievement."
        },
        8: {
            "title": "Formal Global Broadcast",
            "text": "International delegates gathered this morning to discuss the future of sustainable technology and collaborative artificial intelligence research across leading academic institutions."
        },
        9: {
            "title": "Multi-Paragraph Longform (Evolution of TTS)",
            "text": "Speech synthesis has evolved dramatically over the last decade. From early concatenative systems to modern continuous flow matching transformers, the pursuit of truly human-like prosody has inspired researchers worldwide."
        },
        10: {
            "title": "Multimodal Future Outlook (Longform Paragraph)",
            "text": "The integration of semantic representation learning with acoustic generative models represents a promising direction for multimodal artificial intelligence, achieving higher fidelity with significantly fewer training hours."
        }
    }
}

# ----------------------------------------------------------------------------------------
# Hardware Detection
# ----------------------------------------------------------------------------------------
@st.cache_resource
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "Intel XPU"
    elif torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
        return torch.device("cuda"), f"NVIDIA GPU ({device_name})"
    else:
        return torch.device("cpu"), "CPU (Native)"

device, device_name = get_device()

# ----------------------------------------------------------------------------------------
# Model & Vocoder Cached Loaders
# ----------------------------------------------------------------------------------------
def get_latest_checkpoint(log_dir):
    if not os.path.exists(log_dir):
        return None, None, 0
    latest_ckpt, max_ckpt_epoch = None, -1
    latest_pt, max_pt_epoch = None, -1
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            filepath = os.path.join(root, f)
            if f.startswith("jepa_epoch_") and f.endswith(".pt"):
                try:
                    epoch = int(re.search(r"epoch_(\d+)", f).group(1))
                    if epoch > max_pt_epoch:
                        max_pt_epoch = epoch
                        latest_pt = filepath
                except Exception:
                    pass
            elif f.endswith(".ckpt"):
                try:
                    if "epoch=" in f:
                        epoch = int(re.search(r"epoch=(\d+)", f).group(1))
                        if epoch > max_ckpt_epoch:
                            max_ckpt_epoch = epoch
                            latest_ckpt = filepath
                    elif f == "last.ckpt":
                        max_ckpt_epoch = 99999999
                        latest_ckpt = filepath
                except Exception:
                    pass
    if max_ckpt_epoch == -1 and max_pt_epoch == -1:
        return None, None, 0
    return (latest_ckpt, "ckpt", max_ckpt_epoch) if max_ckpt_epoch >= max_pt_epoch else (latest_pt, "pt", max_pt_epoch)

@st.cache_resource(show_spinner="Loading Fusion-JEPA Neural Model...")
def load_jepa_model(lang: str, custom_ckpt: str = None):
    model = JEPA_base(
        in_channels=1,
        language=lang,
        spec_height=128,
        spec_width=512,
        patch_size=16,
        diffloss='flow',
        diffloss_d_model=384,
        cls_token=False,
        learnable_pos=True,
        drop_path_rate=0.0
    ).to(device)

    found_path = None
    ckpt_type = None

    if custom_ckpt and os.path.exists(custom_ckpt):
        found_path = custom_ckpt
        ckpt_type = "pt" if custom_ckpt.endswith(".pt") else "ckpt"
    else:
        log_dir = os.path.join("training_logs", lang)
        found_path, ckpt_type, epoch_num = get_latest_checkpoint(log_dir)

    if found_path and os.path.exists(found_path):
        if ckpt_type == "pt":
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)
            model_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    model_dict[k.replace("model.", "", 1)] = v
                elif not k.startswith("ema_model."):
                    model_dict[k] = v
            model.load_state_dict(model_dict, strict=False)
        ckpt_display = os.path.basename(found_path)
    else:
        ckpt_display = "Untrained (No checkpoint in training_logs/)"

    model.eval()
    return model, ckpt_display, found_path

@st.cache_resource(show_spinner="Loading BigVGAN v2 Vocoder (44.1kHz Studio)...")
def load_vocoder():
    return VocoderManager(device=device)

# ----------------------------------------------------------------------------------------
# Session State Management for Text & Sample Index
# ----------------------------------------------------------------------------------------
if "input_text_content" not in st.session_state:
    st.session_state["input_text_content"] = BENCHMARKS["arabic"][1]["text"]

if "input_mode" not in st.session_state:
    st.session_state["input_mode"] = "index"

def sync_text_from_index():
    lang = st.session_state.get("lang_choice_widget", "arabic")
    idx = st.session_state.get("sample_idx_widget", 1)
    st.session_state["input_text_content"] = BENCHMARKS[lang][idx]["text"]

def sync_text_on_lang_switch():
    lang = st.session_state.get("lang_choice_widget", "arabic")
    mode = st.session_state.get("mode_choice_widget", "index")
    if mode == "index":
        idx = st.session_state.get("sample_idx_widget", 1)
        st.session_state["input_text_content"] = BENCHMARKS[lang][idx]["text"]

# ----------------------------------------------------------------------------------------
# UI Header & Badges
# ----------------------------------------------------------------------------------------
st.markdown('<div class="main-title">🎙️ Fusion-JEPA Studio — Expressive Bilingual TTS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Continuous Flow Matching & Decoupled Latent Joint-Embedding Predictive Architecture</div>', unsafe_allow_html=True)

# Preview active checkpoint name
current_lang_preview = st.session_state.get("lang_choice_widget", "arabic")
_, ckpt_name_preview, _ = load_jepa_model(current_lang_preview)

st.markdown(f"""
<div class="badge-container">
    <span class="badge badge-gpu">⚡ Device: {device_name}</span>
    <span class="badge badge-arch">🧩 Model: MM-DiT (128x512 Canvas)</span>
    <span class="badge badge-sr">📻 Vocoder: BigVGAN v2 (44.1 kHz Studio)</span>
    <span class="badge badge-ckpt">📁 Checkpoint: {ckpt_name_preview}</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------
# Configuration Grid (3 Styled Cards)
# ----------------------------------------------------------------------------------------
col_left, col_mid, col_right = st.columns([1.1, 1.25, 1.15])

# CARD 1: Language & Input Mode Selection
with col_left:
    st.markdown('<div class="card-title">🌐 1. Language & Input Mode</div>', unsafe_allow_html=True)
    
    lang_choice = st.radio(
        "Language",
        options=["arabic", "english"],
        format_func=lambda x: "🇸🇦 Arabic (العربية)" if x == "arabic" else "🇬🇧 English",
        horizontal=True,
        key="lang_choice_widget",
        on_change=sync_text_on_lang_switch
    )

    mode_choice = st.radio(
        "Input Mode",
        options=["index", "custom"],
        format_func=lambda x: "🔢 Benchmark Sample Index" if x == "index" else "✏️ Custom Freeform Text",
        horizontal=True,
        key="mode_choice_widget",
        on_change=sync_text_on_lang_switch
    )

    if mode_choice == "index":
        selected_idx = st.number_input(
            "Select Sample Index # (1 to 10)",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            key="sample_idx_widget",
            on_change=sync_text_from_index,
            help="Picks a curated evaluation sentence and automatically pastes it into the text box below."
        )
        sample_meta = BENCHMARKS[lang_choice][selected_idx]
        st.caption(f"📌 **Sample #{selected_idx}:** {sample_meta['title']}")

# CARD 2: Synthesis Hyperparameters
with col_mid:
    st.markdown('<div class="card-title">⚙️ 2. Flow Matching Controls</div>', unsafe_allow_html=True)
    
    steps = st.slider(
        "Euler ODE Integration Steps ($N$)",
        min_value=16,
        max_value=100,
        value=60,
        step=4,
        help="16 = Ultra-Fast (~0.15s), 32 = Balanced (~0.25s), 60 = Studio Quality (~0.45s)."
    )
    
    cfg_scale = st.slider(
        "Classifier-Free Guidance Scale (CFG $w$)",
        min_value=1.0,
        max_value=15.0,
        value=7.0,
        step=0.5,
        help="Controls adherence to text phonemes. w=7.0 provides crystal-clear consonants and removes muffled speech."
    )

# CARD 3: Phrasing & Feature Toggles
with col_right:
    st.markdown('<div class="card-title">🎛️ 3. Phrasing & Diagnostics</div>', unsafe_allow_html=True)
    
    pause_ms = st.slider(
        "Inter-Clause Pause Duration (ms)",
        min_value=0,
        max_value=400,
        value=100,
        step=25,
        help="Acoustic silence inserted between stitched prosodic sentences in longform speech."
    )
    
    trim_silence = st.checkbox("✂️ Adaptive Silence Truncation", value=True, help="Trims unconditioned trailing canvas noise.")
    save_mel = st.checkbox("📊 Display Mel-Spectrogram Plot", value=True, help="Renders time-frequency harmonic spectrogram.")
    show_chunks = st.checkbox("🔍 Show Prosodic Clause Table", value=True, help="Displays segmented clause diagnostics.")

# Advanced Checkpoint Expander
with st.expander("🛠️ Advanced: Custom Checkpoint Override"):
    custom_ckpt_path = st.text_input(
        "Checkpoint File Path (Optional - Leave blank for automatic latest detection)",
        value="",
        placeholder="e.g. training_logs/arabic/jepa_epoch_200.pt"
    )

# ----------------------------------------------------------------------------------------
# Text Input Area
# ----------------------------------------------------------------------------------------
st.markdown("##### ✍️ Input Text (Single Sentences or Multi-Paragraph Longform)")

# Text area bound to session state
main_text = st.text_area(
    "Text Input",
    value=st.session_state["input_text_content"],
    height=125,
    placeholder="Type or paste Arabic text with Tashkeel or English text...",
    label_visibility="collapsed"
)

# Keep session state in sync with manual user edits
st.session_state["input_text_content"] = main_text

# Text Statistics Badge
char_count = len(main_text)
word_count = len(main_text.split())
est_dur = max(0.5, word_count * 0.45)
st.caption(f"📝 Length: **{word_count}** words | **{char_count}** characters | Estimated Audio Duration: **~{est_dur:.1f}s**")

# Phonetic Token Preview Expander
with st.expander("🔎 Phonetization Token Inspector"):
    if main_text.strip():
        try:
            if lang_choice == "arabic":
                preview_tokens = arabic_to_tokens(main_text)
            else:
                preview_tokens = english_to_tokens(main_text)
            valid_toks = [t for t in preview_tokens if t in phon_to_id_]
            st.write(f"**Total Phonetized Tokens ({len(valid_toks)}):**")
            st.code(" ".join(valid_toks), language="text")
        except Exception as e:
            st.warning(f"Could not compute phonetic preview: {e}")
    else:
        st.write("Enter text above to preview phonetic tokens.")

# ----------------------------------------------------------------------------------------
# Generation Execution
# ----------------------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
generate_btn = st.button("🚀 Generate High-Fidelity Speech", type="primary", use_container_width=True)

if generate_btn:
    if not main_text.strip():
        st.error("Please enter or select some text before generating audio.")
    else:
        # Load model and vocoder
        model, ckpt_status, active_ckpt = load_jepa_model(lang_choice, custom_ckpt_path if custom_ckpt_path else None)
        vocoder = load_vocoder()

        # Step 1: Prosodic chunking
        with st.spinner("Analyzing linguistic prosody & sentence segmentation..."):
            chunks = split_into_prosodic_chunks(main_text, lang=lang_choice, max_phonemes=55, min_phonemes=18)

        if not chunks:
            st.error("No valid text clauses found to synthesize.")
        else:
            st.info(f"✨ Input segmented into **{len(chunks)}** coherent prosodic clause(s). Synthesizing on **{device_name}**...")

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            sample_rate = 44100
            audio_segments = []
            mel_segments = []
            clause_diagnostics = []

            t_start = time.time()

            for i, (chunk_text, chunk_tokens) in enumerate(chunks):
                status_text.text(f"Synthesizing Clause {i+1}/{len(chunks)}: \"{chunk_text}\" (ODE steps: {steps}, CFG: {cfg_scale})...")
                
                try:
                    if lang_choice == "arabic":
                        tokens = arabic_to_tokens(chunk_text)
                    else:
                        tokens = english_to_tokens(chunk_text)
                    tokens = [t for t in tokens if t in phon_to_id_]
                    token_ids = tokens_to_ids(tokens)
                except Exception as e:
                    st.warning(f"Error phonetizing clause {i+1}: {e}")
                    continue

                text_input = [token_ids]

                with torch.no_grad():
                    generated_mel = model.sample_tokens(
                        bsz=1,
                        num_iter=steps,
                        cfg_scale=cfg_scale,
                        labels=text_input
                    )

                mel_for_vocoder = generated_mel.squeeze(1)
                with torch.no_grad():
                    audio_wave = vocoder.generate_audio(mel_for_vocoder)

                if trim_silence:
                    clean_audio, cut_frame = truncate_trailing_silence(
                        audio_waveform=audio_wave,
                        mel_spectrogram=mel_for_vocoder,
                        num_phonemes=len(tokens),
                        sample_rate=sample_rate,
                        hop_length=512
                    )
                    clean_mel = mel_for_vocoder[:, :, :cut_frame].squeeze(0).cpu().numpy()
                else:
                    clean_audio = audio_wave.squeeze().cpu().numpy()
                    clean_mel = mel_for_vocoder.squeeze(0).cpu().numpy()

                dur = len(clean_audio) / sample_rate
                audio_segments.append(clean_audio)
                mel_segments.append(clean_mel)
                clause_diagnostics.append({
                    "Clause #": i + 1,
                    "Text": chunk_text,
                    "Phonemes": len(tokens),
                    "Duration": f"{dur:.2f}s"
                })

                progress_bar.progress((i + 1) / len(chunks))

            t_end = time.time()
            total_gen_time = t_end - t_start

            # Step 2: Audio Stitching
            status_text.text("Stitching audio segments with natural phrasing...")
            stitched_audio = stitch_audio_segments(audio_segments, pause_ms=pause_ms, sample_rate=sample_rate)
            
            if len(stitched_audio) > 0:
                stitched_audio = stitched_audio / max(abs(stitched_audio).max(), 1e-8)
                audio_int16 = (stitched_audio * 32767).astype(np.int16)
                total_duration = len(stitched_audio) / sample_rate
                rtf = total_gen_time / max(total_duration, 1e-8)

                # Save output WAV
                os.makedirs("test_results", exist_ok=True)
                output_wav_path = os.path.join("test_results", "web_output.wav")
                wavfile.write(output_wav_path, sample_rate, audio_int16)

                # In-memory WAV buffer for instant browser playback
                wav_buffer = io.BytesIO()
                wavfile.write(wav_buffer, sample_rate, audio_int16)
                wav_bytes = wav_buffer.getvalue()

                progress_bar.progress(1.0)
                status_text.empty()

                st.success("🎉 **Speech Synthesis Complete!**")

                # Audio Player & Download Section
                st.markdown("### 🎧 Audio Output")
                st.audio(wav_bytes, format="audio/wav")

                col_dl, col_space = st.columns([1, 3])
                with col_dl:
                    st.download_button(
                        label="⬇️ Download Audio (.wav)",
                        data=wav_bytes,
                        file_name=f"fusion_jepa_{lang_choice}_{int(time.time())}.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )

                # Performance Metrics
                st.markdown("### 📊 Performance & Audio Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("⏱️ Generation Time", f"{total_gen_time:.2f} s")
                m2.metric("⚡ Real-Time Factor (RTF)", f"{rtf:.3f}x", help="Lower is faster. < 1.0 means faster than real-time playback.")
                m3.metric("🎵 Total Audio Duration", f"{total_duration:.2f} s")
                m4.metric("📻 Sampling Rate", "44.1 kHz Studio")

                # Spectrogram Visualization
                if save_mel and mel_segments:
                    st.markdown("### 🌈 Stitched Mel-Spectrogram (128-Band Studio)")
                    try:
                        mel_stitched = np.concatenate(mel_segments, axis=1)
                        fig, ax = plt.subplots(figsize=(14, 3.8))
                        im = ax.imshow(mel_stitched, origin='lower', aspect='auto', cmap='viridis')
                        ax.set_title(f"Fusion-JEPA Mel-Spectrogram — Duration: {total_duration:.2f}s | Clauses: {len(chunks)}", fontsize=12, color='white', pad=10)
                        ax.set_xlabel("Time Frames (Hop = 512)", color='white')
                        ax.set_ylabel("Mel Frequency Bins (128)", color='white')
                        ax.tick_params(colors='white')
                        fig.patch.set_facecolor('#0e1117')
                        ax.set_facecolor('#0e1117')
                        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.25, shrink=0.6, label='Log-Mel Energy')
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"Could not render spectrogram: {e}")

                # Diagnostics Breakdown Table
                if show_chunks and clause_diagnostics:
                    st.markdown("### 📑 Prosodic Clause Breakdown")
                    st.table(clause_diagnostics)
