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
from models.vocoder_manager import VocoderManager
from data.dataset import JEPADataset
from text.phonetise_buckwalter import buckwalter_to_arabic
from pipelines.longform_inference import (
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
# High-End Custom CSS (Centered Container & Sleek Modern Card Layout)
# ----------------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Theme Core Background */
    .stApp {
        background-color: #0b0d13;
        color: #e2e8f0;
    }
    
    /* Clean Top Header Spacing */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Centered Container Wrapper with Generous Top Padding */
    .block-container {
        max-width: 980px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 5.5rem !important;
        padding-bottom: 3.5rem !important;
    }
    
    /* Header Gradient & Centered Typography */
    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        text-align: center;
        background: linear-gradient(135deg, #c084fc 0%, #38bdf8 50%, #fb923c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
        line-height: 1.3;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    
    /* Centered System Status Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .badge-gpu { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }
    .badge-arch { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); }
    .badge-sr { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.35); }
    .badge-ckpt { background: rgba(251, 146, 60, 0.15); color: #fb923c; border: 1px solid rgba(251, 146, 60, 0.35); }

    /* Expander Container (Folded Options) */
    .streamlit-expanderHeader {
        background-color: #131620 !important;
        border: 1px solid #232736 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }
    .streamlit-expanderContent {
        background-color: #10131c !important;
        border: 1px solid #232736 !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
        padding: 1.2rem !important;
    }

    /* Option Columns Margin Justification */
    .options-box {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    .card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.7rem;
        border-bottom: 1px solid #232736;
        padding-bottom: 0.4rem;
    }

    /* Primary Generate Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #9333ea 0%, #d97706 100%) !important;
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        padding: 0.8rem 1.6rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 18px rgba(147, 51, 234, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
        display: block !important;
        margin: 0 auto !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(147, 51, 234, 0.6) !important;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        font-size: 1.15rem !important;
        line-height: 1.65 !important;
        border-radius: 12px !important;
        background-color: #10131c !important;
        border: 1px solid #232736 !important;
        color: #f8fafc !important;
    }
    .stTextArea textarea:focus {
        border-color: #9333ea !important;
        box-shadow: 0 0 0 1px #9333ea !important;
    }

    /* Metric Value Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

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
# Cached Dataset & Model Loaders
# ----------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Dataset Database...")
def load_database_cached(lang: str, db_name: str):
    try:
        ds = JEPADataset(split="test", lang=lang, db=db_name)
        return ds
    except Exception as e:
        st.warning(f"Could not load dataset {db_name} for {lang}: {e}")
        return None

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
# Text & Index Sync Callbacks
# ----------------------------------------------------------------------------------------
def extract_text_from_db(lang: str, db_name: str, index: int):
    ds = load_database_cached(lang, db_name)
    if ds is None or len(ds.dataset) == 0:
        return "Fusion-JEPA is a deep multimodal architecture designed for expressive text-to-speech synthesis." if lang == "english" else "يَعْتَمِدُ نِظَامُ فُيُوجِن جِيبَا عَلَى التَّعَلُّمِ الذَّاتِيِّ لِتَوْلِيدِ صَوْتٍ عَالِي الْجَوْدَةِ."
    
    idx = max(0, min(index, len(ds.dataset) - 1))
    item = ds.dataset[idx]
    raw_text = item.get("sentence", item.get("text", ""))
    
    if lang == "arabic" and any(c in raw_text for c in ['>', '<', 'p', '~', 'o', 'E', 'H', 'S', 'D', 'T', 'Z', '*']):
        try:
            converted = buckwalter_to_arabic(raw_text)
            if converted.strip():
                return converted
        except Exception:
            pass
    return raw_text

def sync_text_from_db_index():
    lang = st.session_state.get("lang_choice_widget", "english")
    db_name = st.session_state.get("db_choice_widget", "ljspeech" if lang == "english" else "nawar_halabi")
    idx = st.session_state.get("db_index_widget", 0)
    st.session_state["input_text_content"] = extract_text_from_db(lang, db_name, idx)

def sync_text_on_lang_switch():
    lang = st.session_state.get("lang_choice_widget", "english")
    mode = st.session_state.get("mode_choice_widget", "custom")
    default_db = "ljspeech" if lang == "english" else "nawar_halabi"
    st.session_state["db_choice_widget"] = default_db
    
    if mode == "db_index":
        idx = st.session_state.get("db_index_widget", 0)
        st.session_state["input_text_content"] = extract_text_from_db(lang, default_db, idx)
    elif mode == "custom":
        if lang == "english" and "يَعْتَمِدُ" in st.session_state.get("input_text_content", ""):
            st.session_state["input_text_content"] = "Fusion-JEPA is a deep multimodal architecture designed for expressive text-to-speech synthesis, achieving studio-quality audio through continuous flow matching."
        elif lang == "arabic" and "Fusion-JEPA" in st.session_state.get("input_text_content", ""):
            st.session_state["input_text_content"] = "يَعْتَمِدُ نِظَامُ فُيُوجِن جِيبَا عَلَى التَّعَلُّمِ الذَّاتِيِّ لِتَوْلِيدِ صَوْتٍ عَالِي الْجَوْدَةِ، وَيَتَمَيَّزُ بِقُدْرَتِهِ عَلَى مُعَالَجَةِ النُّصُوصِ الْعَرَبِيَّةِ الْمُعَقَّدَةِ بِكُلِّ دِقَّةٍ وَوُضُوحٍ."

# Initialize Session State (Default Custom Text)
if "input_text_content" not in st.session_state:
    st.session_state["input_text_content"] = "Fusion-JEPA is a deep multimodal architecture designed for expressive text-to-speech synthesis, achieving studio-quality audio through continuous flow matching."

if "mode_choice_widget" not in st.session_state:
    st.session_state["mode_choice_widget"] = "custom"

# ----------------------------------------------------------------------------------------
# Centered UI Header
# ----------------------------------------------------------------------------------------
st.markdown('<div class="main-title">🎙️ Fusion-JEPA Studio — Expressive Bilingual TTS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Continuous Flow Matching & Decoupled Latent Joint-Embedding Predictive Architecture</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------
# Folded Options Expander (Justified & Vertically Aligned Grid)
# ----------------------------------------------------------------------------------------
with st.expander("⚙️ Options & Synthesis Configuration (Language, Sampling, Dataset)", expanded=False):
    # Device & Checkpoint Status inside the options box
    current_lang_preview = st.session_state.get("lang_choice_widget", "english")
    _, ckpt_name_preview, _ = load_jepa_model(current_lang_preview)
    st.markdown(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding: 0.5rem 0.8rem; background: #0c0e15; border-radius: 8px; border: 1px solid #1f2330; font-size: 0.83rem; color: #94a3b8;">
        <span>⚡ <b style="color:#c084fc;">Hardware Device:</b> <code style="color:#e2e8f0;">{device_name}</code></span>
        <span>🧩 <b style="color:#38bdf8;">Architecture:</b> MM-DiT 128x512</span>
        <span>📻 <b style="color:#34d399;">Vocoder:</b> BigVGAN v2 (44.1 kHz)</span>
        <span>📁 <b style="color:#fb923c;">Checkpoint:</b> <code style="color:#e2e8f0;">{ckpt_name_preview}</code></span>
    </div>
    """, unsafe_allow_html=True)

    opt_col1, opt_col2, opt_col3 = st.columns(3)

    # Column 1: Language & Input Mode
    with opt_col1:
        st.markdown('<div class="card-title">🌐 1. Language & Dataset Mode</div>', unsafe_allow_html=True)
        
        lang_choice = st.radio(
            "Language",
            options=["english", "arabic"],
            format_func=lambda x: "🇬🇧 English" if x == "english" else "🇸🇦 Arabic (العربية)",
            horizontal=True,
            key="lang_choice_widget",
            on_change=sync_text_on_lang_switch
        )

        mode_choice = st.radio(
            "Input Mode",
            options=["custom", "db_index"],
            format_func=lambda x: "✏️ Custom Freeform Text" if x == "custom" else "🗄️ Database Sample Index",
            horizontal=True,
            key="mode_choice_widget",
            on_change=sync_text_on_lang_switch
        )

        if mode_choice == "db_index":
            available_dbs = ["ljspeech", "libritts"] if lang_choice == "english" else ["nawar_halabi", "common_voice", "clartts"]
            db_choice = st.selectbox(
                "Select Database",
                options=available_dbs,
                key="db_choice_widget",
                on_change=sync_text_from_db_index
            )
            
            loaded_ds = load_database_cached(lang_choice, db_choice)
            max_idx = len(loaded_ds.dataset) - 1 if loaded_ds and len(loaded_ds.dataset) > 0 else 1000

            selected_idx = st.number_input(
                f"Dataset Index # (0 to {max_idx})",
                min_value=0,
                max_value=max_idx,
                value=0,
                step=1,
                key="db_index_widget",
                on_change=sync_text_from_db_index,
                help="Pulls the exact ground-truth sentence from the database and pastes it into the text box below."
            )
            if loaded_ds and selected_idx < len(loaded_ds.dataset):
                item_info = loaded_ds.dataset[selected_idx]
                clip_name = os.path.basename(item_info.get("audio_path", f"Clip #{selected_idx}"))
                st.caption(f"📁 **Clip:** `{clip_name}` ({len(loaded_ds.dataset)} in DB)")
        else:
            loaded_ds = None
            selected_idx = 0
            st.caption("✏️ Freeform mode active: Type or paste any text directly below.")

    # Column 2: Flow Matching Hyperparameters
    with opt_col2:
        st.markdown('<div class="card-title">⚙️ 2. Flow Matching Controls</div>', unsafe_allow_html=True)
        
        steps = st.slider(
            "Euler ODE Steps ($N$)",
            min_value=16,
            max_value=100,
            value=60,
            step=4,
            help="16 = Ultra-Fast (~0.15s), 32 = Balanced (~0.25s), 60 = Studio Quality (~0.45s)."
        )
        
        cfg_scale = st.slider(
            "Guidance Scale (CFG $w$)",
            min_value=1.0,
            max_value=15.0,
            value=7.0,
            step=0.5,
            help="Amplifies text conditioning. w=7.0 gives loud, crisp, and well-articulated consonants."
        )

    # Column 3: Phrasing & Diagnostics
    with opt_col3:
        st.markdown('<div class="card-title">🎛️ 3. Phrasing & Diagnostics</div>', unsafe_allow_html=True)
        
        pause_ms = st.slider(
            "Inter-Clause Pause (ms)",
            min_value=0,
            max_value=400,
            value=100,
            step=25,
            help="Acoustic silence inserted between stitched sentences in longform speech."
        )
        
        trim_silence = st.checkbox("✂️ Adaptive Silence Truncation", value=True, help="Trims unconditioned trailing canvas noise.")
        save_mel = st.checkbox("📊 Display Mel-Spectrogram Plot", value=True, help="Renders time-frequency harmonic spectrogram.")
        show_chunks = st.checkbox("🔍 Show Prosodic Clause Table", value=True, help="Displays segmented clause diagnostics.")

    st.markdown("---")
    custom_ckpt_path = st.text_input(
        "🛠️ Custom Checkpoint File Path (Optional - Leave blank for automatic latest detection)",
        value="",
        placeholder="e.g. training_logs/english/jepa_epoch_200.pt"
    )

# ----------------------------------------------------------------------------------------
# Text Input Area (Centered & Prominent)
# ----------------------------------------------------------------------------------------
st.markdown("##### ✍️ Input Text (Single Sentences or Multi-Paragraph Longform)")

# Text area bound to session state
main_text = st.text_area(
    "Text Input",
    value=st.session_state["input_text_content"],
    height=130,
    placeholder="Type or paste Arabic text with Tashkeel or English text...",
    label_visibility="collapsed"
)

# Keep session state in sync with manual user edits
st.session_state["input_text_content"] = main_text

# Text Statistics Row
char_count = len(main_text)
word_count = len(main_text.split())
est_dur = max(0.5, word_count * 0.45)
st.caption(f"📝 Length: **{word_count}** words | **{char_count}** characters | Estimated Audio Duration: **~{est_dur:.1f}s**")

# Phonetic Token Preview Expander
with st.expander("🔎 Phonetization Token Inspector (IPA & Vowel Breakdown)"):
    if main_text.strip():
        try:
            lang_now = st.session_state.get("lang_choice_widget", "english")
            if lang_now == "arabic":
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
# Centered Generation Button
# ----------------------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
generate_btn = st.button("🚀 Generate High-Fidelity Speech", type="primary", use_container_width=True)

if generate_btn:
    if not main_text.strip():
        st.error("Please enter or select some text before generating audio.")
    else:
        lang_now = st.session_state.get("lang_choice_widget", "english")
        mode_now = st.session_state.get("mode_choice_widget", "custom")
        
        # Load model and vocoder
        model, ckpt_status, active_ckpt = load_jepa_model(lang_now, custom_ckpt_path if custom_ckpt_path else None)
        vocoder = load_vocoder()

        # Step 1: Prosodic chunking
        with st.spinner("Analyzing linguistic prosody & sentence segmentation..."):
            chunks = split_into_prosodic_chunks(main_text, lang=lang_now, max_phonemes=90, min_phonemes=20)

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
                    if lang_now == "arabic":
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

                # Get optional ground truth path if in DB mode
                gt_audio_path = None
                if mode_now == "db_index":
                    db_now = st.session_state.get("db_choice_widget", "ljspeech" if lang_now == "english" else "nawar_halabi")
                    loaded_ds = load_database_cached(lang_now, db_now)
                    idx_now = st.session_state.get("db_index_widget", 0)
                    if loaded_ds and idx_now < len(loaded_ds.dataset):
                        item_gt = loaded_ds.dataset[idx_now]
                        cand_gt = item_gt.get("audio_path", "")
                        if cand_gt and os.path.exists(cand_gt):
                            gt_audio_path = cand_gt

                st.session_state["last_synthesis_result"] = {
                    "wav_bytes": wav_bytes,
                    "lang": lang_now,
                    "total_gen_time": total_gen_time,
                    "total_duration": total_duration,
                    "rtf": rtf,
                    "mel_segments": mel_segments,
                    "clause_diagnostics": clause_diagnostics,
                    "gt_audio_path": gt_audio_path,
                    "num_chunks": len(chunks),
                    "created_at": time.time()
                }

                progress_bar.progress(1.0)
                status_text.empty()

# ----------------------------------------------------------------------------------------
# Persistent Synthesis Results Display
# ----------------------------------------------------------------------------------------
if st.session_state.get("last_synthesis_result") is not None:
    res = st.session_state["last_synthesis_result"]
    
    st.success("🎉 **Speech Synthesis Complete!**")

    # Audio Player & Download Section
    st.markdown("### 🎧 Audio Output")
    st.audio(res["wav_bytes"], format="audio/wav")

    # Optional Ground-Truth Audio Player if in Database Mode
    if res.get("gt_audio_path") and os.path.exists(res["gt_audio_path"]):
        with st.expander("🔊 Listen to Ground-Truth Human Audio for Comparison"):
            st.audio(res["gt_audio_path"], format="audio/wav")

    col_dl, col_space = st.columns([1.2, 2.8])
    with col_dl:
        st.download_button(
            label="⬇️ Download Audio (.wav)",
            data=res["wav_bytes"],
            file_name=f"fusion_jepa_{res['lang']}_{int(res.get('created_at', time.time()))}.wav",
            mime="audio/wav",
            use_container_width=True
        )

    # Performance Metrics (Centered 4 KPIs)
    st.markdown("### 📊 Performance & Audio Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱️ Generation Time", f"{res['total_gen_time']:.2f} s")
    m2.metric("⚡ Real-Time Factor (RTF)", f"{res['rtf']:.3f}x", help="Lower is faster. < 1.0 means faster than real-time playback.")
    m3.metric("🎵 Total Audio Duration", f"{res['total_duration']:.2f} s")
    m4.metric("📻 Sampling Rate", "44.1 kHz Studio")

    # Spectrogram Visualization
    if save_mel and res.get("mel_segments"):
        st.markdown("### 🌈 Stitched Mel-Spectrogram (128-Band Studio)")
        try:
            mel_stitched = np.concatenate(res["mel_segments"], axis=1)
            fig, ax = plt.subplots(figsize=(14, 3.8))
            im = ax.imshow(mel_stitched, origin='lower', aspect='auto', cmap='viridis')
            ax.set_title(f"Fusion-JEPA Mel-Spectrogram — Duration: {res['total_duration']:.2f}s | Clauses: {res.get('num_chunks', 1)}", fontsize=12, color='white', pad=10)
            ax.set_xlabel("Time Frames (Hop = 512)", color='white')
            ax.set_ylabel("Mel Frequency Bins (128)", color='white')
            ax.tick_params(colors='white')
            fig.patch.set_facecolor('#0b0d13')
            ax.set_facecolor('#0b0d13')
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.25, shrink=0.6, label='Log-Mel Energy')
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not render spectrogram: {e}")

    # Diagnostics Breakdown Table
    if show_chunks and res.get("clause_diagnostics"):
        st.markdown("### 📑 Prosodic Clause Breakdown")
        st.table(res["clause_diagnostics"])
