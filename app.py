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

# Custom CSS for Sleek Dark / Purple Theme
st.markdown("""
<style>
    /* Global Card & Header Styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #9d4edd, #ff7b00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #a0a0a0;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .badge-gpu { background-color: #2d124d; color: #d8b4fe; border: 1px solid #7c3aed; }
    .badge-arch { background-color: #1e293b; color: #94a3b8; border: 1px solid #475569; }
    .badge-sr { background-color: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
    .stTextArea textarea {
        font-size: 1.15rem !important;
        border-radius: 8px !important;
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
        ckpt_status = f"Loaded ({os.path.basename(found_path)})"
    else:
        ckpt_status = "Untrained (No checkpoint in training_logs/)"

    model.eval()
    return model, ckpt_status, found_path

@st.cache_resource(show_spinner="Loading BigVGAN v2 Vocoder (44.1kHz Studio)...")
def load_vocoder():
    return VocoderManager(device=device)

# ----------------------------------------------------------------------------------------
# Sample Presets Bank
# ----------------------------------------------------------------------------------------
PRESETS = {
    "arabic": {
        "1. Standard MSA Overview (With Tashkeel)": (
            "يَعْتَمِدُ نِظَامُ فُيُوجِن جِيبَا عَلَى التَّعَلُّمِ الذَّاتِيِّ لِتَوْلِيدِ صَوْتٍ عَالِي الْجَوْدَةِ، "
            "وَيَتَمَيَّزُ بِقُدْرَتِهِ عَلَى مُعَالَجَةِ النُّصُوصِ الْعَرَبِيَّةِ الْمُعَقَّدَةِ بِكُلِّ دِقَّةٍ وَوُضُوحٍ."
        ),
        "2. Classical / Poetic Verse (Rhythmic Cadence)": (
            "وَإِذَا السَّمَاءُ انْفَطَرَتْ، وَإِذَا الْكَوَاكِبُ انْتَثَرَتْ، وَإِذَا الْبِحَارُ فُجِّرَتْ، "
            "وَإِذَا الْقُبُورُ بُعْثِرَتْ، عَلِمَتْ نَفْسٌ مَا قَدَّمَتْ وَأَخَّرَتْ."
        ),
        "3. Technology & Scientific News (Longform Paragraph)": (
            "أَعْلَنَتْ مَدِينَةُ الْمَلِكِ عَبْدِاللَّهِ لِلْعُلُومِ وَالتَّقْنِيَّةِ عَنْ إِطْلَاقِ حُزْمَةٍ جَدِيدَةٍ "
            "مِنْ نَمَاذِجِ الذَّكَاءِ الاصْطِنَاعِيِّ الْمُتَقَدِّمَةِ لِدَعْمِ اللُّغَةِ الْعَرَبِيَّةِ فِي شَتَّى الْمَجَالَاتِ، "
            "مِمَّا يُعَزِّزُ مِنْ مَكَانَةِ الْمَمْلَكَةِ كَمَرْكَزٍ إِقْلِيمِيٍّ لِلابْتِكَارِ وَالتَّطْوِيرِ التِّقْنِيِّ."
        ),
        "4. Custom Arabic Text": ""
    },
    "english": {
        "1. Technical Introduction": (
            "Fusion-JEPA is a deep multimodal architecture designed for expressive text-to-speech synthesis, "
            "achieving studio-quality audio through continuous flow matching and joint-embedding representations."
        ),
        "2. Narrative / Storytelling (Longform Paragraph)": (
            "The journey of artificial intelligence has reached an exciting milestone. Today, neural networks can understand "
            "complex linguistic patterns and generate human-like speech with remarkable naturalness, capturing subtle inflections "
            "and emotional cadence across multiple languages."
        ),
        "3. Custom English Text": ""
    }
}

# ----------------------------------------------------------------------------------------
# UI Header
# ----------------------------------------------------------------------------------------
st.markdown('<div class="main-title">🎙️ Fusion-JEPA Studio — Expressive Bilingual TTS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Continuous Flow Matching & Decoupled Latent Joint-Embedding Predictive Architecture</div>', unsafe_allow_html=True)

# Hardware & System Badges
st.markdown(f"""
<div>
    <span class="badge badge-gpu">⚡ Device: {device_name}</span>
    <span class="badge badge-arch">🧩 Model: MM-DiT 128-band Mel</span>
    <span class="badge badge-sr">📻 Vocoder: BigVGAN v2 (44.1 kHz Studio)</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------
# Top Control Grid (Above Text Box)
# ----------------------------------------------------------------------------------------
with st.container():
    col_lang, col_hyper, col_opt = st.columns([1.1, 1.3, 1.2])

    with col_lang:
        st.markdown("##### 🌐 Language & Sample Presets")
        lang_choice = st.radio(
            "Select Language",
            options=["arabic", "english"],
            format_func=lambda x: "🇸🇦 Arabic (العربية)" if x == "arabic" else "🇬🇧 English",
            horizontal=True,
            label_visibility="collapsed"
        )
        preset_names = list(PRESETS[lang_choice].keys())
        selected_preset = st.selectbox(
            "Load Sample Preset",
            options=preset_names,
            index=0
        )
        default_text = PRESETS[lang_choice][selected_preset]

    with col_hyper:
        st.markdown("##### ⚙️ Synthesis Hyperparameters")
        steps = st.slider(
            "Euler ODE Steps (Inference Speed vs. Quality)",
            min_value=16,
            max_value=100,
            value=60,
            step=4,
            help="16 = Fast generation (~0.15s), 32 = Balanced (~0.25s), 60 = Studio Quality (~0.45s)."
        )
        cfg_scale = st.slider(
            "Classifier-Free Guidance (CFG Scale $w$)",
            min_value=1.0,
            max_value=15.0,
            value=7.0,
            step=0.5,
            help="Higher values (e.g. 7.0) strongly enforce phonetic alignment and remove muffled speech."
        )

    with col_opt:
        st.markdown("##### 🎛️ Phrasing & Features")
        pause_ms = st.slider(
            "Inter-Clause Breath Pause (ms)",
            min_value=0,
            max_value=400,
            value=100,
            step=25,
            help="Acoustic silence inserted between stitched prosodic sentences in longform audio."
        )
        trim_silence = st.checkbox("✂️ Smart Adaptive Silence Truncation", value=True, help="Removes trailing canvas noise.")
        save_mel = st.checkbox("📊 Display Mel-Spectrogram Analysis", value=True, help="Visualizes frequency harmonics.")
        show_chunks = st.checkbox("🔍 Show Prosodic Clause Breakdown", value=True, help="Displays segmented sentence clauses.")

# Advanced Checkpoint Expander
with st.expander("🛠️ Advanced Checkpoint & Model Settings"):
    custom_ckpt_path = st.text_input(
        "Custom Checkpoint Path (Optional - Leave blank for auto-detection)",
        value="",
        placeholder="e.g. training_logs/arabic/jepa_epoch_200.pt"
    )

# ----------------------------------------------------------------------------------------
# Text Input Area
# ----------------------------------------------------------------------------------------
st.markdown("##### ✍️ Input Text (Single Sentences or Multi-Paragraph Longform)")
input_text = st.text_area(
    "Input Text",
    value=default_text,
    height=120,
    placeholder="Type or paste Arabic text with Tashkeel or English text...",
    label_visibility="collapsed"
)

# Text Stats & Token Preview
char_count = len(input_text)
word_count = len(input_text.split())
st.caption(f"📝 Length: **{word_count}** words | **{char_count}** characters")

with st.expander("🔎 Phonetization Token Preview"):
    if input_text.strip():
        try:
            if lang_choice == "arabic":
                preview_tokens = arabic_to_tokens(input_text)
            else:
                preview_tokens = english_to_tokens(input_text)
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
generate_btn = st.button("🚀 Generate Speech", type="primary", use_container_width=True)

if generate_btn:
    if not input_text.strip():
        st.error("Please enter some text before generating audio.")
    else:
        # Load model and vocoder
        model, ckpt_status, active_ckpt = load_jepa_model(lang_choice, custom_ckpt_path if custom_ckpt_path else None)
        vocoder = load_vocoder()

        # Step 1: Prosodic chunking
        with st.spinner("Analyzing linguistic prosody & sentence segmentation..."):
            chunks = split_into_prosodic_chunks(input_text, lang=lang_choice, max_phonemes=55, min_phonemes=18)

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

                # Audio Player & Download
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
