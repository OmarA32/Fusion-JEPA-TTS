import torch
import os
import sys
import re
import argparse
import numpy as np
import scipy.io.wavfile as wavfile

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from models.jepa import JEPA_base
from vocoder_manager import VocoderManager
from text import (
    arabic_to_tokens, 
    english_to_tokens, 
    tokens_to_ids, 
    phon_to_id_, 
    arabic_to_phonemes,
    phonemes_to_tokens
)

def split_into_prosodic_chunks(text, lang="arabic", max_phonemes=30, min_phonemes=12):
    """
    Splits long text or phoneme streams into natural prosodic clauses.
    Handles punctuated text, unpunctuated text, and raw phoneme streams.
    Ensures no word is ever split in half.
    """
    text = text.strip()
    if not text:
        return []

    # 1. First-pass: Split by standard punctuation if available
    punct_pattern = r'[\n\.\!\?\,\;\:\–\—\،\؛\؟]+'
    raw_clauses = [c.strip() for c in re.split(punct_pattern, text) if c.strip()]
    
    if not raw_clauses:
        raw_clauses = [text]

    chunks = []

    for clause in raw_clauses:
        words = clause.split()
        if not words:
            continue

        current_words = []
        current_tokens = []

        for word in words:
            # Measure phonemes for this word
            try:
                if lang == "arabic":
                    word_tokens = arabic_to_tokens(word)
                else:
                    word_tokens = english_to_tokens(word)
                valid_word_tokens = [t for t in word_tokens if t in phon_to_id_]
            except Exception:
                valid_word_tokens = list(word)

            # If adding this word exceeds max_phonemes and we have enough tokens, create a chunk
            if len(current_tokens) + len(valid_word_tokens) > max_phonemes and len(current_tokens) >= min_phonemes:
                chunk_text = " ".join(current_words)
                chunks.append((chunk_text, current_tokens))
                current_words = [word]
                current_tokens = valid_word_tokens
            else:
                current_words.append(word)
                current_tokens.extend(valid_word_tokens)

        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append((chunk_text, current_tokens))

    return chunks

def truncate_trailing_silence(audio_waveform, mel_spectrogram, num_phonemes=None, sample_rate=44100, hop_length=512):
    """
    4-Stage Adaptive Boundary Detector:
    Truncates hallucinated tail noise after the utterance naturally finishes,
    while preserving intra-sentence pauses and soft trailing consonants.
    """
    if isinstance(audio_waveform, torch.Tensor):
        audio = audio_waveform.detach().cpu().squeeze().numpy()
    else:
        audio = np.squeeze(audio_waveform)
        
    if isinstance(mel_spectrogram, torch.Tensor):
        mel = mel_spectrogram.detach().cpu().squeeze().numpy()
    else:
        mel = np.squeeze(mel_spectrogram)
        
    n_frames = mel.shape[-1]
    
    # 1. Compute frame energy in linear power domain
    mel_linear = np.exp(mel) if mel.min() < 0 else mel
    frame_energy = np.mean(mel_linear, axis=0)
    
    # 5-frame moving average (~58ms smoothing)
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_energy = np.convolve(frame_energy, kernel, mode='same')
    
    # 2. Dynamic thresholding adapting to speaker level and noise floor
    e_peak = np.percentile(smoothed_energy, 95)
    e_floor = np.percentile(smoothed_energy, 5)
    e_thresh = e_floor + 0.08 * (e_peak - e_floor)
    
    # 3. Linguistic Duration Anchor (Phoneme prior)
    if num_phonemes is not None and num_phonemes > 0:
        min_search_frame = max(20, int(num_phonemes * 6.5))
    else:
        min_search_frame = 30
        
    min_search_frame = min(min_search_frame, n_frames - 30)
    
    # 4. Search for sustained silence (>= 18 consecutive frames ~ 210ms)
    silence_run_required = 18
    current_run = 0
    cut_frame = n_frames
    
    for t in range(min_search_frame, n_frames):
        if smoothed_energy[t] < e_thresh:
            current_run += 1
            if current_run >= silence_run_required:
                # Add a gentle 8-frame (~90ms) room decay pad
                cut_frame = min(n_frames, (t - silence_run_required) + 8)
                break
        else:
            current_run = 0
            
    cut_sample = min(len(audio), int(cut_frame * hop_length))
    clean_audio = audio[:cut_sample].copy()
    
    # Smooth 20ms raised-cosine fade-out
    fade_len = min(len(clean_audio), int(sample_rate * 0.020))
    if fade_len > 0:
        fade_curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        clean_audio[-fade_len:] *= fade_curve
        
    return clean_audio, cut_frame

def stitch_audio_segments(segments, pause_ms=100, crossfade_ms=15, sample_rate=44100):
    """
    Seamlessly concatenates audio segments with natural acoustic breath pauses
    and raised-cosine crossfading at chunk boundaries.
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)
    if len(segments) == 1:
        return segments[0]

    pause_samples = int(sample_rate * (pause_ms / 1000.0))
    crossfade_samples = int(sample_rate * (crossfade_ms / 1000.0))
    
    stitched = segments[0]

    for seg in segments[1:]:
        if len(seg) == 0:
            continue
        
        # 1. Insert pause
        pause_block = np.zeros(pause_samples, dtype=np.float32)
        
        # 2. Crossfade junction if long enough
        if len(stitched) >= crossfade_samples and len(seg) >= crossfade_samples:
            fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, crossfade_samples)))
            fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, crossfade_samples)))
            
            # Apply fade to previous tail and new head
            stitched[-crossfade_samples:] = stitched[-crossfade_samples:] * fade_out
            seg_head = seg[:crossfade_samples] * fade_in
            seg_rest = seg[crossfade_samples:]
            
            stitched = np.concatenate([stitched, pause_block, seg_head, seg_rest])
        else:
            stitched = np.concatenate([stitched, pause_block, seg])

    return stitched

def generate_longform_speech(
    text, 
    lang="arabic", 
    output_path="test_results/longform_output.wav", 
    cfg_scale=7.0, 
    steps=60, 
    ckpt_path=None, 
    pause_ms=100, 
    save_mel=False,
    trim_silence=True
):
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model
    print("Initializing JEPA Model...")
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

    # 2. Checkpoint Loading
    import re
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
                    except: pass
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
                    except: pass
        if max_ckpt_epoch == -1 and max_pt_epoch == -1:
            return None, None, 0
        return (latest_ckpt, "ckpt", max_ckpt_epoch) if max_ckpt_epoch >= max_pt_epoch else (latest_pt, "pt", max_pt_epoch)

    if ckpt_path and os.path.exists(ckpt_path):
        found_path = ckpt_path
        ckpt_type = "pt" if ckpt_path.endswith(".pt") else "ckpt"
    else:
        log_dir = os.path.join("training_logs", lang)
        found_path, ckpt_type, _ = get_latest_checkpoint(log_dir)

    if found_path and os.path.exists(found_path):
        print(f"Loading checkpoint from {found_path} ({ckpt_type})...")
        if ckpt_type == "pt":
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)
            model_dict = {}
            for k, v in state_dict.items():
                k_new = k.replace("model.", "", 1) if k.startswith("model.") else k
                model_dict[k_new] = v
            model.load_state_dict(model_dict, strict=False)
    else:
        print(f"\n[WARNING] No checkpoint found in 'training_logs/{lang}'! Generating with untrained weights.")

    model.eval()

    # 3. Initialize Vocoder
    print("Loading BigVGAN Vocoder...")
    vocoder_instance = VocoderManager(device=device)

    # 4. Prosodic Chunking
    chunks = split_into_prosodic_chunks(text, lang=lang, max_phonemes=30, min_phonemes=12)
    print(f"\n[Long-Form Plan] Split input into {len(chunks)} prosodic clause(s):")
    for i, (c_text, c_tokens) in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)} ({len(c_tokens)} phonemes): \"{c_text}\"")
    print("")

    sample_rate = 44100
    audio_segments = []
    mel_segments = []

    # 5. Sequential Generation Pipeline
    for i, (chunk_text, chunk_tokens) in enumerate(chunks):
        print(f"--- Synthesizing Chunk {i+1}/{len(chunks)} ---")
        token_ids = tokens_to_ids(chunk_tokens)
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
            audio_wave = vocoder_instance.generate_audio(mel_for_vocoder)

        if trim_silence:
            clean_audio, cut_frame = truncate_trailing_silence(
                audio_waveform=audio_wave,
                mel_spectrogram=mel_for_vocoder,
                num_phonemes=len(chunk_tokens),
                sample_rate=sample_rate,
                hop_length=512
            )
            clean_mel = mel_for_vocoder[:, :, :cut_frame].squeeze(0).cpu().numpy()
        else:
            clean_audio = audio_wave.squeeze().cpu().numpy()
            clean_mel = mel_for_vocoder.squeeze(0).cpu().numpy()

        chunk_dur = len(clean_audio) / sample_rate
        print(f"Chunk {i+1} Synthesized: {chunk_dur:.2f}s audio.")
        audio_segments.append(clean_audio)
        mel_segments.append(clean_mel)

    # 6. Seamless Audio Stitching
    print("\nStitching all audio segments with natural phrasing and crossfades...")
    stitched_audio = stitch_audio_segments(audio_segments, pause_ms=pause_ms, sample_rate=sample_rate)

    # Normalize audio
    stitched_audio = stitched_audio / max(abs(stitched_audio).max(), 1e-8)
    audio_int16 = (stitched_audio * 32767).astype(np.int16)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    wavfile.write(output_path, sample_rate, audio_int16)
    total_dur = len(stitched_audio) / sample_rate
    print(f"\n==============================================================================")
    print(f"🎉 Long-Form Synthesis Complete!")
    print(f"Total Duration: {total_dur:.2f} seconds ({len(chunks)} clauses concatenated)")
    print(f"Saved to: {output_path}")
    print(f"==============================================================================\n")

    # 7. Optional Stitched Mel Plot
    if save_mel:
        try:
            import matplotlib.pyplot as plt
            mel_stitched = np.concatenate(mel_segments, axis=1)
            mel_plot_path = output_path.replace('.wav', '_mel.png')
            plt.figure(figsize=(14, 4))
            plt.imshow(mel_stitched, origin='lower', aspect='auto', cmap='viridis')
            plt.title(f'Stitched Long-Form Mel-Spectrogram ({total_dur:.2f}s)')
            plt.xlabel('Frames')
            plt.ylabel('Mel Frequency Bins')
            plt.tight_layout()
            plt.savefig(mel_plot_path, dpi=300)
            plt.close()
            print(f"Saved stitched Mel-Spectrogram to: {mel_plot_path}")
        except Exception as e:
            print(f"Could not save Mel plot: {e}")

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion-JEPA Long-Form Speech Synthesis")
    parser.add_argument("--text", type=str, default=None, help="Long text string or paragraph to synthesize.")
    parser.add_argument("--file", type=str, default=None, help="Path to a text file containing the long paragraph.")
    parser.add_argument("--output", "--file_name", dest="output", type=str, default="test_results/longform_output.wav", help="Output WAV file path.")
    parser.add_argument("--lang", type=str, default="arabic", choices=["arabic", "english"], help="Language of the model.")
    parser.add_argument("--ckpt", type=str, default=None, help="Explicit path to model checkpoint.")
    parser.add_argument("--cfg-scale", type=float, default=7.0, help="Classifier-Free Guidance scale.")
    parser.add_argument("--steps", type=int, default=60, help="Diffusion steps per clause.")
    parser.add_argument("--pause-ms", type=int, default=100, help="Pause duration between clauses in milliseconds.")
    parser.add_argument("--save-mel", action="store_true", help="Save stitched Mel-spectrogram comparison image.")
    parser.add_argument("--no-trim", action="store_true", help="Disable per-chunk silence trimming.")
    args = parser.parse_args()

    # Load text from argument or file
    if args.file and os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            raw_text = f.read()
    elif args.text:
        raw_text = args.text
    else:
        if args.lang == "arabic":
            raw_text = "يَعْتَمِدُ نِظَامُ فُيُوجِن جِيبَا عَلَى التَّعَلُّمِ الذَّاتِيِّ لِتَوْلِيدِ صَوْتٍ عَالِي الْجَوْدَةِ، وَيَتَمَيَّزُ بِقُدْرَتِهِ عَلَى مُعَالَجَةِ النُّصُوصِ الْعَرَبِيَّةِ الْمُعَقَّدَةِ بِكُلِّ دِقَّةٍ وَوُضُوحٍ."
        else:
            raw_text = "Fusion-JEPA is a deep multimodal architecture designed for expressive text-to-speech synthesis, achieving studio-quality audio through continuous flow matching and joint-embedding representations."

    generate_longform_speech(
        text=raw_text,
        lang=args.lang,
        output_path=args.output,
        cfg_scale=args.cfg_scale,
        steps=args.steps,
        ckpt_path=args.ckpt,
        pause_ms=args.pause_ms,
        save_mel=args.save_mel,
        trim_silence=not args.no_trim
    )
