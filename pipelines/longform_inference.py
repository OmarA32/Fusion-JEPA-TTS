import torch
import os
import sys
import re
import argparse
import numpy as np
import scipy.io.wavfile as wavfile

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from models.jepa import JEPA_base
from models.vocoder_manager import VocoderManager
from text import (
    arabic_to_tokens, 
    english_to_tokens, 
    tokens_to_ids, 
    phon_to_id_, 
    arabic_to_phonemes,
    phonemes_to_tokens
)

def count_phonemes(text, lang="arabic"):
    """Accurately counts valid phonetic tokens for a given text snippet."""
    try:
        toks = arabic_to_tokens(text) if lang == "arabic" else english_to_tokens(text)
        return [t for t in toks if t in phon_to_id_]
    except Exception:
        return list(text)

def split_into_prosodic_chunks(text, lang="arabic", max_phonemes=90, min_phonemes=20):
    """
    4-Level Hierarchical Prosodic Chunker:
    1. Sentence boundaries (\n, ., !, ?, ؟)
    2. Clause punctuation (; : — – , ، ؛) with numeric comma (23,000) & paren (Gemma, Qwen) protection.
    3. Conjunction & Preposition acoustic phrasing (and, with, that, including, using, etc.)
    4. Indivisible parenthetical sliding-window budget ensuring no clause ever exceeds max_phonemes.
    5. Bidirectional orphan sweep ensuring natural sentence cadence and zero canvas overflow.
    """
    text = text.strip()
    if not text:
        return []

    # 1. Protect numeric commas e.g. 23,000 -> 23__NUMCOMMA__000
    text = re.sub(r'(\d),(\d)', r'\1__NUMCOMMA__\2', text)
    
    # 2. Protect parenthetical commas e.g. (Gemma, Qwen) -> (Gemma__PARENCOMMA__ Qwen)
    def protect_parens(m):
        return m.group(0).replace(',', '__PARENCOMMA__').replace('،', '__PARENCOMMA__')
    text = re.sub(r'\([^)]*\)', protect_parens, text)

    # 3. Primary Split: Sentence-level punctuation
    sentence_punct = r'[\n\.\!\?؟]+'
    raw_units = [s.strip() for s in re.split(sentence_punct, text) if s.strip()]

    def split_clause_recursive(clause_text):
        tokens = count_phonemes(clause_text, lang)
        if len(tokens) <= max_phonemes:
            return [(clause_text, tokens)]

        # Level 2: Clause punctuation
        clause_punct = r'[\,\;\:\–\—\،\؛]+'
        sub_clauses = [c.strip() for c in re.split(clause_punct, clause_text) if c.strip()]
        if len(sub_clauses) > 1:
            results = []
            for sc in sub_clauses:
                results.extend(split_clause_recursive(sc))
            return results

        # Level 3: Conjunction & Preposition Phrasing
        if lang == 'arabic':
            conj_pattern = r'\s+(و|مع|التي|الذي|من|إلى|على|حيث|لكن|أو|لأن|في)\s+'
        else:
            conj_pattern = r'\s+(and|with|that|which|including|utilizing|leveraging|using|from|while|because|but|or|where|such as)\s+'

        parts = re.split(conj_pattern, clause_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            reconstructed_segments = []
            cur_seg = parts[0].strip()
            i = 1
            while i < len(parts):
                conj = parts[i]
                next_part = parts[i+1].strip() if i+1 < len(parts) else ''
                combined_candidate = f'{cur_seg} {conj} {next_part}'.strip()
                if len(count_phonemes(combined_candidate, lang)) <= max_phonemes:
                    cur_seg = combined_candidate
                else:
                    if cur_seg:
                        reconstructed_segments.append(cur_seg)
                    cur_seg = f'{conj} {next_part}'.strip()
                i += 2
            if cur_seg:
                reconstructed_segments.append(cur_seg)

            if len(reconstructed_segments) > 1:
                results = []
                for seg in reconstructed_segments:
                    results.extend(split_clause_recursive(seg))
                return results

        # Level 4: Group words so parenthetical expressions remain indivisible units
        raw_words = clause_text.split()
        words = []
        buf = []
        in_paren = False
        for w in raw_words:
            if '(' in w and ')' not in w:
                in_paren = True
                buf.append(w)
            elif in_paren:
                buf.append(w)
                if ')' in w:
                    in_paren = False
                    words.append(' '.join(buf))
                    buf = []
            else:
                words.append(w)
        if buf:
            words.append(' '.join(buf))

        current_words = []
        current_tokens = []
        sub_chunks = []
        for word in words:
            w_toks = count_phonemes(word, lang)
            if len(current_tokens) + len(w_toks) > max_phonemes and len(current_tokens) >= min_phonemes:
                sub_chunks.append((' '.join(current_words), current_tokens))
                current_words = [word]
                current_tokens = w_toks
            else:
                current_words.append(word)
                current_tokens.extend(w_toks)

        if current_words:
            if sub_chunks and len(current_tokens) < min_phonemes:
                prev_text, prev_tokens = sub_chunks[-1]
                if len(prev_tokens) + len(current_tokens) <= max_phonemes + 10:
                    sub_chunks[-1] = (prev_text + ' ' + ' '.join(current_words), prev_tokens + current_tokens)
                else:
                    sub_chunks.append((' '.join(current_words), current_tokens))
            else:
                sub_chunks.append((' '.join(current_words), current_tokens))

        return sub_chunks

    all_chunks = []
    for unit in raw_units:
        all_chunks.extend(split_clause_recursive(unit))

    # Clean strings & restore protected tokens
    cleaned = []
    for c_text, c_tokens in all_chunks:
        c_text = c_text.replace('__NUMCOMMA__', ',').replace('__PARENCOMMA__', ',').strip()
        c_words = c_text.split()
        if c_words:
            cleaned.append((c_text, count_phonemes(c_text, lang)))

    # Level 5: Bidirectional sweep to eliminate orphan fragments (< min_phonemes or <= 2 words)
    merged = []
    i = 0
    while i < len(cleaned):
        c_text, c_tokens = cleaned[i]
        c_words = c_text.split()

        if (len(c_words) <= 2 or len(c_tokens) < min_phonemes):
            merged_backward = False
            if merged:
                prev_text, prev_tokens = merged[-1]
                if len(prev_tokens) + len(c_tokens) <= max_phonemes:
                    merged[-1] = (prev_text + ' ' + c_text, prev_tokens + c_tokens)
                    merged_backward = True

            if not merged_backward:
                if i + 1 < len(cleaned):
                    next_text, next_tokens = cleaned[i + 1]
                    if len(c_tokens) + len(next_tokens) <= max_phonemes:
                        cleaned[i + 1] = (c_text + ' ' + next_text, c_tokens + next_tokens)
                        i += 1
                        continue
                merged.append((c_text, c_tokens))
        else:
            merged.append((c_text, c_tokens))
        i += 1

    return merged

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
    # Average speaking rate: ~14-16 phonemes per second (~5.5-6.5 frames per phoneme)
    if num_phonemes is not None and num_phonemes > 0:
        min_search_frame = max(15, int(num_phonemes * 5.0))
        max_expected_frame = min(n_frames, int(num_phonemes * 7.5) + 15)
    else:
        min_search_frame = 30
        max_expected_frame = n_frames
        
    min_search_frame = min(min_search_frame, n_frames - 20)
    
    # 4. Search for sustained silence (>= 12 consecutive frames ~ 140ms)
    silence_run_required = 12
    current_run = 0
    cut_frame = max_expected_frame
    
    for t in range(min_search_frame, n_frames):
        if smoothed_energy[t] < e_thresh:
            current_run += 1
            if current_run >= silence_run_required:
                # Add a gentle 6-frame (~70ms) room decay pad
                cut_frame = min(n_frames, (t - silence_run_required) + 6)
                break
        else:
            current_run = 0
            
    # Never exceed max_expected_frame to eliminate any potential trailing hallucinations
    cut_frame = min(cut_frame, max_expected_frame)
    cut_sample = min(len(audio), int(cut_frame * hop_length))
    clean_audio = audio[:cut_sample].copy()
    
    # Smooth 20ms raised-cosine fade-out on the tail to prevent click at the cut
    fade_len = min(len(clean_audio), int(sample_rate * 0.020))
    if fade_len > 0:
        fade_curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        clean_audio[-fade_len:] *= fade_curve
        
    return clean_audio, cut_frame

def stitch_audio_segments(segments, pause_ms=100, sample_rate=44100):
    """
    Seamlessly concatenates audio segments with natural acoustic breath pauses.
    Preserves full attack/onsets of initial consonants without muting or clipping.
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)
    if len(segments) == 1:
        return segments[0]

    pause_samples = int(sample_rate * (pause_ms / 1000.0))
    pause_block = np.zeros(pause_samples, dtype=np.float32)
    
    stitched_parts = []
    for i, seg in enumerate(segments):
        if len(seg) == 0:
            continue
        # Normalize each segment to uniform amplitude
        seg_max = max(abs(seg).max(), 1e-8)
        norm_seg = seg / seg_max
        
        stitched_parts.append(norm_seg)
        if i < len(segments) - 1:
            stitched_parts.append(pause_block)

    return np.concatenate(stitched_parts)

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
    print(f"Using natively accelerated PyTorch device: {device}")

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

    # 2. Checkpoint Loading (matches inference.py exactly)
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
        print(f"Loading weights from {found_path} ({ckpt_type})...")
        if ckpt_type == "pt":
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)
            model_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    k_new = k.replace("model.", "", 1)
                    model_dict[k_new] = v
                elif not k.startswith("ema_model."):
                    model_dict[k] = v
            try:
                model.load_state_dict(model_dict, strict=False)
            except Exception as e:
                print(f"Warning during state_dict load: {e}")
    else:
        print(f"\n[WARNING] No checkpoint found in 'training_logs/{lang}'! Generating with untrained weights.")

    model.eval()

    # 3. Initialize Vocoder
    print("Loading BigVGAN Vocoder...")
    vocoder_instance = VocoderManager(device=device)

    # 4. Prosodic Chunking
    chunks = split_into_prosodic_chunks(text, lang=lang, max_phonemes=90, min_phonemes=20)
    print(f"\n[Long-Form Plan] Split input into {len(chunks)} coherent prosodic clause(s):")
    for i, (c_text, _) in enumerate(chunks):
        print(f"  Clause {i+1}/{len(chunks)}: \"{c_text}\"")
    print("")

    sample_rate = 44100
    audio_segments = []
    mel_segments = []

    # 5. Sequential Generation Pipeline (Exact inference.py parity per clause)
    for i, (chunk_text, _) in enumerate(chunks):
        print(f"--- Synthesizing Clause {i+1}/{len(chunks)}: \"{chunk_text}\" ---")
        try:
            if lang == "arabic":
                tokens = arabic_to_tokens(chunk_text)
            elif lang == "english":
                tokens = english_to_tokens(chunk_text)
            else:
                raise ValueError(f"Language {lang} not supported.")
            tokens = [t for t in tokens if t in phon_to_id_]
            tokens = tokens_to_ids(tokens)
        except Exception as e:
            print(f"Error processing text for clause {i+1}: {e}")
            continue

        text_input = [tokens]

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
                num_phonemes=len(tokens),
                sample_rate=sample_rate,
                hop_length=512
            )
            clean_mel = mel_for_vocoder[:, :, :cut_frame].squeeze(0).cpu().numpy()
        else:
            clean_audio = audio_wave.squeeze().cpu().numpy()
            clean_mel = mel_for_vocoder.squeeze(0).cpu().numpy()

        chunk_dur = len(clean_audio) / sample_rate
        print(f"Clause {i+1} Synthesized: {chunk_dur:.2f}s audio.")
        audio_segments.append(clean_audio)
        mel_segments.append(clean_mel)

    # 6. Seamless Audio Stitching
    print("\nStitching all audio segments with natural phrasing...")
    stitched_audio = stitch_audio_segments(audio_segments, pause_ms=pause_ms, sample_rate=sample_rate)

    # Normalize final audio
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

    # Ensure output path defaults into test_results/ if not specified
    if not os.path.dirname(args.output):
        args.output = os.path.join("test_results", args.output)

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
