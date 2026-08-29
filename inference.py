import torch
import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torchaudio
import argparse
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from models.jepa import JEPA_base
from vocoder_manager import VocoderManager
from text import arabic_to_tokens, tokens_to_ids

def truncate_trailing_silence(audio_waveform, mel_spectrogram, num_phonemes=None, sample_rate=44100, hop_length=512):
    """
    4-Stage Adaptive Boundary Detector:
    Truncates hallucinated tail noise after the utterance naturally finishes,
    while preserving intra-sentence pauses and soft trailing consonants.
    """
    import numpy as np
    
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
    # Speech rate ~8-12 frames/phoneme; never cut before 6.5 frames/phoneme
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
    
    # Smooth 20ms raised-cosine fade-out to prevent pops/clicks
    fade_len = min(len(clean_audio), int(sample_rate * 0.020))
    if fade_len > 0:
        fade_curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        clean_audio[-fade_len:] *= fade_curve
        
    if cut_frame < n_frames:
        trimmed_sec = (n_frames - cut_frame) * hop_length / sample_rate
        print(f"[Smart Truncate] Trimmed {trimmed_sec:.2f}s of unconditioned tail noise (Sentence ended at frame {cut_frame}/{n_frames}).")
        
    return clean_audio, cut_frame

def generate_audio(text, lang="arabic", db="common_voice", output_path="output_test.wav", save_mel=False, mel_gt=None, cfg_scale=3.0, steps=60, ckpt_path=None, trim_silence=True):
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using natively accelerated PyTorch device: {device}")

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
    
    import re
    def get_latest_checkpoint(log_dir):
        if not os.path.exists(log_dir):
            return None, None, 0

        latest_ckpt = None
        max_ckpt_epoch = -1
        latest_pt = None
        max_pt_epoch = -1

        for root, dirs, files in os.walk(log_dir):
            for f in files:
                filepath = os.path.join(root, f)
                
                # Check for raw .pt files
                if f.startswith("jepa_epoch_") and f.endswith(".pt"):
                    try:
                        epoch = int(re.search(r"epoch_(\d+)", f).group(1))
                        if epoch > max_pt_epoch:
                            max_pt_epoch = epoch
                            latest_pt = filepath
                    except:
                        pass
                
                # Check for Lightning .ckpt files
                elif f.endswith(".ckpt"):
                    try:
                        if "epoch=" in f:
                            epoch = int(re.search(r"epoch=(\d+)", f).group(1))
                            if epoch > max_ckpt_epoch:
                                max_ckpt_epoch = epoch
                                latest_ckpt = filepath
                        elif f == "last.ckpt":
                            # last.ckpt takes absolute highest priority
                            max_ckpt_epoch = 99999999
                            latest_ckpt = filepath
                    except:
                        pass

        if max_ckpt_epoch == -1 and max_pt_epoch == -1:
            return None, None, 0
            
        if max_ckpt_epoch >= max_pt_epoch:
            return latest_ckpt, "ckpt", max_ckpt_epoch
        else:
            return latest_pt, "pt", max_pt_epoch

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
            # Load Lightning checkpoints by stripping 'model.' prefix
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)
            model_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    k_new = k.replace("model.", "", 1)
                    model_dict[k_new] = v
                else:
                    model_dict[k] = v
            try:
                model.load_state_dict(model_dict, strict=False)
            except Exception as e:
                print(f"Warning during state_dict load: {e}")
    else:
        print(f"\n[WARNING] No checkpoint found in 'training_logs/{lang}'!")
        print(f"To download pretrained weights, run: python download_from_hf.py --lang {lang}")
        print(f"Generating with untrained random weights (will sound noisy).\n")
        
    model.eval()

    print("Loading BigVGAN Vocoder...")
    vocoder_instance = VocoderManager(device=device)

    print("Processing Text...")
    try:
        if lang == "arabic":
            from text import arabic_to_tokens
            tokens = arabic_to_tokens(text)
        elif lang == "english":
            from text import english_to_tokens
            tokens = english_to_tokens(text)
        else:
            raise ValueError(f"Language {lang} not supported.")
            
        from text import tokens_to_ids, phon_to_id_
        tokens = [t for t in tokens if t in phon_to_id_]
        tokens = tokens_to_ids(tokens)
        print(f"Token IDs: {tokens}")
    except Exception as e:
        print(f"Error processing text: {e}")
        return

    text_input = [tokens] 

    print(f"Running Diffusion Generation ({steps} Flow Matching steps)...")
    with torch.no_grad():
        generated_mel = model.sample_tokens(
            bsz=1,
            num_iter=steps, 
            cfg_scale=cfg_scale,
            labels=text_input
        )
    
    print(f"Raw Generated Mel Shape: {generated_mel.shape}")
    
    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    
    if save_mel:
        try:
            import matplotlib.pyplot as plt
            gen_mel_np = generated_mel.squeeze().cpu().numpy()
            mel_path = output_path.replace('.wav', '_mel.png')
            
            if mel_gt is not None:
                gt_mel_np = mel_gt.squeeze().cpu().numpy()
                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                axes[0].imshow(gt_mel_np, origin='lower', aspect='auto', cmap='viridis')
                axes[0].set_title('Ground Truth Mel')
                axes[1].imshow(gen_mel_np, origin='lower', aspect='auto', cmap='viridis')
                axes[1].set_title('Generated Mel')
                plt.tight_layout()
                plt.savefig(mel_path)
            else:
                plt.figure(figsize=(10, 4))
                plt.imshow(gen_mel_np, origin='lower', aspect='auto', cmap='viridis')
                plt.title('Generated Mel Spectrogram')
                plt.tight_layout()
                plt.savefig(mel_path)
                
            plt.close()
            print(f"Saved Mel Spectrogram plot to: {mel_path}")
        except ImportError:
            print("WARNING: matplotlib is not installed. Cannot save mel spectrogram image. Please run: pip install matplotlib")

    
    mel_for_vocoder = generated_mel.squeeze(1)
    print(f"Vocoder Input Mel Shape: {mel_for_vocoder.shape}")

    print("Running Vocoder Synthesis...")
    with torch.no_grad():
        audio_waveform = vocoder_instance.generate_audio(mel_for_vocoder)
    
    print(f"Generated Audio Shape: {audio_waveform.shape}")

    print(f"Saving to {output_path}...")
    import scipy.io.wavfile as wavfile
    import numpy as np
    
    sample_rate = 44100
    
    if trim_silence:
        audio_np, cut_frame = truncate_trailing_silence(
            audio_waveform=audio_waveform,
            mel_spectrogram=mel_for_vocoder,
            num_phonemes=len(tokens),
            sample_rate=sample_rate,
            hop_length=512
        )
    else:
        audio_np = audio_waveform.squeeze().cpu().numpy()
    
    audio_np = audio_np / max(abs(audio_np).max(), 1e-8)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    wavfile.write(output_path, sample_rate, audio_int16)
    print(f"Synthesis Complete! Audio saved to: {output_path} ({len(audio_np)/sample_rate:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA TTS Inference")
    parser.add_argument("--text", type=str, default="مَرْحَبَاً بِكُمْ فِي هَذَا الِاخْتِبَار", help="Text to synthesize")
    parser.add_argument("--output", "--file_name", dest="output", type=str, default=None, help="Output WAV file path or filename")
    parser.add_argument("--lang", type=str, default="arabic", choices=["arabic", "english"], help="Language of the model.")
    parser.add_argument("--db", type=str, default=None, choices=["common_voice", "nawar_halabi", "clartts", "libritts", "ljspeech"], help="Database to fetch index from (only used with --index).")
    parser.add_argument("--index", type=int, default=None, help="Optionally fetch text directly from the test dataset by index.")
    parser.add_argument("--ckpt", type=str, default=None, help="Explicit path to a .ckpt or .pt model checkpoint.")
    parser.add_argument('--cfg-scale', type=float, default=7.0, help="Classifier-Free Guidance scale (1.0 disables it).")
    parser.add_argument('--steps', type=int, default=60, help="Number of diffusion steps for Flow Matching.")
    parser.add_argument('--save-mel', action='store_true', help="Save an image of the mel spectrogram(s)")
    parser.add_argument('--no-trim', action='store_true', help="Disable automatic post-speech silence trimming")
    args = parser.parse_args()

    os.makedirs("test_results", exist_ok=True)
    
    mel_gt = None
    if args.index is not None:
        valid_dbs = {
            "arabic": ["common_voice", "nawar_halabi", "clartts"],
            "english": ["libritts", "ljspeech"]
        }
        # Default database if not explicitly set
        if args.db is None:
            args.db = "nawar_halabi" if args.lang == "arabic" else "ljspeech"
            
        if args.db not in valid_dbs[args.lang]:
            print(f"\n[ERROR] Language/Database mismatch! You cannot use database '{args.db}' with language '{args.lang}'.")
            print(f"Valid databases for {args.lang} are: {', '.join(valid_dbs[args.lang])}\n")
            sys.exit(1)

        from data.dataset import JEPADataset
        print(f"Loading {args.lang.upper()} {args.db.upper()} test dataset to fetch index {args.index}...")
        test_dataset = JEPADataset(split="test", lang=args.lang, db=args.db)
        if args.index >= len(test_dataset):
            print(f"[ERROR] Index {args.index} is out of bounds! The dataset only has {len(test_dataset)} items.")
            sys.exit(1)
            
        # Extract original text directly from dataset object
        item_raw = test_dataset.dataset[args.index]
        args.text = item_raw.get('sentence', item_raw.get('text', ''))
        print(f"\n[Index {args.index} Text]: {args.text}\n")
        
        if args.save_mel:
            print("Fetching ground truth mel spectrogram...")
            item_processed = test_dataset[args.index]
            mel_gt = item_processed['mel_tgt']
        
        # Default output naming for index if no custom output filename provided
        if not args.output:
            args.output = f"test_results/inference_index_{args.index}.wav"
        elif not os.path.dirname(args.output):
            args.output = os.path.join("test_results", args.output)
    else:
        # Default output naming for custom text if not provided
        if not args.output:
            args.output = "test_results/output_test.wav"
        elif not os.path.dirname(args.output):
            args.output = os.path.join("test_results", args.output)
    
    generate_audio(
        text=args.text, 
        lang=args.lang, 
        db=args.db or ("nawar_halabi" if args.lang == "arabic" else "ljspeech"), 
        output_path=args.output, 
        save_mel=args.save_mel, 
        mel_gt=mel_gt, 
        cfg_scale=args.cfg_scale, 
        steps=args.steps,
        ckpt_path=args.ckpt,
        trim_silence=not args.no_trim
    )
