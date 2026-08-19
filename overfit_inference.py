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

from models.jepat import JEPAT_base
from vocoder_manager import VocoderManager
from text import arabic_to_tokens, tokens_to_ids

def generate_audio(text, lang="arabic", db="common_voice", output_path="output_test.wav", save_mel=False, mel_gt=None, cfg_scale=3.0, steps=60):
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using natively accelerated PyTorch device: {device}")

    print("Initializing JEPA-T Model...")
    model = JEPAT_base(
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
            if "checkpoints" in root:
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

    log_dir = os.path.join("training_logs", f"overfit_{lang}")
    found_path, ckpt_type, _ = get_latest_checkpoint(log_dir)
    if found_path and os.path.exists(found_path):
        print(f"Loading weights from {found_path} ({ckpt_type})...")
        if ckpt_type == "pt":
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            state_dict = ckpt['state_dict']
            model_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    k_new = k.replace("model.", "", 1)
                    model_dict[k_new] = v
            try:
                model.load_state_dict(model_dict, strict=False)
            except Exception as e:
                print(f"Warning during state_dict load: {e}")
    else:
        print(f"WARNING: No checkpoint found! Generating with untrained weights.")
        
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

    print("Running Diffusion Generation (This may take a minute on CPU)...")
    with torch.no_grad():
        generated_mel = model.sample_tokens(
            bsz=1,
            num_iter=64, 
            cfg_scale=cfg_scale,
            labels=text_input
        )
    
    print(f"Raw Generated Mel Shape: {generated_mel.shape}")
    
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
    audio_np = audio_waveform.squeeze().cpu().numpy()
    
    audio_np = audio_np / max(abs(audio_np).max(), 1e-8)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    wavfile.write(output_path, sample_rate, audio_int16)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA-T TTS Inference")
    parser.add_argument("--text", type=str, default="مَرْحَبَاً بِكُمْ فِي هَذَا الِاخْتِبَار", help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output_test.wav", help="Output WAV file path")
    parser.add_argument("--lang", type=str, default="arabic", choices=["arabic", "english"], help="Language of the model.")
    parser.add_argument("--db", type=str, default="nawar_halabi", choices=["common_voice", "nawar_halabi", "clartts", "libritts", "ljspeech"], help="Database the model was trained on.")
    parser.add_argument("--index", type=int, default=None, help="Optionally fetch text directly from the test dataset by index.")
    parser.add_argument('--cfg-scale', type=float, default=7.0, help="Classifier-Free Guidance scale (1.0 disables it).")
    parser.add_argument('--steps', type=int, default=60, help="Number of diffusion steps for Flow Matching.")
    parser.add_argument('--save-mel', action='store_true', help="Save an image of the mel spectrogram(s)")
    args = parser.parse_args()

    valid_dbs = {
        "arabic": ["common_voice", "nawar_halabi", "clartts"],
        "english": ["libritts", "ljspeech"]
    }
    if args.db not in valid_dbs[args.lang]:
        print(f"\n[ERROR] Language/Database mismatch! You cannot use database '{args.db}' with language '{args.lang}'.")
        print(f"Valid databases for {args.lang} are: {', '.join(valid_dbs[args.lang])}\n")
        sys.exit(1)
        
    if args.index is not None:
        from data.dataset import JEPADataset
        import os
        print(f"Loading {args.lang.upper()} {args.db.upper()} test dataset to fetch index {args.index}...")
        test_dataset = JEPADataset(split="test", lang=args.lang, db=args.db)
        if args.index >= len(test_dataset):
            print(f"[ERROR] Index {args.index} is out of bounds! The dataset only has {len(test_dataset)} items.")
            sys.exit(1)
            
        # Extract original text directly from HuggingFace dataset object
        item_raw = test_dataset.dataset[args.index]
        args.text = item_raw.get('sentence', item_raw.get('text', ''))
        print(f"\n[Index {args.index} Text]: {args.text}\n")
        
        mel_gt = None
        if args.save_mel:
            print("Fetching ground truth mel spectrogram...")
            item_processed = test_dataset[args.index]
            mel_gt = item_processed['mel_tgt']
        
        os.makedirs("test_results", exist_ok=True)
        args.output = f"test_results/inference_index_{args.index}.wav"
    else:
        mel_gt = None
    
    generate_audio(args.text, args.lang, args.db, args.output, args.vocoder, args.save_mel, mel_gt, args.cfg_scale, args.steps)
