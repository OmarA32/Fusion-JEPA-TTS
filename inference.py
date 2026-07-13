import torch
import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torchaudio
import argparse
from models.jepat import JEPAT_base
from vocoder_manager import VocoderManager
from text import arabic_to_tokens, tokens_to_ids

def generate_audio(text, output_path="output_arabic_test.wav"):
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print("Initializing JEPA-T Model...")
    model = JEPAT_base(
        in_channels=1, 
        language='arabic',
        spec_height=100, 
        spec_width=512,
        diffloss='flow', 
        jepaloss='jepa'
    ).to(device)
    
    import re
    def get_latest_checkpoint(log_dir):
        """Finds the most recent checkpoint recursively inside the log directory."""
        latest_pt = None
        max_pt_epoch = -1
        latest_ckpt = None
        max_ckpt_epoch = -1
        
        if os.path.exists(log_dir):
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
            return None, None
            
        if max_ckpt_epoch >= max_pt_epoch:
            return latest_ckpt, "ckpt"
        else:
            return latest_pt, "pt"

    found_path, ckpt_type = get_latest_checkpoint("training_logs")
    if found_path and os.path.exists(found_path):
        print(f"Loading weights from {found_path} ({ckpt_type})...")
        if ckpt_type == "pt":
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            # We must load Lightning checkpoints robustly by stripping 'model.' prefix
            ckpt = torch.load(found_path, map_location=device, weights_only=False)
            state_dict = ckpt['state_dict']
            model_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    model_dict[k.replace("model.", "", 1)] = v
            model.load_state_dict(model_dict)
    else:
        print(f"WARNING: No checkpoint found! Generating with untrained weights.")
        
    model.eval()

    print("Initializing Vocos Vocoder (100-mel 24kHz)...")
    vocoder = VocoderManager(vocoder_type='vocos', device=device)

    print("Processing Text...")
    try:
        phonemes = arabic_to_tokens(text)
        tokens = tokens_to_ids(phonemes)
        print(f"Token IDs: {tokens}")
    except Exception as e:
        print(f"Error processing text: {e}")
        return

    text_input = [text] 

    print("Running Diffusion Generation (This may take a minute on CPU)...")
    with torch.no_grad():
        generated_mel = model.sample_tokens(
            bsz=1,
            num_iter=64, 
            cfg_scale=3.0,
            labels=text_input
        )
    
    print(f"Raw Generated Mel Shape: {generated_mel.shape}")
    
    mel_for_vocoder = generated_mel.squeeze(1)
    print(f"Vocoder Input Mel Shape before padding: {mel_for_vocoder.shape}")
    
    import torch.nn.functional as F
    # The diffusion model operates in 16x16 patches, meaning it truncates 100 to 96.
    # We must restore the top 4 high-frequency bins (with silence) so Vocos can process it.
    mel_for_vocoder = F.pad(mel_for_vocoder, (0, 0, 0, 4), mode='constant', value=-11.5129)
    print(f"Vocoder Input Mel Shape after padding: {mel_for_vocoder.shape}")

    print("Running Vocoder Synthesis...")
    with torch.no_grad():
        audio_waveform = vocoder.generate_audio(mel_for_vocoder)
    
    print(f"Generated Audio Shape: {audio_waveform.shape}")

    print(f"Saving to {output_path}...")
    import scipy.io.wavfile as wavfile
    import numpy as np
    
    sample_rate = 24000
    audio_np = audio_waveform.squeeze().cpu().numpy()
    
    audio_np = audio_np / max(abs(audio_np).max(), 1e-8)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    wavfile.write(output_path, sample_rate, audio_int16)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA-T TTS Inference")
    parser.add_argument("--text", type=str, default="مَرْحَبَاً بِكُمْ فِي هَذَا الِاخْتِبَار", help="Arabic text to synthesize")
    parser.add_argument("--output", type=str, default="output_arabic_test.wav", help="Output WAV file path")
    args = parser.parse_args()

    generate_audio(args.text, args.output)
