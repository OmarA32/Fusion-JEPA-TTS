import os
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from data.dataset import JEPADataset
from vocoder_manager import VocoderManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocoder", type=str, required=True, choices=["hifigan", "vocos", "bigvgan"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading test dataset using native synchronized defaults...")
    test_dataset = JEPADataset(split="test")
    item = test_dataset[10]
    
    mel_tgt = item['mel_tgt']
    if mel_tgt.dim() == 4:
        mel_tgt = mel_tgt.squeeze(0)
        
    print(f"Ground truth mel-spectrogram shape: {mel_tgt.shape}")
    
    print(f"Loading {args.vocoder} Vocoder...")
    vocoder = VocoderManager(vocoder_type=args.vocoder, device=device)
    
    print("Passing ground-truth mel directly through vocoder...")
    with torch.no_grad():
        audio_waveform = vocoder.generate_audio(mel_tgt.to(device))
        
    audio_np = audio_waveform.squeeze().cpu().numpy()
    audio_np = audio_np / max(abs(audio_np).max(), 1e-8)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    out_dir = "test_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ground_truth_index_10_{args.vocoder}.wav")
    
    wavfile.write(out_path, 24000, audio_int16)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
