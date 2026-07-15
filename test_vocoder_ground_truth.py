import os
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from data.dataset import JEPADataset
from vocoder_manager import VocoderManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocoder", type=str, default="vocos", choices=["vocos"])
    parser.add_argument("--lang", type=str, default="arabic", choices=["arabic", "english"])
    parser.add_argument("--db", type=str, default="common_voice", choices=["common_voice", "nawar_halabi", "libritts", "ljspeech"])
    args = parser.parse_args()
    
    valid_dbs = {
        "arabic": ["common_voice", "nawar_halabi"],
        "english": ["libritts", "ljspeech"]
    }
    if args.db not in valid_dbs[args.lang]:
        print(f"\n[ERROR] Language/Database mismatch! You cannot use database '{args.db}' with language '{args.lang}'.")
        print(f"Valid databases for {args.lang} are: {', '.join(valid_dbs[args.lang])}\n")
        import sys
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {args.lang.upper()} {args.db.upper()} test dataset...")
    test_dataset = JEPADataset(split="test", lang=args.lang, db=args.db)
    # Grab the 10th item for testing
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
