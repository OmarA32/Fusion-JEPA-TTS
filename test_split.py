import argparse
import os
import sys
import torch
import scipy.io.wavfile as wavfile
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from data.dataset import JEPADataset
from models.jepat_lightning import JEPATLightning
from vocoder_manager import VocoderManager

def get_latest_checkpoint(log_dir):
    checkpoints_dir = os.path.join(log_dir, "lightning_logs")
    if not os.path.exists(checkpoints_dir):
        return None
    versions = [d for d in os.listdir(checkpoints_dir) if d.startswith("version_")]
    if not versions:
        return None
    versions.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
    latest_version = versions[0]
    ckpt_dir = os.path.join(checkpoints_dir, latest_version, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        return None
        
    # Always prioritize the 'last.ckpt' if it exists
    if "last.ckpt" in ckpts:
        return os.path.join(ckpt_dir, "last.ckpt")
        
    return os.path.join(ckpt_dir, ckpts[0])

def generate_and_save(model, vocoder, text_input, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        generated_mel = model.model.sample_tokens(
            bsz=1,
            num_iter=64,
            cfg_scale=3.0,
            labels=[text_input]
        )
    mel_for_vocoder = generated_mel.squeeze(1)
    
    with torch.no_grad():
        audio_waveform = vocoder.generate_audio(mel_for_vocoder)
        
    sample_rate = 24000
    audio_np = audio_waveform.squeeze().cpu().numpy()
    audio_np = audio_np / max(abs(audio_np).max(), 1e-8)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_int16)
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocoder", type=str, required=True, choices=["hifigan", "vocos", "bigvgan"])
    parser.add_argument("--index", type=int, default=None, help="Test a specific index only")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Test Dataset...")
    test_dataset = JEPADataset(split="test")
    
    print("Loading JEPA-T Model...")
    ckpt_path = get_latest_checkpoint("training_logs")
    if ckpt_path:
        print(f"Loading checkpoint: {ckpt_path}")
        model = JEPATLightning.load_from_checkpoint(ckpt_path)
    else:
        print("WARNING: No trained weights found! Running with completely untrained model.")
        model = JEPATLightning()
        
    model = model.to(device)
    model.eval()

    print(f"Loading {args.vocoder} Vocoder...")
    vocoder = VocoderManager(vocoder_type=args.vocoder, device=device)

    out_dir = "test_results"
    os.makedirs(out_dir, exist_ok=True)

    if args.index is not None:
        idx = args.index
        if idx >= len(test_dataset):
            print(f"Error: Index {idx} out of bounds for test dataset of size {len(test_dataset)}")
            return
        print(f"Testing only index {idx}")
        item = test_dataset.dataset[idx]
        text = item['sentence']
        output_path = os.path.join(out_dir, f"test_index_{idx}_{args.vocoder}.wav")
        print(f"Processing text: {text}")
        generate_and_save(model, vocoder, text, output_path)
    else:
        print(f"Testing full dataset ({len(test_dataset)} samples)...")
        for idx in range(len(test_dataset)):
            item = test_dataset.dataset[idx]
            text = item['sentence']
            output_path = os.path.join(out_dir, f"test_full_{idx}_{args.vocoder}.wav")
            generate_and_save(model, vocoder, text, output_path)
            
if __name__ == "__main__":
    main()
