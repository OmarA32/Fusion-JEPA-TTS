import os
import sys
import torch
import torch.nn as nn
from functools import partial
import torchaudio
import argparse
import lightning as L

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tts-arabic-pytorch"))
sys.path.insert(1, os.path.join(PROJECT_ROOT, "Audio-JEPA"))

for mod in ["utils", "text", "models", "vocoder"]:
    if mod in sys.modules:
        sys.modules.pop(mod)

from src.models.zero_shot_tts import ZeroShotTTS
from src.data.zero_shot_datamodule import ZeroShotDataModule
from vocoder import load_hifigan

HIFIGAN_CONFIG = os.path.join(PROJECT_ROOT, "tts-arabic-pytorch/pretrained/hifigan-asc-v1/config.json")
HIFIGAN_WEIGHTS = os.path.join(PROJECT_ROOT, "tts-arabic-pytorch/pretrained/hifigan-asc-v1/hifigan-asc.pth")

def main(ckpt_path, out_dir):
    print(f"Initializing DataModule...")
    datamodule = ZeroShotDataModule(batch_size=1)
    datamodule.setup('test')
    
    if ckpt_path.lower() != 'none':
        print(f"Loading checkpoint from {ckpt_path}")
        model = ZeroShotTTS.load_from_checkpoint(ckpt_path)
    else:
        print("Using untrained model without checkpoint...")
        from models.fastpitch import net_config
        fastpitch_kwargs = net_config
        jepa_encoder_kwargs = {
            "input_size": [128, 256],
            "patch_size": [16, 16],
            "in_chans": 1,
            "embed_dim": 768,
            "depth": 2,
            "num_heads": 4,
        }
        jepa_ckpt = os.path.join(PROJECT_ROOT, "Audio-JEPA", "logs", "train", "runs", "2026-07-06_18-09-56", "checkpoints", "last.ckpt")
        model = ZeroShotTTS(
            jepa_encoder_kwargs=jepa_encoder_kwargs,
            fastpitch_kwargs=fastpitch_kwargs,
            jepa_embed_dim=768,
            jepa_checkpoint_path=jepa_ckpt
        )
    
    model.eval()
    
    print("Loading HiFi-GAN Vocoder...")
    vocoder = load_hifigan(HIFIGAN_WEIGHTS, HIFIGAN_CONFIG)
    vocoder = vocoder.to(model.device)
    
    os.makedirs(out_dir, exist_ok=True)
    
    val_loader = datamodule.val_dataloader()
    
    print("Generating 10 samples...")
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            fp_inputs, ref_specs = batch
            
            # Move to device
            fp_inputs = tuple(t.to(model.device) if isinstance(t, torch.Tensor) else t for t in fp_inputs)
            ref_specs = ref_specs.to(model.device)
            
            # Inference
            # model.infer requires input text tokens (fp_inputs[0]) and reference spectrogram
            mel_out, mel_lens, *_ = model.infer(fp_inputs, ref_specs)
            
            # Vocode
            audio_out = vocoder(mel_out)
            audio_out = audio_out.squeeze(1).cpu() # [1, T]
            
            sample_path = os.path.join(out_dir, f"sample_{count}.wav")
            torchaudio.save(sample_path, audio_out, 22050)
            print(f"Saved generated audio to: {sample_path}")
            
            count += 1
            if count >= 10:
                break
                
    print(f"Successfully generated {count} samples in '{out_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, default='generated_samples')
    args = parser.parse_args()
    
    main(args.ckpt, args.out)
