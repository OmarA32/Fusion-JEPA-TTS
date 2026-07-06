import sys
import os
import torch

# Add paths at the very top
sys.path.insert(0, os.path.abspath("tts-arabic-pytorch"))
sys.path.append(os.path.abspath("Audio-JEPA"))

from src.models.zero_shot_tts import ZeroShotTTS
import yaml

def main():
    print("1. Loading configurations...")
    
    # Audio-JEPA config
    jepa_kwargs = {
        "input_size": [128, 256],
        "patch_size": [16, 16],
        "in_chans": 1,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0
    }
    
    # FastPitch config 
    from models.fastpitch import net_config as fp_config
    
    print("2. Instantiating ZeroShotTTS model...")
    model = ZeroShotTTS(
        jepa_encoder_kwargs=jepa_kwargs,
        fastpitch_kwargs=fp_config,
        jepa_embed_dim=768,
        fastpitch_embed_dim=fp_config["symbols_embedding_dim"]
    )
    
    print("Model instantiated successfully!")
    
    print("3. Creating mock tensors...")
    B = 2
    # Audio-JEPA reference input [Batch, Channels, Mel_Bins, Time_Bins]
    ref_spec = torch.randn(B, 1, 128, 256)
    
    # FastPitch Inputs (inputs, input_lens, mel_tgt, mel_lens, pitch_dense, energy_dense, speaker, attn_prior, audiopaths)
    text_max_len = 50
    mel_max_len = 100
    
    text_inputs = torch.randint(0, fp_config["n_symbols"], (B, text_max_len))
    input_lens = torch.tensor([text_max_len, text_max_len])
    mel_tgt = torch.randn(B, fp_config["n_mel_channels"], mel_max_len)
    mel_lens = torch.tensor([mel_max_len, mel_max_len])
    pitch_dense = torch.randn(B, 1, mel_max_len)
    energy_dense = torch.randn(B, mel_max_len)
    speaker = torch.tensor([0, 0]) # Will be replaced inside forward
    attn_prior = None
    audiopaths = ["dummy1", "dummy2"]
    
    fp_inputs = (text_inputs, input_lens, mel_tgt, mel_lens, pitch_dense, energy_dense, speaker, attn_prior, audiopaths)
    
    print("4. Executing forward pass...")
    mel_out, mel_out_postnet, dur_pred, pitch_pred, energy_pred = model(fp_inputs, ref_spec)
    
    print(f"Success! Output mel shape: {mel_out.shape}")
    print("Everything connected properly.")

if __name__ == "__main__":
    main()
