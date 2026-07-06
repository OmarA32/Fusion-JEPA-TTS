import os
import sys
import torch
import torchaudio
import lightning as L

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tts-arabic-pytorch"))
sys.path.insert(1, os.path.join(PROJECT_ROOT, "Audio-JEPA"))

for mod in ["utils", "text", "models", "vocoder"]:
    if mod in sys.modules:
        sys.modules.pop(mod)

from src.data.zero_shot_datamodule import ZeroShotDataModule
from src.models.zero_shot_tts import ZeroShotTTS
from vocoder import load_hifigan

# Paths for HiFi-GAN
HIFIGAN_CONFIG = os.path.join(PROJECT_ROOT, "tts-arabic-pytorch/pretrained/hifigan-asc-v1/config.json")
HIFIGAN_WEIGHTS = os.path.join(PROJECT_ROOT, "tts-arabic-pytorch/pretrained/hifigan-asc-v1/hifigan-asc.pth")

def main():
    print("Initializing DataModule...")
    datamodule = ZeroShotDataModule(batch_size=2)
    datamodule.setup()

    print("Initializing ZeroShotTTS Model...")
    
    # Minimal config to test functionality
    jepa_kwargs = {
        "input_size": [128, 256],
        "patch_size": [16, 16],
        "in_chans": 1,
        "embed_dim": 768,
        "predictor_embed_dim": 384,
        "depth": 2,
        "predictor_depth": 2,
        "num_heads": 4,
    }
    
    from models.fastpitch import net_config
    
    model = ZeroShotTTS(
        jepa_encoder_kwargs=jepa_kwargs,
        fastpitch_kwargs=net_config,
        jepa_embed_dim=768,
        fastpitch_embed_dim=384,
        learning_rate=1e-5
    )

    # Initialize PyTorch Lightning Trainer
    # Use max_epochs=2 to quickly test the pipeline
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="auto", # Uses CUDA if available, else CPU
        devices=1,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        enable_checkpointing=True
    )

    print("Starting overfit training loop on 10 samples...")
    trainer.fit(model, datamodule)

    print("Training finished! Generating sample audio files...")
    model.eval()
    
    # Load Vocoder
    print("Loading HiFi-GAN Vocoder...")
    vocoder = load_hifigan(HIFIGAN_WEIGHTS, HIFIGAN_CONFIG)
    vocoder = vocoder.to(model.device)
    
    # Save directory
    out_dir = os.path.join(PROJECT_ROOT, "generated_samples")
    os.makedirs(out_dir, exist_ok=True)
    
    # Inference on a few samples
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))
    fp_inputs, ref_specs = batch
    
    # Move to device
    fp_inputs = tuple(t.to(model.device) if isinstance(t, torch.Tensor) else t for t in fp_inputs)
    ref_specs = ref_specs.to(model.device)
    
    with torch.no_grad():
        # model.infer() returns (mel_out, dec_lens, dur_pred, pitch_pred)
        mel_out, _, _, _, _ = model.infer(fp_inputs, ref_specs)
        
        # Convert Mel to Audio
        audio_out = vocoder(mel_out)
        audio_out = audio_out.squeeze(1).cpu() # [B, T]
        
    for i in range(audio_out.size(0)):
        sample_path = os.path.join(out_dir, f"sample_{i}.wav")
        # Save as 22050 Hz (FastPitch default output sr)
        torchaudio.save(sample_path, audio_out[i].unsqueeze(0), 22050)
        print(f"Saved generated audio to: {sample_path}")
        
    print("Done! You can listen to the generated samples in the 'generated_samples' folder.")

if __name__ == "__main__":
    main()
