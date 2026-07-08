import torch
import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torchaudio
import argparse
from models.jepat_lightning import JEPATLightning
from vocoder_manager import VocoderManager
from text import arabic_to_tokens, tokens_to_ids

def generate_audio(text, output_path="output_arabic_test.wav"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing JEPA-T Model...")
    # For testing the pipeline, we just use an untrained model.
    # In a real scenario, we would load from checkpoint:
    # model = JEPATLightning.load_from_checkpoint("path/to/checkpoint.ckpt")
    model = JEPATLightning()
    model = model.to(device)
    model.eval()

    print("Initializing HiFi-GAN Vocoder...")
    vocoder = VocoderManager(vocoder_type='hifigan', device=device)

    print("Processing Text...")
    # Convert Arabic text to phoneme token IDs
    try:
        phonemes = arabic_to_tokens(text)
        tokens = tokens_to_ids(phonemes)
        print(f"Token IDs: {tokens}")
    except Exception as e:
        print(f"Error processing text: {e}")
        return

    # JEPAT expects a list of token lists for text batching
    text_input = [text] 

    print("Running Diffusion Generation (This may take a minute on CPU)...")
    with torch.no_grad():
        # Generate the Mel Spectrogram
        generated_mel = model.model.sample_tokens(
            bsz=1,
            num_iter=64, # Default denoising steps
            cfg_scale=3.0,
            labels=text_input
        )
    
    print(f"Raw Generated Mel Shape: {generated_mel.shape}")
    
    # Unpatchify and adjust shape for Vocoder
    # Current shape: [1, 1, 80, 512] -> needs to be [1, 80, 512]
    mel_for_vocoder = generated_mel.squeeze(1)
    print(f"Vocoder Input Mel Shape: {mel_for_vocoder.shape}")

    print("Running Vocoder Synthesis...")
    with torch.no_grad():
        audio_waveform = vocoder.generate_audio(mel_for_vocoder)
    
    print(f"Generated Audio Shape: {audio_waveform.shape}")

    print(f"Saving to {output_path}...")
    import scipy.io.wavfile as wavfile
    import numpy as np
    
    # HiFi-GAN generates at 24000Hz
    sample_rate = 24000
    audio_np = audio_waveform.squeeze().cpu().numpy()
    
    # Normalize and convert to int16 to prevent audio clipping
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
