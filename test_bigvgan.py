import os
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from data.dataset import JEPADataset
from transformers import AutoModel
import sys

def main():
    print("Loading test dataset for Nawar Halabi...")
    test_dataset = JEPADataset(split="test", lang="arabic", db="nawar_halabi")
    
    index = 20
    raw_item = test_dataset.dataset[index]
    
    import soundfile as sf
    wav_np, sr = sf.read(raw_item['audio_path'])
    wav = torch.tensor(wav_np, dtype=torch.float32)
    if len(wav.shape) > 1:
        wav = wav[:, 0]
    if sr != 24000:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, 24000)
    
    # We only take 512 frames max
    wav = wav[:(512 - 1) * 256].unsqueeze(0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading official NVIDIA BigVGAN 24kHz 100-band model...")
    # Instantiate BigVGAN from HuggingFace directly using transformers
    try:
        sys.path.append('BigVGAN')
        from bigvgan import BigVGAN
        model = BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
    except Exception as e:
        print(f"Failed to load BigVGAN: {e}")
        sys.exit(1)
        
    # **CRITICAL FIX**: Extract the Mel using BigVGAN's own exact feature extractor
    from meldataset import mel_spectrogram
    mel_tgt_log_magnitude = mel_spectrogram(
        wav.to(device),
        model.h.n_fft,
        model.h.num_mels,
        model.h.sampling_rate,
        model.h.hop_size,
        model.h.win_size,
        model.h.fmin,
        model.h.fmax
    )
        
    model.eval()
    model.to(device)
    mel_tgt_log_magnitude = mel_tgt_log_magnitude.to(device)
    
    print("Synthesizing audio with BigVGAN...")
    with torch.no_grad():
        audio_waveform = model(mel_tgt_log_magnitude)
        
    audio_np = audio_waveform.squeeze().cpu().numpy()
    audio_np = audio_np / max(abs(audio_np).max(), 1e-8)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    out_dir = "test_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ground_truth_index_{index}_bigvgan.wav")
    wavfile.write(out_path, 24000, audio_int16)
    
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
