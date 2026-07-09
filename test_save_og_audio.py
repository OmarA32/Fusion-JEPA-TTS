import os
import io
import soundfile as sf
import scipy.io.wavfile as wavfile
from datasets import load_dataset, Audio

def main():
    print("Loading HF dataset split test...")
    # Load dataset exactly as it is in dataset.py
    dataset = list(load_dataset("MohamedRashad/common-voice-18-arabic", split="test").cast_column("audio", Audio(decode=False)))
    
    item = dataset[10]
    
    audio_bytes = item['audio']['bytes']
    wav_np, sr = sf.read(io.BytesIO(audio_bytes))
    
    print(f"Original Audio Sample Rate: {sr} Hz")
    print(f"Original Audio Shape: {wav_np.shape}")
    
    out_dir = "test_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ground_truth_index_10_ORIGINAL.wav")
    
    # Save exact original array with exact original sample rate
    sf.write(out_path, wav_np, sr)
    print(f"Saved exact original audio to: {out_path}")

if __name__ == "__main__":
    main()
