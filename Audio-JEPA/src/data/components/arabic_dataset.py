import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from src.data.components.text_utils import ArabicTextProcessor
import collections
import random
class AudioJEPADataset(Dataset):
    def __init__(self, hf_dataset_name="MohamedRashad/common-voice-18-arabic", split="train[:1%]", sample_rate=32000, n_mels=128):
        # Load a small slice for the baseline
        print(f"Loading dataset {hf_dataset_name} ({split})...")
        self.dataset = load_dataset(hf_dataset_name, split=split)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Audio-JEPA parameters: 32 kHz, 10s clips, 128 mel bins, 256 time bins
        # To get 256 time bins for 10s (320,000 samples), hop_length = 320000 / 256 = 1250
        self.hop_length = 1250
        self.n_fft = 2048
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        self.text_processor = ArabicTextProcessor()

        print(f"Grouping {len(self.dataset)} items by client_id for reference sampling...")
        client_ids = self.dataset['client_id']
        self.client_to_indices = collections.defaultdict(list)
        for i, cid in enumerate(client_ids):
            self.client_to_indices[cid].append(i)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # In HF datasets, 'audio' contains 'array' and 'sampling_rate' if decoding is enabled
        audio_array = torch.tensor(item['audio']['array'], dtype=torch.float32)
        orig_sr = item['audio']['sampling_rate']
        text = item["sentence"]
        
        # Resample if needed
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            audio_array = resampler(audio_array)
            
        # Pad or truncate to 10 seconds (32000 * 10 = 320000 samples)
        target_length = self.sample_rate * 10
        if audio_array.shape[0] > target_length:
            audio_array = audio_array[:target_length]
        else:
            padding = target_length - audio_array.shape[0]
            audio_array = torch.nn.functional.pad(audio_array, (0, padding))
            
        # Compute Mel Spectrogram
        mel_spec = self.mel_spectrogram(audio_array)
        # Convert to log scale (adding small constant for numerical stability)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        # Process text
        processed_text = self.text_processor.process(text)
        
        # Fetch 3-second reference audio for speaker context
        client_id = item['client_id']
        ref_idx = random.choice(self.client_to_indices[client_id])
        ref_item = self.dataset[ref_idx]
        ref_audio_array = torch.tensor(ref_item['audio']['array'], dtype=torch.float32)
        ref_sr = ref_item['audio']['sampling_rate']
        
        if ref_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, self.sample_rate)
            ref_audio_array = resampler(ref_audio_array)
            
        ref_target_length = self.sample_rate * 3  # 3 seconds
        if ref_audio_array.shape[0] > ref_target_length:
            ref_audio_array = ref_audio_array[:ref_target_length]
        else:
            padding = ref_target_length - ref_audio_array.shape[0]
            ref_audio_array = torch.nn.functional.pad(ref_audio_array, (0, padding))
            
        ref_mel_spec = self.mel_spectrogram(ref_audio_array)
        ref_log_mel_spec = torch.log(torch.clamp(ref_mel_spec, min=1e-5))

        return {
            "waveform": audio_array,
            "transformed_waveform": log_mel_spec.unsqueeze(0),
            "reference_waveform": ref_audio_array,
            "reference_transformed_waveform": ref_log_mel_spec.unsqueeze(0),
            "target": torch.zeros(1), # Dummy target
            "audio_name": "arabic_sample",
            "text": text,
            "phonemes": processed_text["phonemes"] if processed_text else ""
        }

if __name__ == "__main__":
    print("Testing Dataset Initialization...")
    ds = AudioJEPADataset(split="train[:10]")
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    
    for batch in loader:
        print("Batch Log-Mel Shape:", batch["transformed_waveform"].shape)
        print("Phonemes example:", batch["phonemes"][0])
        break
