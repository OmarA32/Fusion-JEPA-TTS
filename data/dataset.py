import os
import sys
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio

# Ensure local imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from text import arabic_to_phonemes, phonemes_to_tokens, tokens_to_ids, phon_to_id_

class JEPADataset(Dataset):
    def __init__(self, split="train[:100]", jepa_sr=24000, max_frames=512, n_mels=100):
        super().__init__()
        print(f"Loading HF dataset split {split}...")
        self.dataset = list(load_dataset("MohamedRashad/common-voice-18-arabic", split=split).cast_column("audio", Audio(decode=False)))
        self.jepa_sr = jepa_sr
        self.max_frames = max_frames
        
        # Extract mel spectrograms dynamically
        self.jepa_mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=jepa_sr,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            f_min=0,
            f_max=12000
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # --- Audio Processing ---
        import io
        import soundfile as sf
        audio_bytes = item['audio']['bytes']
        wav_np, sr = sf.read(io.BytesIO(audio_bytes))
        wav = torch.tensor(wav_np, dtype=torch.float32)
        
        if sr != self.jepa_sr:
            wav = torchaudio.functional.resample(wav, sr, self.jepa_sr)
            
        # JEPA expects input_size=[128, 512], which means 512 frames.
        # With hop_length=320, 512 frames = (512 - 1) * 320 = 163520 samples.
        target_len = (self.max_frames - 1) * self.jepa_mel_fn.hop_length
        if len(wav) > target_len:
            wav = wav[:target_len]
        else:
            wav = F.pad(wav, (0, target_len - len(wav)))
            
        jepa_mel = self.jepa_mel_fn(wav).unsqueeze(0) # [1, 128, T]
        jepa_mel = torch.log(torch.clamp(jepa_mel, min=1e-5)) # log mel
        
        # --- Text Processing ---
        phonemes = arabic_to_phonemes(item['sentence'])
        tokens = phonemes_to_tokens(phonemes)
        # Filter out punctuation/unknown tokens
        tokens = [t for t in tokens if t in phon_to_id_]
        text_ids = torch.LongTensor(tokens_to_ids(tokens))
        
        return {
            "text_ids": text_ids,
            "mel_tgt": jepa_mel
        }

def jepa_collate_fn(batch):
    batch.sort(key=lambda x: x['text_ids'].size(0), reverse=True)
    
    max_text_len = max([x['text_ids'].size(0) for x in batch])
    
    text_ids_pad = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    input_lens = torch.zeros(len(batch), dtype=torch.long)
    
    mel_specs = []
    
    for i, item in enumerate(batch):
        text = item['text_ids']
        text_len = text.size(0)
        
        text_ids_pad[i, :text_len] = text
        input_lens[i] = text_len
        
        mel_specs.append(item['mel_tgt'])
        
    mel_specs = torch.stack(mel_specs, dim=0) # [B, 1, 128, 512]
    
    return mel_specs, text_ids_pad, input_lens
