import os
import sys
import torch
import torchaudio
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from datasets import load_dataset
import random

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tts-arabic-pytorch"))
print("PROJECT_ROOT inside datamodule:", PROJECT_ROOT)
print("os.path.exists(tts-arabic-pytorch/utils/audio.py):", os.path.exists(os.path.join(PROJECT_ROOT, "tts-arabic-pytorch", "utils", "audio.py")))

import sys
print("sys.path:", sys.path)
import utils
print("utils module:", utils.__file__)

from text import arabic_to_phonemes, phonemes_to_tokens, tokens_to_ids
try:
    from utils.audio import MelSpectrogram
except Exception as e:
    print("FAILED TO IMPORT utils.audio:", type(e), str(e))
    raise e

class ZeroShotDataset(Dataset):
    def __init__(self, split="train[:10]", fp_sr=22050, jepa_sr=32000):
        super().__init__()
        print(f"Loading HF dataset split {split}...")
        from datasets import Audio
        self.dataset = list(load_dataset("MohamedRashad/common-voice-18-arabic", split=split).cast_column("audio", Audio(decode=False)))
        self.fp_sr = fp_sr
        self.jepa_sr = jepa_sr
        
        # FastPitch requires 80 mel bins, JEPA requires 128
        self.fp_mel_fn = MelSpectrogram(sample_rate=fp_sr, n_mels=80)
        self.jepa_mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=jepa_sr,
            n_mels=128,
            n_fft=1024,
            hop_length=320,
            win_length=1024,
            f_min=0,
            f_max=8000
        )
        
        # Pre-group by client_id to find references easily
        from collections import defaultdict
        self.client_audio = defaultdict(list)
        for i, item in enumerate(self.dataset):
            self.client_audio[item['client_id']].append(i)

    def __len__(self):
        return len(self.dataset)

    def _extract_pitch_energy(self, wav, mel_spec):
        # Energy
        energy = torch.norm(mel_spec, dim=0)
        
        # Pitch
        pitch_mel, _, _ = librosa.pyin(
            wav, sr=self.fp_sr,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=self.fp_mel_fn.win_length,
            hop_length=self.fp_mel_fn.hop_length
        )
        pitch_mel = np.where(np.isnan(pitch_mel), 0., pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel)
        pitch_mel = F.pad(pitch_mel, (0, max(0, mel_spec.size(1) - pitch_mel.size(0))))
        pitch_mel = pitch_mel[:mel_spec.size(1)] # Truncate if longer
        return pitch_mel.float(), energy.float()

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # --- FastPitch Target Audio ---
        import io
        import soundfile as sf
        audio_bytes = item['audio']['bytes']
        wav_np, sr = sf.read(io.BytesIO(audio_bytes))
        wav = torch.tensor(wav_np, dtype=torch.float32)
        
        if sr != self.fp_sr:
            fp_wav = torchaudio.functional.resample(wav, sr, self.fp_sr)
        else:
            fp_wav = wav
            
        fp_mel = self.fp_mel_fn(fp_wav.unsqueeze(0)).squeeze(0) # [80, T]
        pitch, energy = self._extract_pitch_energy(fp_wav.numpy(), fp_mel)
        
        # --- Text to IDs ---
        try:
            phonemes = arabic_to_phonemes(item['sentence'])
            tokens = phonemes_to_tokens(phonemes)
            text_ids = torch.LongTensor(tokens_to_ids(tokens))
        except Exception as e:
            # Fallback to dummy if normalization fails
            text_ids = torch.LongTensor([1, 2, 3])
            
        # --- JEPA Reference Audio ---
        client_id = item['client_id']
        candidates = self.client_audio[client_id]
        if len(candidates) > 1:
            ref_idx = random.choice([i for i in candidates if i != idx])
        else:
            ref_idx = idx
            
        ref_item = self.dataset[ref_idx]
        ref_bytes = ref_item['audio']['bytes']
        ref_wav_np, ref_sr = sf.read(io.BytesIO(ref_bytes))
        ref_wav = torch.tensor(ref_wav_np, dtype=torch.float32)
        
        if ref_sr != self.jepa_sr:
            jepa_wav = torchaudio.functional.resample(ref_wav, ref_sr, self.jepa_sr)
        else:
            jepa_wav = ref_wav
            
        # JEPA expects input_size=[128, 256], which means 256 frames.
        # With hop_length=320, 256 frames = (256 - 1) * 320 = 81600 samples.
        target_len = 81600
        if len(jepa_wav) > target_len:
            jepa_wav = jepa_wav[:target_len]
        else:
            jepa_wav = F.pad(jepa_wav, (0, target_len - len(jepa_wav)))
            
        jepa_mel = self.jepa_mel_fn(jepa_wav).unsqueeze(0) # [1, 128, T]
        jepa_mel = torch.log(torch.clamp(jepa_mel, min=1e-5)) # log mel
        
        return {
            "text_ids": text_ids,
            "mel_tgt": fp_mel,
            "pitch": pitch,
            "energy": energy,
            "ref_spec": jepa_mel
        }

def zero_shot_collate_fn(batch):
    # Sort by text length descending (required by FastPitch)
    batch.sort(key=lambda x: x['text_ids'].size(0), reverse=True)
    
    max_text_len = max([x['text_ids'].size(0) for x in batch])
    max_mel_len = max([x['mel_tgt'].size(1) for x in batch])
    
    text_ids_pad = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    input_lens = torch.zeros(len(batch), dtype=torch.long)
    mel_pad = torch.zeros(len(batch), 80, max_mel_len, dtype=torch.float32)
    output_lens = torch.zeros(len(batch), dtype=torch.long)
    pitch_pad = torch.zeros(len(batch), 1, max_mel_len, dtype=torch.float32)
    energy_pad = torch.zeros(len(batch), max_mel_len, dtype=torch.float32)
    
    ref_specs = []
    
    for i, item in enumerate(batch):
        text = item['text_ids']
        mel = item['mel_tgt']
        pitch = item['pitch']
        energy = item['energy']
        
        text_len = text.size(0)
        mel_len = mel.size(1)
        
        text_ids_pad[i, :text_len] = text
        input_lens[i] = text_len
        
        mel_pad[i, :, :mel_len] = mel
        output_lens[i] = mel_len
        
        pitch_pad[i, 0, :mel_len] = pitch
        energy_pad[i, :mel_len] = energy
        
        ref_specs.append(item['ref_spec'])
        
    ref_specs = torch.stack(ref_specs, dim=0) # [B, 1, 128, T]
    
    # FastPitch expects: (text_padded, input_lengths, mel_padded, output_lengths, pitch_padded, energy_padded, speaker, attn_prior, audiopaths)
    fp_inputs = (
        text_ids_pad,
        input_lens,
        mel_pad,
        output_lens,
        pitch_pad,
        energy_pad,
        torch.zeros(len(batch), dtype=torch.long), # Dummy speaker
        None, # attn_prior
        None  # audiopaths
    )
    
    return fp_inputs, ref_specs

class ZeroShotDataModule(L.LightningDataModule):
    def __init__(self, batch_size=2):
        super().__init__()
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        self.train_dataset = ZeroShotDataset(split="train[:10]")
        self.val_dataset = ZeroShotDataset(split="train[:2]") # Just 2 for val

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=zero_shot_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=zero_shot_collate_fn)
