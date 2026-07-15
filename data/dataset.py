import os
import sys
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import urllib.request
import zipfile

# Ensure local imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from text import arabic_to_phonemes, phonemes_to_tokens, tokens_to_ids, phon_to_id_

def download_and_extract_nawar_halabi(data_dir):
    """Downloads and extracts the Nawar Halabi dataset if missing."""
    target_dir = os.path.join(data_dir, "arabic-speech-corpus")
    if os.path.exists(target_dir):
        return target_dir
        
    print("Nawar Halabi dataset not found locally. Auto-downloading...")
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "arabic-speech-corpus.zip")
    
    url = "https://en.arabicspeechcorpus.com/arabic-speech-corpus.zip"
    print(f"Downloading from {url} (this may take a while)...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    print("Download and extraction complete!")
    return target_dir

class JEPADataset(Dataset):
    def __init__(self, split="train", lang="arabic", db="common_voice", jepa_sr=24000, max_frames=512, n_mels=100):
        super().__init__()
        self.lang = lang.lower()
        self.db = db.lower()
        self.split = split
        self.jepa_sr = jepa_sr
        self.max_frames = max_frames
        
        # Audio extraction function
        self.jepa_mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=jepa_sr,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            f_min=0,
            f_max=12000
        )
        
        self.data_dir = os.path.join(PROJECT_ROOT, "data")
        self.dataset = []
        self._load_database()

    def _load_database(self):
        print(f"Loading {self.lang.upper()} dataset: {self.db.upper()}...")
        
        if self.lang == "arabic":
            if self.db == "common_voice":
                # Use HF native filter for high-speed offline disk caching
                hf_split = "train" if "train" in self.split else "validation"
                ds = load_dataset("MohamedRashad/common-voice-18-arabic", split=hf_split)
                ds = ds.cast_column("audio", Audio(decode=False))
                self.dataset = ds.filter(lambda x: x['gender'] == 'male_masculine')
                print(f"Loaded {len(self.dataset)} male clips from Common Voice.")
                
            elif self.db == "nawar_halabi":
                target_dir = download_and_extract_nawar_halabi(self.data_dir)
                wavs_dir = os.path.join(target_dir, "wav")
                transcript_path = os.path.join(target_dir, "orthographic-transcript.txt")
                
                with open(transcript_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split('" "')
                    if len(parts) == 2:
                        wav_name = parts[0].replace('"', '').strip()
                        buckw_text = parts[1].replace('"', '').strip()
                        audio_path = os.path.join(wavs_dir, wav_name)
                        
                        if os.path.exists(audio_path):
                            self.dataset.append({
                                "audio_path": audio_path,
                                "sentence": buckw_text
                            })
                            
                print(f"Loaded {len(self.dataset)} clips from Nawar Halabi.")
            else:
                raise ValueError(f"Database {self.db} not supported for Arabic.")
                
        elif self.lang == "english":
            if self.db == "ljspeech":
                ds = torchaudio.datasets.LJSPEECH(self.data_dir, download=True)
                for item in ds:
                    self.dataset.append({
                        "audio_tensor": item[0],
                        "sample_rate": item[1],
                        "sentence": item[2]
                    })
                print(f"Loaded {len(self.dataset)} clips from LJSpeech.")
            elif self.db == "libritts":
                ds = torchaudio.datasets.LIBRITTS(self.data_dir, url="train-clean-100", download=True)
                for item in ds:
                    self.dataset.append({
                        "audio_tensor": item[0],
                        "sample_rate": item[1],
                        "sentence": item[2]
                    })
                print(f"Loaded {len(self.dataset)} clips from LibriTTS.")
            else:
                raise ValueError(f"Database {self.db} not supported for English.")
        else:
            raise ValueError(f"Language {self.lang} not supported.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # --- Audio Processing ---
        if self.lang == "arabic" and self.db == "common_voice":
            import io
            import soundfile as sf
            audio_bytes = item['audio']['bytes']
            wav_np, sr = sf.read(io.BytesIO(audio_bytes))
            wav = torch.tensor(wav_np, dtype=torch.float32)
        elif self.lang == "arabic" and self.db == "nawar_halabi":
            import soundfile as sf
            wav_np, sr = sf.read(item['audio_path'])
            wav = torch.tensor(wav_np, dtype=torch.float32)
            if len(wav.shape) > 1:
                wav = wav[:, 0] # convert to mono if stereo
        else:
            # English datasets via torchaudio return tensors
            wav = item['audio_tensor'].squeeze(0)
            sr = item['sample_rate']
            
        if sr != self.jepa_sr:
            wav = torchaudio.functional.resample(wav, sr, self.jepa_sr)
            
        target_len = (self.max_frames - 1) * self.jepa_mel_fn.hop_length
        if len(wav) > target_len:
            wav = wav[:target_len]
        else:
            wav = F.pad(wav, (0, target_len - len(wav)))
            
        jepa_mel = self.jepa_mel_fn(wav).unsqueeze(0) # [1, 128, T]
        jepa_mel = torch.log(torch.clamp(jepa_mel, min=1e-5)) # log mel
        
        # --- Text Processing ---
        if self.lang == "arabic":
            if self.db == "nawar_halabi":
                from text import buckwalter_to_phonemes
                phonemes = buckwalter_to_phonemes(item['sentence'])
            else:
                phonemes = arabic_to_phonemes(item['sentence'])
            tokens = phonemes_to_tokens(phonemes)
            tokens = [t for t in tokens if t in phon_to_id_]
            text_ids = torch.LongTensor(tokens_to_ids(tokens))
        else:
            # English (Placeholder logic as requested by user)
            # Just return a dummy tensor so training logic doesn't crash
            text_ids = torch.LongTensor([0, 1, 2])
            
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
