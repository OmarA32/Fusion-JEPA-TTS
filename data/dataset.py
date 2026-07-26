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
    def __init__(self, split="train", lang="arabic", db="nawar_halabi", jepa_sr=44100, max_frames=512, n_mels=128):
        super().__init__()
        self.lang = lang.lower()
        self.db = db.lower()
        self.split = split
        self.jepa_sr = jepa_sr
        self.max_frames = max_frames
        
        # Audio extraction function
        bigvgan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "BigVGAN")
        if bigvgan_path not in sys.path:
            sys.path.append(bigvgan_path)
        from meldataset import mel_spectrogram
        self.jepa_mel_fn = mel_spectrogram
        
        self.mel_kwargs = {
            "n_fft": 2048,
            "num_mels": n_mels,
            "sampling_rate": jepa_sr,
            "hop_size": 512,
            "win_size": 2048,
            "fmin": 0,
            "fmax": None,
            "center": False
        }
        
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
                
            elif self.db == "clartts":
                ds = load_dataset("MBZUAI/ClArTTS", split="train")
                ds = ds.cast_column("audio", Audio(decode=False))
                self.dataset = ds
                print(f"Loaded {len(self.dataset)} clips from ClArTTS.")
                
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
                ljs_path = os.path.join(self.data_dir, "LJSpeech-1.1")
                if not os.path.exists(ljs_path):
                    print("LJSpeech not found. Downloading via Python...")
                    import urllib.request
                    import tarfile
                    
                    tar_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
                    tar_path = os.path.join(self.data_dir, "LJSpeech-1.1.tar.bz2")
                    
                    # 1. Download
                    if not os.path.exists(tar_path):
                        print(f"Downloading {tar_url}...")
                        def _progress(count, block_size, total_size):
                            percent = int(count * block_size * 100 / total_size)
                            if percent % 10 == 0:
                                print(f"\rDownloading... {percent}%", end="")
                        urllib.request.urlretrieve(tar_url, tar_path, reporthook=_progress)
                        print("\nDownload complete.")
                    
                    # 2. Extract
                    print("Extracting LJSpeech...")
                    with tarfile.open(tar_path, "r:bz2") as tar:
                        tar.extractall(path=self.data_dir)
                    
                    # 3. Clean up
                    os.remove(tar_path)
                    
                    if not os.path.exists(ljs_path):
                        raise RuntimeError("LJSpeech download/extraction failed!")
                        
                csv_path = os.path.join(ljs_path, "metadata.csv")
                wavs_dir = os.path.join(ljs_path, "wavs")
                with open(csv_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("|")
                        if len(parts) >= 2:
                            self.dataset.append({
                                "audio_path": os.path.join(wavs_dir, parts[0] + ".wav"),
                                "sentence": parts[2] if len(parts) > 2 and parts[2] else parts[1]
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
        if self.lang == "arabic" and self.db in ["common_voice", "clartts"]:
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
        elif self.lang == "english":
            import soundfile as sf
            if "audio_path" in item:
                wav_np, sr = sf.read(item['audio_path'])
                wav = torch.tensor(wav_np, dtype=torch.float32)
            else:
                wav = item['audio_tensor'].squeeze(0)
                sr = item['sample_rate']
            
        if sr != self.jepa_sr:
            wav = torchaudio.functional.resample(wav, sr, self.jepa_sr)
            
        target_len = self.max_frames * self.mel_kwargs["hop_size"]
        if len(wav) > target_len:
            wav = wav[:target_len]
        else:
            wav = F.pad(wav, (0, target_len - len(wav)))
            
        # BigVGAN's mel_spectrogram expects [B, T] and outputs natively log-compressed magnitudes
        jepa_mel = self.jepa_mel_fn(wav.unsqueeze(0), **self.mel_kwargs) # [1, 128, T]
        
        # --- Text Processing ---
        if self.lang == "arabic":
            if self.db == "nawar_halabi":
                from text import buckwalter_to_phonemes
                phonemes = buckwalter_to_phonemes(item['sentence'])
            else:
                sentence_text = item.get('sentence', item.get('text', ''))
                phonemes = arabic_to_phonemes(sentence_text)
            tokens = phonemes_to_tokens(phonemes)
            tokens = [t for t in tokens if t in phon_to_id_]
            text_ids = torch.LongTensor(tokens_to_ids(tokens))
        else:
            from text import english_to_tokens
            sentence_text = item.get('sentence', item.get('text', ''))
            tokens = english_to_tokens(sentence_text)
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
