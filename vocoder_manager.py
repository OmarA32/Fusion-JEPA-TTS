import torch
from vocos import Vocos
from bigvganinference import BigVGANInference
import json
import sys
import os

# Append the vocoder directory so the local HiFi-GAN models module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "vocoder"))
from hifigan.env import AttrDict
from hifigan.models import Generator as HiFiGANGenerator

class VocoderManager:
    def __init__(self, vocoder_type='hifigan', device='cpu'):
        self.vocoder_type = vocoder_type.lower()
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.vocoder_type.upper()} vocoder...")
        if self.vocoder_type == 'vocos':
            # Dynamically pull state-of-the-art Vocos from HuggingFace
            self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)
        elif self.vocoder_type == 'hifigan':
            # Load local custom HiFi-GAN
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "pretrained", "hifigan-asc-v1", "config.json")
            weights_path = os.path.join(base_dir, "pretrained", "hifigan-asc-v1", "hifigan-asc.pth")
            
            with open(config_path) as f:
                h = AttrDict(json.load(f))
            self.model = HiFiGANGenerator(h).to(self.device)
            state_dict = torch.load(weights_path, map_location=self.device)
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.remove_weight_norm()
        elif self.vocoder_type == 'bigvgan':
            # Dynamically pull BigVGAN from HuggingFace
            self.model = BigVGANInference.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
            self.model = self.model.to(self.device)
        else:
            raise ValueError("Invalid vocoder_type. Choose 'vocos', 'hifigan', or 'bigvgan'.")
        print(f"{self.vocoder_type.upper()} loaded successfully.")

    @torch.no_grad()
    def generate_audio(self, mel_spectrogram):
        """
        Takes a Mel-Spectrogram tensor [B, Channels, Time] and returns audio waveform.
        """
        mel_spectrogram = mel_spectrogram.to(self.device)
        
        if self.vocoder_type == 'vocos':
            audio = self.model.decode(mel_spectrogram)
        elif self.vocoder_type == 'hifigan':
            audio = self.model(mel_spectrogram)
            audio = audio.squeeze(1) # Remove channel dim if present
        elif self.vocoder_type == 'bigvgan':
            audio = self.model(mel_spectrogram)
            if audio.dim() == 3:
                audio = audio.squeeze(1)
            
        return audio
