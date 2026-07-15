import torch
from vocos import Vocos

class VocoderManager:
    def __init__(self, vocoder_type="vocos", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocoder_type = vocoder_type.lower()
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.vocoder_type.upper()} vocoder...")
        if self.vocoder_type == 'vocos':
            # Dynamically pull state-of-the-art Vocos from HuggingFace
            self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)
        elif self.vocoder_type == 'bigvgan':
            import sys
            if 'BigVGAN' not in sys.path:
                sys.path.append('BigVGAN')
            from bigvgan import BigVGAN
            self.model = BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False).to(self.device)
            self.model.eval()
        else:
            raise ValueError("Invalid vocoder_type. Only 'vocos' and 'bigvgan' are supported.")
        print(f"{self.vocoder_type.upper()} loaded successfully.")

    @torch.no_grad()
    def generate_audio(self, mel_spectrogram):
        """
        Takes a Mel-Spectrogram tensor [B, Channels, Time] and returns audio waveform.
        """
        mel_spectrogram = mel_spectrogram.to(self.device)
        
        if self.vocoder_type == 'vocos':
            audio = self.model.decode(mel_spectrogram)
        elif self.vocoder_type == 'bigvgan':
            audio = self.model(mel_spectrogram)
            
        return audio
