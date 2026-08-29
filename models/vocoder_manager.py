import os
import sys
import torch

# Ensure BigVGAN submodule is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
bigvgan_dir = os.path.join(PROJECT_ROOT, "BigVGAN")
if bigvgan_dir not in sys.path:
    sys.path.insert(0, bigvgan_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class VocoderManager:
    """
    Manages neural vocoding using NVIDIA's BigVGAN v2 (44.1 kHz, 128-band Mel).
    """
    def __init__(self, device=None):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = device or torch.device("xpu")
        elif torch.cuda.is_available():
            self.device = device or torch.device("cuda")
        else:
            self.device = device or torch.device("cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        print("Loading BigVGAN v2 44.1kHz vocoder...")
        if 'BigVGAN' not in sys.path:
            sys.path.append('BigVGAN')
        from bigvgan import BigVGAN
        
        # Compatibility monkeypatch for huggingface-hub parameter changes
        original_from_pretrained = BigVGAN._from_pretrained
        @classmethod
        def _from_pretrained_patched(cls, *args, **kwargs):
            if 'proxies' not in kwargs:
                kwargs['proxies'] = None
            if 'resume_download' not in kwargs:
                kwargs['resume_download'] = False
            return original_from_pretrained.__func__(cls, *args, **kwargs)
        BigVGAN._from_pretrained = _from_pretrained_patched
        
        self.model = BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False).to(self.device)
        self.model.eval()
        print("BigVGAN loaded successfully.")

    @torch.no_grad()
    def generate_audio(self, mel_spectrogram):
        """
        Takes a Mel-Spectrogram tensor [B, Channels, Time] and returns audio waveform tensor.
        """
        mel_spectrogram = mel_spectrogram.to(self.device)
        audio = self.model(mel_spectrogram)
        return audio
