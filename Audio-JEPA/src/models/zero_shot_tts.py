import torch
import torch.nn as nn
from lightning import LightningModule
from typing import Dict, Any, Tuple

from src.models.components.vision_transformer import VisionTransformer
import sys
import os
# Assuming zero_shot_tts is run/imported from a path that makes the root available
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
tts_dir = os.path.join(project_root, "tts-arabic-pytorch")
if tts_dir not in sys.path:
    sys.path.insert(0, tts_dir)

from models.fastpitch.fastpitch.model import FastPitch

class ZeroShotTTS(LightningModule):
    def __init__(self, jepa_encoder_kwargs: Dict[str, Any], fastpitch_kwargs: Dict[str, Any], jepa_embed_dim: int = 1024, fastpitch_embed_dim: int = 384, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # 1. Context Encoder (Audio-JEPA)
        self.context_encoder = VisionTransformer(**jepa_encoder_kwargs)
        
        # We freeze the context encoder initially to stabilize FastPitch training
        for param in self.context_encoder.parameters():
            param.requires_grad = False

        # 2. Projection Layer
        # Maps the mean-pooled JEPA patch embeddings to FastPitch speaker embedding dimension
        self.proj = nn.Linear(jepa_embed_dim, fastpitch_embed_dim)

        # 3. TTS Decoder (FastPitch)
        self.decoder = FastPitch(**fastpitch_kwargs)
        
    def forward(self, inputs: Tuple, reference_spectrogram: torch.Tensor):
        """
        inputs: Tuple containing text symbols, mel target, lens, etc. for FastPitch
        reference_spectrogram: [B, C, Freq, Time] spectrogram of the 3s reference audio
        """
        # A. Extract Speaker Fingerprint
        # The context encoder returns patches [B, N, D]. We mean-pool to get a global representation [B, D].
        patch_embeddings = self.context_encoder(reference_spectrogram)
        global_embedding = patch_embeddings.mean(dim=1)
        
        # B. Project to FastPitch Dimension
        speaker_embedding = self.proj(global_embedding)
        
        # Replace the `speaker` index in the `inputs` tuple with the continuous speaker_embedding
        # FastPitch expects inputs = (inputs, input_lens, mel_tgt, mel_lens, pitch_dense, energy_dense, speaker, attn_prior, audiopaths)
        fastpitch_inputs = list(inputs)
        fastpitch_inputs[6] = speaker_embedding
        fastpitch_inputs = tuple(fastpitch_inputs)
        
        # C. Decode
        mel_out, mel_out_postnet, dec_lens, dur_pred, pitch_pred, energy_pred, pitch_tgt, energy_tgt, attn_soft, attn_hard, attn_hard_dur, attn_logprob = self.decoder(fastpitch_inputs)
        
        return mel_out, mel_out_postnet, dur_pred, pitch_pred, energy_pred

    def training_step(self, batch, batch_idx):
        # We assume the batch provides the reference spectrogram and FastPitch inputs
        fastpitch_inputs = batch['fastpitch_inputs']
        reference_spectrogram = batch['reference_transformed_waveform']
        
        # Forward pass
        mel_out, mel_out_postnet, dur_pred, pitch_pred, energy_pred = self(fastpitch_inputs, reference_spectrogram)
        
        # We would compute the loss here (e.g. MSE between mel_out and mel_tgt)
        # For the sake of Phase 2 structural validation, we use a dummy loss 
        # (The actual loss requires importing FastPitchLoss)
        mel_tgt = fastpitch_inputs[2]
        loss = torch.nn.functional.mse_loss(mel_out, mel_tgt)
        
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.learning_rate
        )
        return optimizer
