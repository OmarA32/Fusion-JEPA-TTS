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
    def __init__(self, jepa_encoder_kwargs: Dict[str, Any], fastpitch_kwargs: Dict[str, Any], jepa_embed_dim: int = 1024, fastpitch_embed_dim: int = 384, learning_rate: float = 1e-4, jepa_checkpoint_path: str = None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # 1. Context Encoder (Audio-JEPA)
        if jepa_checkpoint_path and os.path.exists(jepa_checkpoint_path):
            from src.models.jepa_module import JEPAModule
            print(f"Loading full JEPAModule from {jepa_checkpoint_path}")
            jepa_module = JEPAModule.load_from_checkpoint(jepa_checkpoint_path)
            self.context_encoder = jepa_module.encoder
        else:
            print("Warning: No JEPA checkpoint provided. Initializing random VisionTransformer.")
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
        # inputs: (text_padded, input_lengths, mel_padded, output_lengths, pitch_padded, energy_padded, speaker, attn_prior, audiopaths)
        
        # Extract Fingerprint
        patch_embeddings = self.context_encoder(reference_spectrogram)
        global_embedding = patch_embeddings.mean(dim=1)
        speaker_embedding = self.proj(global_embedding)
        
        # Inject into FastPitch
        fastpitch_inputs = list(inputs)
        fastpitch_inputs[6] = speaker_embedding
        
        # Return exact FastPitch outputs for the loss function
        return self.decoder(tuple(fastpitch_inputs))

    def infer(self, inputs: Tuple, reference_spectrogram: torch.Tensor):
        patch_embeddings = self.context_encoder(reference_spectrogram)
        global_embedding = patch_embeddings.mean(dim=1)
        speaker_embedding = self.proj(global_embedding)
        return self.decoder.infer(inputs[0], pace=1.0, speaker=speaker_embedding)

    def training_step(self, batch, batch_idx):
        from models.fastpitch.fastpitch.loss_function import FastPitchLoss
        if not hasattr(self, 'loss_fn'):
            from models.fastpitch.fastpitch.loss_function import FastPitchLoss
            self.loss_fn = FastPitchLoss(pitch_predictor_loss_scale=0.0)
            
        fp_inputs, ref_specs = batch
        mel_tgt, in_lens, out_lens = fp_inputs[2], fp_inputs[1], fp_inputs[3]
        targets = (mel_tgt, in_lens, out_lens)

        model_out = self(fp_inputs, ref_specs)
        loss, meta = self.loss_fn(model_out, targets)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if not hasattr(self, 'loss_fn'):
            from models.fastpitch.fastpitch.loss_function import FastPitchLoss
            self.loss_fn = FastPitchLoss(pitch_predictor_loss_scale=0.0)
            
        fp_inputs, ref_specs = batch
        mel_tgt, in_lens, out_lens = fp_inputs[2], fp_inputs[1], fp_inputs[3]
        targets = (mel_tgt, in_lens, out_lens)
        
        model_out = self(fp_inputs, ref_specs)
        loss, meta = self.loss_fn(model_out, targets)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.learning_rate
        )
        return optimizer
