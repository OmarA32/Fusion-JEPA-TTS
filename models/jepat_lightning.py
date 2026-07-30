import copy
import torch
import lightning as L
from models.jepat import JEPAT_base

class JEPATLightning(L.LightningModule):
    def __init__(self, learning_rate=1e-4, ema_decay=0.9999, language='arabic', **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.ema_decay = ema_decay
        
        # Initialize the base model
        self.model = JEPAT_base(
            in_channels=1, 
            language=language,
            spec_height=128, 
            spec_width=512,
            diffloss='flow', # Using Flow Matching
            jepaloss='jepa'
        )
        
        # Initialize EMA model
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA model parameters
        decay = self.ema_decay
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_v.copy_(ema_v * decay + (1. - decay) * model_v)

    def training_step(self, batch, batch_idx):
        mel_specs, text_ids, _ = batch
        
        # 1. Get targets from EMA model
        # The EMA encoder processes the unmasked image to create targets for the JEPA loss
        with torch.no_grad():
            ema_x = self.ema_model.forward_ema_encoder(mel_specs, text_ids)
            
        # 2. Forward pass main model
        diffloss, jepa_loss = self.model(mel_specs, text_ids, ema_x=ema_x)
        
        # 3. Calculate total loss
        # Give diffusion a strong weight
        total_loss = diffloss + jepa_loss
        
        # Logging
        self.log("train/diffloss", diffloss, prog_bar=True)
        self.log("train/jepa_loss", jepa_loss, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        mel_specs, text_ids, _ = batch
        
        with torch.no_grad():
            ema_x = self.ema_model.forward_ema_encoder(mel_specs, text_ids)
            diffloss, jepa_loss = self.model(mel_specs, text_ids, ema_x=ema_x)
            total_loss = diffloss + jepa_loss
            
        self.log("val/diffloss", diffloss, prog_bar=True, sync_dist=True)
        self.log("val/jepa_loss", jepa_loss, prog_bar=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, sync_dist=True)
        
        return total_loss

    def configure_optimizers(self):
        # Filter parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=0.02)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
