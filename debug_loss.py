import torch
import sys
import os

sys.path.insert(0, os.path.abspath('Audio-JEPA'))
sys.path.insert(0, os.path.abspath('tts-arabic-pytorch'))

from src.data.zero_shot_datamodule import ZeroShotDataModule
from src.models.zero_shot_tts import ZeroShotTTS
from models.fastpitch.fastpitch.loss_function import FastPitchLoss
from models.fastpitch import net_config

jepa_kwargs = {"input_size": [128, 256], "patch_size": [16, 16], "in_chans": 1, "embed_dim": 768, "depth": 2, "num_heads": 4}
model = ZeroShotTTS(jepa_encoder_kwargs=jepa_kwargs, fastpitch_kwargs=net_config, jepa_embed_dim=768, fastpitch_embed_dim=384)
model.eval()

datamodule = ZeroShotDataModule(batch_size=2)
datamodule.setup()
batch = next(iter(datamodule.train_dataloader()))
fp_inputs, ref_specs = batch

with torch.no_grad():
    model_out = model(fp_inputs, ref_specs)
    loss_fn = FastPitchLoss()
    mel_tgt, in_lens, out_lens = fp_inputs[2], fp_inputs[1], fp_inputs[3]
    targets = (mel_tgt, in_lens, out_lens)
    loss, meta = loss_fn(model_out, targets)

print(f"Total Loss: {loss}")
print(f"Loss Meta: {meta}")

# Let's inspect model outputs to see which one is NaN
mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, energy_pred = model_out
print(f"mel_out has nan: {torch.isnan(mel_out).any()}")
print(f"dur_pred has nan: {torch.isnan(dur_pred).any()}")
print(f"pitch_pred has nan: {torch.isnan(pitch_pred).any()}")
print(f"energy_pred has nan: {torch.isnan(energy_pred).any()}")

# Let's inspect targets to see if they are NaN
print(f"mel_tgt has nan: {torch.isnan(mel_tgt).any()}")
