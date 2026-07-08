import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from data.dataset import JEPADataset, jepa_collate_fn
from models.jepat_lightning import JEPATLightning

import re

def get_latest_checkpoint(log_dir):
    """Finds the most recent checkpoint between raw .pt files and Lightning .ckpt files."""
    latest_pt = None
    max_pt_epoch = -1
    
    # 1. Search for raw .pt files
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            if f.startswith("jepa_epoch_") and f.endswith(".pt"):
                try:
                    epoch = int(re.search(r"epoch_(\d+)", f).group(1))
                    if epoch > max_pt_epoch:
                        max_pt_epoch = epoch
                        latest_pt = os.path.join(log_dir, f)
                except:
                    pass
                    
    latest_ckpt = None
    max_ckpt_epoch = -1
    
    # 2. Search for Lightning .ckpt files
    checkpoints_dir = os.path.join(log_dir, "lightning_logs")
    if os.path.exists(checkpoints_dir):
        versions = [d for d in os.listdir(checkpoints_dir) if d.startswith("version_")]
        if versions:
            versions.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
            for version in versions:
                ckpt_dir = os.path.join(checkpoints_dir, version, "checkpoints")
                if os.path.exists(ckpt_dir):
                    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
                    for ckpt in ckpts:
                        try:
                            if "epoch=" in ckpt:
                                epoch = int(re.search(r"epoch=(\d+)", ckpt).group(1))
                                if epoch > max_ckpt_epoch:
                                    max_ckpt_epoch = epoch
                                    latest_ckpt = os.path.join(ckpt_dir, ckpt)
                        except:
                            pass
                    
                    if latest_ckpt:
                        # Favor last.ckpt if it exists alongside the highest epoch
                        if "last.ckpt" in ckpts:
                            latest_ckpt = os.path.join(ckpt_dir, "last.ckpt")
                        break

    # 3. Compare and return
    if max_ckpt_epoch == -1 and max_pt_epoch == -1:
        return None, None
        
    if max_ckpt_epoch >= max_pt_epoch:
        return latest_ckpt, "ckpt"
    else:
        return latest_pt, "pt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if it exists.")
    args = parser.parse_args()

    print("Initializing DataModule...")
    # Train on the full dataset split
    train_dataset = JEPADataset(split="train", max_frames=512)
    # Dynamically optimize data loading for Linux while keeping Windows safe
    workers = 4 if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=jepa_collate_fn,
        num_workers=workers 
    )

    print("Initializing Lightning JEPAT Model...")
    model = JEPATLightning(learning_rate=1e-4)

    # Configure checkpointing to save every 100 epochs, but also always save the very last completed epoch
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=100,
        save_top_k=-1, # Save all of them
        save_last=True, # Guarantee the latest epoch (even Epoch 1) is saved
        filename="jepa-{epoch:03d}"
    )

    print("Configuring Lightning Trainer...")
    trainer = L.Trainer(
        max_epochs=10000, # Train indefinitely until stopped
        accelerator="auto", # Supercomputer will use NVIDIA CUDA naturally
        devices=1,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        default_root_dir="training_logs"
    )

    ckpt_path = None
    if args.resume:
        found_path, ckpt_type = get_latest_checkpoint("training_logs")
        if found_path and os.path.exists(found_path):
            if ckpt_type == "pt":
                print(f"Resuming from raw PyTorch XPU weights: {found_path}")
                checkpoint = torch.load(found_path, map_location="cpu")
                model.model.load_state_dict(checkpoint['model_state_dict'])
                model.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                # We do NOT pass ckpt_path to trainer.fit because Lightning cannot parse raw .pt state
                ckpt_path = None
            else:
                print(f"Resuming natively from Lightning checkpoint: {found_path}")
                ckpt_path = found_path
        else:
            print("No checkpoint found to resume from. Starting from scratch.")

    print("Starting Training Loop!")
    trainer.fit(model, train_loader, ckpt_path=ckpt_path)
    
    print("Training finished! Checkpoints saved to 'training_logs/'")

if __name__ == "__main__":
    main()
