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

def get_latest_checkpoint(log_dir):
    """Finds the most recent checkpoint in the lightning_logs directory."""
    checkpoints_dir = os.path.join(log_dir, "lightning_logs")
    if not os.path.exists(checkpoints_dir):
        return None
    
    versions = [d for d in os.listdir(checkpoints_dir) if d.startswith("version_")]
    if not versions:
        return None
        
    versions.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
    latest_version = versions[0]
    
    ckpt_dir = os.path.join(checkpoints_dir, latest_version, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None
        
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        return None
        
    # Always prioritize the 'last.ckpt' if it exists
    if "last.ckpt" in ckpts:
        return os.path.join(ckpt_dir, "last.ckpt")
        
    return os.path.join(ckpt_dir, ckpts[0])

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
        accelerator="auto", 
        devices=1,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        default_root_dir="training_logs"
    )

    ckpt_path = None
    if args.resume:
        ckpt_path = get_latest_checkpoint("training_logs")
        if ckpt_path:
            print(f"Resuming from checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found to resume from. Starting from scratch.")

    print("Starting Training Loop!")
    trainer.fit(model, train_loader, ckpt_path=ckpt_path)
    
    print("Training finished! Checkpoints saved to 'training_logs/'")

if __name__ == "__main__":
    main()
