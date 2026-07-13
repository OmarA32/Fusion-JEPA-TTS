import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

from data.dataset import JEPADataset, jepa_collate_fn
from models.jepat_lightning import JEPATLightning

import re

def get_latest_checkpoint(log_dir):
    """Finds the most recent checkpoint recursively inside the log directory."""
    latest_pt = None
    max_pt_epoch = -1
    latest_ckpt = None
    max_ckpt_epoch = -1
    
    if os.path.exists(log_dir):
        for root, dirs, files in os.walk(log_dir):
            for f in files:
                filepath = os.path.join(root, f)
                
                # Check for raw .pt files
                if f.startswith("jepa_epoch_") and f.endswith(".pt"):
                    try:
                        epoch = int(re.search(r"epoch_(\d+)", f).group(1))
                        if epoch > max_pt_epoch:
                            max_pt_epoch = epoch
                            latest_pt = filepath
                    except:
                        pass
                
                # Check for Lightning .ckpt files
                elif f.endswith(".ckpt"):
                    try:
                        if "epoch=" in f:
                            epoch = int(re.search(r"epoch=(\d+)", f).group(1))
                            if epoch > max_ckpt_epoch:
                                max_ckpt_epoch = epoch
                                latest_ckpt = filepath
                        elif f == "last.ckpt":
                            # last.ckpt takes absolute highest priority
                            max_ckpt_epoch = 99999999
                            latest_ckpt = filepath
                    except:
                        pass

    if max_ckpt_epoch == -1 and max_pt_epoch == -1:
        return None, None, 0
        
    if max_ckpt_epoch >= max_pt_epoch:
        return latest_ckpt, "ckpt", max_ckpt_epoch
    else:
        return latest_pt, "pt", max_pt_epoch

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
    
    # Initialize Validation
    val_dataset = JEPADataset(split="validation", max_frames=512)
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=jepa_collate_fn,
        num_workers=workers
    )

    print("Initializing Lightning JEPAT Model...")
    model = JEPATLightning(learning_rate=1e-4)

    # Configure checkpointing to save the best 3 epochs based on validation loss
    checkpoint_callback_best = ModelCheckpoint(
        monitor="val/total_loss",
        mode="min",
        every_n_epochs=1,
        save_top_k=3, # Keep the 3 best epochs
        filename="best-epoch={epoch:03d}"
    )

    # Configure a second checkpoint to ALWAYS save the absolute latest epoch with its number
    checkpoint_callback_last = ModelCheckpoint(
        monitor="step", # Monitor the global step to always get the latest
        mode="max",
        every_n_epochs=1,
        save_top_k=1, # Only keep 1 file to prevent disk crash
        filename="last-epoch={epoch:03d}"
    )

    print("Configuring Lightning Trainer...")
    trainer = L.Trainer(
        max_epochs=10000, # Train indefinitely until stopped
        accelerator="auto", # Supercomputer will use NVIDIA CUDA naturally
        devices="auto",
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback_best, checkpoint_callback_last, TQDMProgressBar(refresh_rate=300)],
        default_root_dir="training_logs"
    )

    ckpt_path = None
    if args.resume:
        found_path, ckpt_type, found_epoch = get_latest_checkpoint("training_logs")
        if found_path and os.path.exists(found_path):
            if ckpt_type == "pt":
                print(f"Upgrading raw PyTorch weights ({found_path}) to a Lightning Checkpoint...")
                checkpoint = torch.load(found_path, map_location="cpu")
                
                # Synthesize a PyTorch Lightning Checkpoint
                lightning_ckpt = {
                    "epoch": found_epoch,
                    "global_step": found_epoch * len(train_loader), # approximate global step
                    "state_dict": {},
                    "pytorch-lightning_version": L.__version__
                }
                
                # Prefix model state
                if 'model_state_dict' in checkpoint:
                    for k, v in checkpoint['model_state_dict'].items():
                        lightning_ckpt["state_dict"][f"model.{k}"] = v
                if 'ema_model_state_dict' in checkpoint:
                    for k, v in checkpoint['ema_model_state_dict'].items():
                        lightning_ckpt["state_dict"][f"ema_model.{k}"] = v
                        
                temp_ckpt_path = os.path.join("training_logs", "temp_upgrade.ckpt")
                torch.save(lightning_ckpt, temp_ckpt_path)
                
                ckpt_path = temp_ckpt_path
                print(f"Resuming natively from upgraded Lightning checkpoint! (Epoch {found_epoch})")
            else:
                print(f"Resuming natively from Lightning checkpoint: {found_path}")
                ckpt_path = found_path
        else:
            print("No checkpoint found to resume from. Starting from scratch.")

    print("Starting Training Loop!")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    
    print("Training finished! Checkpoints saved to 'training_logs/'")

if __name__ == "__main__":
    main()
