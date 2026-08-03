import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import torch
torch.set_float32_matmul_precision('high') # ⚡ MAXIMIZES speed on A100s and RTX 30/40 series via TF32
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

from data.dataset import JEPADataset, jepa_collate_fn
from models.jepat_lightning import JEPATLightning

import re
import json
import warnings
import time

# Mute harmless PyTorch DDP stream warnings
warnings.filterwarnings("ignore", message=".*AccumulateGrad node's stream does not match.*")
from huggingface_hub import HfApi, login

class HuggingFaceUploadCallback(L.Callback):
    def __init__(self, every_n_epochs, repo_id, log_dir):
        self.every_n_epochs = every_n_epochs
        self.repo_id = repo_id
        self.log_dir = log_dir
        self.hf_token = None
        
        if os.path.exists("hf_config.json"):
            try:
                with open("hf_config.json", "r") as f:
                    self.hf_token = json.load(f).get("HF_TOKEN")
            except Exception as e:
                print(f"Error reading hf_config.json: {e}")
                
        if self.hf_token:
            print(f"Hugging Face token found! Models will be automatically uploaded to {self.repo_id} every {self.every_n_epochs} epochs.")
            try:
                login(token=self.hf_token)
            except Exception as e:
                print(f"HF Login failed: {e}")
        else:
            print("Warning: No hf_config.json found or invalid token. Automatic HF uploads will fail.")
            
    def on_train_epoch_end(self, trainer, pl_module):
        # trainer.current_epoch is 0-indexed, so add 1 to get standard epoch number
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0 and trainer.is_global_zero:
            # Wait for PyTorch Lightning's background saver thread to finish writing the new checkpoint
            time.sleep(3)
            latest_ckpt, _, _ = get_latest_checkpoint(self.log_dir)
            if latest_ckpt and self.hf_token:
                print(f"\n[HF Upload] Epoch {epoch}: Uploading {latest_ckpt} to {self.repo_id}...")
                try:
                    api = HfApi()
                    api.create_repo(repo_id=self.repo_id, exist_ok=True, repo_type="model")
                    filename = os.path.basename(latest_ckpt)
                    api.upload_file(
                        path_or_fileobj=latest_ckpt,
                        path_in_repo=filename,
                        repo_id=self.repo_id,
                        repo_type="model",
                        commit_message=f"Auto-upload from Epoch {epoch}"
                    )
                    print(f"[HF Upload] Success!")
                except Exception as e:
                    print(f"[HF Upload] Error: {e}")

class SaveLastCheckpointCallback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        ckpt_path = os.path.join(trainer.default_root_dir, "last.ckpt")
        trainer.save_checkpoint(ckpt_path)

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
    parser.add_argument("--lang", type=str, default="arabic", choices=["arabic", "english"], help="Language to train on.")
    parser.add_argument("--db", type=str, default="nawar_halabi", choices=["common_voice", "nawar_halabi", "libritts", "ljspeech"], help="Database to use.")
    parser.add_argument("--checkpointnum", type=int, default=0, help="Upload to Hugging Face every N epochs (0 disables).")
    parser.add_argument("--val", action="store_true", help="Enable validation loop during training.")
    parser.add_argument("--freeze_jepa", action="store_true", help="Freeze the JEPA backbone and train only the Diffloss head.")
    parser.add_argument("--hf_token", type=str, default=None, help="Save a Hugging Face token to hf_config.json automatically.")
    parser.add_argument("--download_latest", action="store_true", help="Download the latest checkpoint from Hugging Face before starting.")
    args = parser.parse_args()
    
    if args.hf_token:
        with open("hf_config.json", "w") as f:
            json.dump({"HF_TOKEN": args.hf_token}, f)
        print("Successfully saved HF_TOKEN to hf_config.json!")

    if args.download_latest:
        print(f"Downloading latest {args.lang} checkpoint from Hugging Face...")
        import subprocess
        subprocess.run([sys.executable, "download_from_hf.py", "--lang", args.lang], check=False)
    
    valid_dbs = {
        "arabic": ["common_voice", "nawar_halabi"],
        "english": ["libritts", "ljspeech"]
    }
    if args.db not in valid_dbs[args.lang]:
        print(f"\n[ERROR] Language/Database mismatch! You cannot use database '{args.db}' with language '{args.lang}'.")
        print(f"Valid databases for {args.lang} are: {', '.join(valid_dbs[args.lang])}\n")
        sys.exit(1)
        
    log_dir = os.path.join("training_logs", args.lang)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Initializing DataModule for {args.lang.upper()} using {args.db.upper()}...")
    # Train on the full dataset split
    train_dataset = JEPADataset(split="train", lang=args.lang, db=args.db, max_frames=512)
    # Dynamically optimize data loading for Linux while keeping Windows safe
    workers = 4 if os.name != 'nt' else 0
    
    # Robust Multi-GPU batch scaling
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    batch_size = 12 # Increased batch size for newer architecture iteration
    print(f"Detected {num_gpus} GPUs. Using batch size {batch_size}.")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=jepa_collate_fn,
        num_workers=workers 
    )
    
    # Initialize Validation
    val_dataset = JEPADataset(split="validation", lang=args.lang, db=args.db, max_frames=512)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=jepa_collate_fn,
        num_workers=workers
    )

    print("Initializing Lightning JEPAT Model...")
    model = JEPATLightning(learning_rate=1e-4, language=args.lang)

    if args.freeze_jepa:
        print("Freezing JEPA (ViT) backbone! Only the Diffusion MLP will be trained.")
        for name, param in model.model.named_parameters():
            if not name.startswith("diffloss"):
                param.requires_grad = False
                
    # Freezing the EMA model in Lightning (if it exists as a separate attribute)
    if hasattr(model, "ema_model"):
        for param in model.ema_model.parameters():
            param.requires_grad = False

    # Configure checkpointing to save the best 3 epochs based on validation loss (or latest 3 if val is off)
    if args.val:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            every_n_epochs=1,
            save_top_k=1,
            save_last=True,
            filename="best-epoch={epoch:03d}"
        )
    else:
        checkpoint_callback = SaveLastCheckpointCallback()

    callbacks_list = [checkpoint_callback, TQDMProgressBar(refresh_rate=1)]
    
    if args.checkpointnum > 0:
        repo_id = "KAST-JEPA-QUANTIZED/Arabic" if args.lang == "arabic" else "KAST-JEPA-QUANTIZED/English"
        callbacks_list.append(HuggingFaceUploadCallback(every_n_epochs=args.checkpointnum, repo_id=repo_id, log_dir=log_dir))

    print("Configuring Lightning Trainer...")
    
    # Robust Multi-GPU Strategy
    lightning_strategy = "ddp" if num_gpus > 1 else "auto"
    
    trainer = L.Trainer(
        max_epochs=10000, # Train indefinitely until stopped
        accelerator="auto", # Supercomputer will use NVIDIA CUDA naturally
        devices="auto",
        strategy=lightning_strategy,
        precision="16-mixed", # ⚡ NVIDIA T4 FIX: T4s use Turing architecture which requires standard 16-bit for Tensor Core speedups!
        limit_val_batches=1.0 if args.val else 0.0,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        callbacks=callbacks_list,
        default_root_dir=log_dir
    )

    ckpt_path = None
    if args.resume:
        found_path, ckpt_type, found_epoch = get_latest_checkpoint(log_dir)
        if found_path and os.path.exists(found_path):
            print(f"Inspecting checkpoint payload: {found_path}")
            checkpoint = torch.load(found_path, map_location="cpu")
            
            if "pytorch-lightning_version" not in checkpoint:
                print(f"Detected raw PyTorch weights! Upgrading to Lightning Checkpoint format...")
                
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
                        
                temp_ckpt_path = os.path.join(log_dir, "temp_upgrade.ckpt")
                torch.save(lightning_ckpt, temp_ckpt_path)
                
                ckpt_path = temp_ckpt_path
                print(f"Resuming natively from upgraded Lightning checkpoint! (Epoch {found_epoch})")
            else:
                if "optimizer_states" not in checkpoint:
                    print("Lightning checkpoint is missing optimizer states (likely stripped for HF upload). Loading weights only!")
                    # Load weights manually to avoid crashing Lightning's strict strict resume
                    # We use strict=False in case the checkpoint has EMA models but the code doesn't
                    model.load_state_dict(checkpoint["state_dict"], strict=False)
                    ckpt_path = None # Set to None so Lightning starts fresh optimizers safely
                    
                    # SAFEGUARD: Archive the stripped checkpoint so it doesn't infinitely loop on future resumes
                    archived_path = os.path.join(os.path.dirname(found_path), f"ARCHIVED_{os.path.basename(found_path)}")
                    print(f"Archiving stripped checkpoint to {archived_path} to protect future resumes...")
                    try:
                        import shutil
                        shutil.move(found_path, archived_path)
                    except Exception as e:
                        print(f"Warning: Could not archive {found_path}: {e}")
                else:
                    print(f"Resuming natively from valid Lightning checkpoint: {found_path}")
                    ckpt_path = found_path
        else:
            print("No checkpoint found to resume from. Starting from scratch.")

    print("Starting Training Loop!")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    
    print("Training finished! Checkpoints saved to: ", log_dir)

if __name__ == "__main__":
    main()
