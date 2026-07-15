import os
import sys
import copy
import time
import re
import lightning as L

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from torch.utils.data import DataLoader
from data.dataset import JEPADataset, jepa_collate_fn
from models.jepat import JEPAT_base

def get_latest_checkpoint(log_dir):
    """Finds the most recent checkpoint recursively inside the log directory using os.walk."""
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

def clean_old_checkpoints(log_dir, max_keep=3):
    """Keep only the most recent N best checkpoints to save disk space."""
    best_ckpts = []
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            if f.startswith("best-epoch=") and f.endswith(".ckpt"):
                best_ckpts.append(os.path.join(log_dir, f))
    
    # Sort by epoch number (extracted using regex)
    try:
        best_ckpts.sort(key=lambda x: int(re.search(r"epoch=(\d+)", os.path.basename(x)).group(1)))
    except:
        pass
    
    # Delete oldest if we exceed max_keep
    if len(best_ckpts) > max_keep:
        for f in best_ckpts[:-max_keep]:
            os.remove(f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if it exists.")
    args = parser.parse_args()

    print("Initializing Device...")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using natively accelerated PyTorch device: {device}")

    print("Initializing DataModule...")
    train_dataset = JEPADataset(split="train", max_frames=512)
    val_dataset = JEPADataset(split="validation", max_frames=512)
    workers = 4 if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=jepa_collate_fn,
        num_workers=workers 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=jepa_collate_fn,
        num_workers=workers
    )

    print("Initializing JEPAT Model...")
    model = JEPAT_base(
        in_channels=1, 
        language='arabic',
        spec_height=100, 
        spec_width=512,
        diffloss='flow', 
        jepaloss='jepa'
    ).to(device)
    
    ema_model = copy.deepcopy(model).to(device)
    for param in ema_model.parameters():
        param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.02)
    
    # Setup mixed precision for XPU/CUDA
    scaler = torch.amp.GradScaler(device=device.type) if device.type in ["cuda", "xpu"] else None
    
    os.makedirs("training_logs", exist_ok=True)
    
    start_epoch = 0
    if args.resume:
        found_path, ckpt_type, found_epoch = get_latest_checkpoint("training_logs")
        if found_path and os.path.exists(found_path):
            print(f"Resuming training from {ckpt_type} checkpoint: {found_path}")
            checkpoint = torch.load(found_path, map_location=device, weights_only=False)
            
            if ckpt_type == "pt":
                # Raw PyTorch weights
                model.load_state_dict(checkpoint['model_state_dict'])
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = found_epoch + 1
            elif ckpt_type == "ckpt":
                # PyTorch Lightning weights (requires stripping the 'model.' and 'ema_model.' prefixes)
                state_dict = checkpoint['state_dict']
                model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
                ema_state = {k.replace('ema_model.', ''): v for k, v in state_dict.items() if k.startswith('ema_model.')}
                model.load_state_dict(model_state)
                ema_model.load_state_dict(ema_state)
                # Lightning's optimizer state is nested, so we skip restoring it for manual loops to prevent shape crashing
                start_epoch = found_epoch + 1
            print(f"Resumed successfully. Starting at epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")
    
    print("Starting Raw PyTorch Training Loop with Lightning Compatibility!")
    ema_decay = 0.9999
    
    # Validation trackers
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, 10000):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()
            
            # Unpack batch and move strictly to hardware device
            mel_specs, text_ids, _ = batch
            mel_specs = mel_specs.to(device)
            text_ids = text_ids.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            autocast_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                with torch.no_grad():
                    ema_x = ema_model.forward_ema_encoder(mel_specs, text_ids)
                
                diffloss, jepa_loss = model(mel_specs, text_ids, ema_x=ema_x)
                total_loss = diffloss + jepa_loss
            
            # Backward pass
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            # Update EMA model
            with torch.no_grad():
                for ema_v, model_v in zip(ema_model.parameters(), model.parameters()):
                    ema_v.copy_(ema_v * ema_decay + (1. - ema_decay) * model_v)
            
            # Logging
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Total Loss: {total_loss.item():.4f} | Diff: {diffloss.item():.4f} | JEPA: {jepa_loss.item():.4f} | Time/batch: {elapsed:.2f}s")
                
        # --- VALIDATION LOOP ---
        print(f"Running Validation for Epoch {epoch}...")
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                mel_specs, text_ids, _ = val_batch
                mel_specs = mel_specs.to(device)
                text_ids = text_ids.to(device)
                
                autocast_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                    ema_x = ema_model.forward_ema_encoder(mel_specs, text_ids)
                    diffloss, jepa_loss = model(mel_specs, text_ids, ema_x=ema_x)
                    val_loss = diffloss + jepa_loss
                    val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # --- LIGHTNING CHECKPOINT SYNTHESIZER ---
        # Synthesize a PyTorch Lightning Checkpoint dictionary
        lightning_ckpt = {
            "epoch": epoch,
            "global_step": (epoch + 1) * len(train_loader),
            "state_dict": {},
            "pytorch-lightning_version": L.__version__
        }
        
        # Inject state dict with Lightning prefixes
        for k, v in model.state_dict().items():
            lightning_ckpt["state_dict"][f"model.{k}"] = v
        for k, v in ema_model.state_dict().items():
            lightning_ckpt["state_dict"][f"ema_model.{k}"] = v
            
        # 1. Save Last Epoch (Overwrites previous last-epoch)
        # Find any existing last-epoch files and delete them
        for f in os.listdir("training_logs"):
            if f.startswith("last-epoch=") and f.endswith(".ckpt"):
                os.remove(os.path.join("training_logs", f))
                
        last_ckpt_path = os.path.join("training_logs", f"last-epoch={epoch:03d}.ckpt")
        torch.save(lightning_ckpt, last_ckpt_path)
        print(f"Saved latest checkpoint: {last_ckpt_path}")
        
        # 2. Save Best Epoch if loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join("training_logs", f"best-epoch={epoch:03d}.ckpt")
            torch.save(lightning_ckpt, best_ckpt_path)
            print(f"Saved best checkpoint: {best_ckpt_path}")
            
            # Clean up old best checkpoints (keep top 3)
            clean_old_checkpoints("training_logs", max_keep=3)

if __name__ == "__main__":
    main()
