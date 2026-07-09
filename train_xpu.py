import os
import sys
import copy
import time

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from torch.utils.data import DataLoader
from data.dataset import JEPADataset, jepa_collate_fn
from models.jepat import JEPAT_base
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
    workers = 4 if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
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
    found_path, ckpt_type = get_latest_checkpoint("training_logs")
    if found_path and os.path.exists(found_path):
        print(f"Resuming training from {ckpt_type} checkpoint: {found_path}")
        checkpoint = torch.load(found_path, map_location=device, weights_only=False)
        
        if ckpt_type == "pt":
            # Raw PyTorch weights
            model.load_state_dict(checkpoint['model_state_dict'])
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        elif ckpt_type == "ckpt":
            # PyTorch Lightning weights (requires stripping the 'model.' and 'ema_model.' prefixes)
            state_dict = checkpoint['state_dict']
            model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            ema_state = {k.replace('ema_model.', ''): v for k, v in state_dict.items() if k.startswith('ema_model.')}
            model.load_state_dict(model_state)
            ema_model.load_state_dict(ema_state)
            # Lightning's optimizer state is nested, so we skip restoring it for manual loops to prevent shape crashing
            start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed successfully. Starting at epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")
    
    print("Starting Raw PyTorch Training Loop!")
    ema_decay = 0.9999
    
    for epoch in range(start_epoch, 10000):
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
                
        # Save checkpoint at the end of each epoch
        ckpt_path = os.path.join("training_logs", f"jepa_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss.item(),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
