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
    import glob
    import re
    ckpt_files = glob.glob(os.path.join("training_logs", "jepa_epoch_*.pt"))
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=lambda f: int(re.search(r"jepa_epoch_(\d+)\.pt", f).group(1)))
        print(f"Resuming training from {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed successfully. Starting at epoch {start_epoch}")
    
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
