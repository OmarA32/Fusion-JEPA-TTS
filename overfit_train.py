import os
import sys
import copy
import time
import re
import argparse
import json
import lightning as L

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

from torch.utils.data import DataLoader
from data.dataset import JEPADataset, jepa_collate_fn
from models.jepat import JEPAT_base

def get_latest_checkpoint(log_dir):
    """Finds the most recent checkpoint recursively inside the log directory using os.walk."""
    
    # Priority: If an overfit.ckpt exists from a previous debugging run, load it immediately.
    overfit_path = os.path.join(log_dir, "overfit.ckpt")
    if os.path.exists(overfit_path):
        return overfit_path, "ckpt", -1
        
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
    """Keep only the most recent N periodic checkpoints to save disk space."""
    periodic_ckpts = []
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            if f.startswith("overfit_epoch_") and f.endswith(".ckpt"):
                periodic_ckpts.append(os.path.join(log_dir, f))
    
    # Sort by epoch number (extracted using regex)
    try:
        periodic_ckpts.sort(key=lambda x: int(re.search(r"overfit_epoch_(\d+)", os.path.basename(x)).group(1)))
    except:
        pass
    
    # Delete oldest if we exceed max_keep
    if len(periodic_ckpts) > max_keep:
        for f in periodic_ckpts[:-max_keep]:
            os.remove(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if it exists.")
    parser.add_argument("--lang", type=str, default="arabic", choices=["arabic", "english"], help="Language to train on.")
    parser.add_argument("--db", type=str, default="nawar_halabi", choices=["common_voice", "nawar_halabi", "clartts", "libritts", "ljspeech"], help="Database to use.")
    parser.add_argument("--val", action="store_true", help="Run validation step during training.")
    parser.add_argument("--freeze_jepa", action="store_true", help="Freeze the JEPA backbone and train only the Diffloss head.")
    parser.add_argument("--freeze_diffuser", action="store_true", help="Freeze the SpatialDiT diffuser and train only the JEPA backbone.")
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
        "arabic": ["common_voice", "nawar_halabi", "clartts"],
        "english": ["libritts", "ljspeech"]
    }
    if args.db not in valid_dbs[args.lang]:
        print(f"\n[ERROR] Language/Database mismatch! You cannot use database '{args.db}' with language '{args.lang}'.")
        print(f"Valid databases for {args.lang} are: {', '.join(valid_dbs[args.lang])}\n")
        sys.exit(1)
        
    # Isolate training logs by language to prevent checkpoint clashing
    log_dir = os.path.join("training_logs", args.lang)
    os.makedirs(log_dir, exist_ok=True)

    print("Initializing Device...")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using natively accelerated PyTorch device: {device}")

    print(f"Initializing DataModule for {args.lang.upper()} using {args.db.upper()}...")
    full_train_dataset = JEPADataset(split="train", lang=args.lang, db=args.db, max_frames=512)
    full_val_dataset = JEPADataset(split="validation", lang=args.lang, db=args.db, max_frames=512)
    
    from torch.utils.data import Subset
    
    # 5 random fixed indices for the scorched earth overfitting test
    if args.lang.lower() == 'arabic':
        overfit_indices = [80, 393, 689, 166, 357]
    else:
        overfit_indices = [71, 337, 707, 1429, 2842]
    
    # Ensure indices are within bounds
    overfit_indices = [idx % len(full_train_dataset) for idx in overfit_indices]
    
    train_dataset = Subset(full_train_dataset, overfit_indices)
    
    # We can evaluate on the same 8 files to ensure val loss perfectly tracks train loss
    val_dataset = Subset(full_val_dataset, [idx % len(full_val_dataset) for idx in overfit_indices])
    
    workers = 0 # No need for multiprocessing with 8 files

    # Robust Multi-GPU batch scaling
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if hasattr(torch, "xpu") and device.type == "xpu":
        num_gpus = torch.xpu.device_count()
        
    batch_size = 8 # Fit all 8 into a single batch if possible
    print(f"Detected {num_gpus} GPUs on {device.type.upper()}. Using batch size {batch_size}.")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=jepa_collate_fn,
        num_workers=workers 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=jepa_collate_fn,
        num_workers=workers
    )

    print("Initializing JEPAT Model...")
    model = JEPAT_base(
        in_channels=1, 
        language=args.lang,
        spec_height=128, 
        spec_width=512,
        patch_size=16,
        diffloss='flow', 
        jepaloss='jepa',
        grad_checkpointing=True
    ).to(device)
    
    ema_model = copy.deepcopy(model).to(device)
    for param in ema_model.parameters():
        param.requires_grad = False
        
    if args.freeze_jepa:
        print("Freezing JEPA (ViT) backbone! Only the Diffusion MLP will be trained.")
        for name, param in model.named_parameters():
            if not name.startswith("diffloss"):
                param.requires_grad = False
                
    if args.freeze_diffuser:
        print("Freezing SpatialDiT diffuser! Only the JEPA backbone will be trained.")
        for name, param in model.named_parameters():
            if name.startswith("diffloss"):
                param.requires_grad = False
    if num_gpus > 1:
        print(f"Wrapping models in DataParallel across {num_gpus} GPUs...")
        model = torch.nn.DataParallel(model)
        ema_model = torch.nn.DataParallel(ema_model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=1e-6)
    
    if device.type == "xpu" and "ipex" in sys.modules:
        print("Compiling model graph with IPEX for Float16 natively on XPU...")
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float16)
        ema_model = ipex.optimize(ema_model, dtype=torch.float16)

    # Setup mixed precision for XPU/CUDA
    scaler = torch.amp.GradScaler(device=device.type) if device.type in ["cuda", "xpu"] else None
    
    start_epoch = 0
    if args.resume:
        found_path, ckpt_type, found_epoch = get_latest_checkpoint(log_dir)
        if found_path and os.path.exists(found_path):
            print(f"Resuming training from {ckpt_type} checkpoint: {found_path}")
            checkpoint = torch.load(found_path, map_location=device, weights_only=False)
            
            # Auto-detect if this checkpoint has the legacy diffusion weights or the new SpatialDiT weights
            is_legacy_ckpt = False
            test_key = "model.diffloss.net.res_blocks.0.mlp.0.weight"
            state_dict_to_check = checkpoint['state_dict'] if ckpt_type == "ckpt" else checkpoint.get('model_state_dict', {})
            
            if test_key in state_dict_to_check:
                if state_dict_to_check[test_key].shape == torch.Size([1024, 1024]):
                    is_legacy_ckpt = True
                    
            if is_legacy_ckpt or args.freeze_jepa or args.freeze_diffuser:
                if is_legacy_ckpt:
                    print("Legacy diffusion architecture detected! Stripping incompatible weights and bypassing optimizer restoration...")
                    if ckpt_type == "pt":
                        model_state = {k: v for k, v in checkpoint.get('model_state_dict', {}).items() if 'diffloss' not in k}
                        ema_state = {k: v for k, v in checkpoint.get('ema_model_state_dict', {}).items() if 'diffloss' not in k}
                    else:
                        model_state = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.') and 'diffloss' not in k}
                        ema_state = {k.replace('ema_model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('ema_model.') and 'diffloss' not in k}
                else:
                    print("Freeze mechanism toggled on a checkpoint! Bypassing optimizer restoration to prevent shape mismatch...")
                    if ckpt_type == "pt":
                        model_state = checkpoint.get('model_state_dict', {})
                        ema_state = checkpoint.get('ema_model_state_dict', {})
                    else:
                        model_state = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
                        ema_state = {k.replace('ema_model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('ema_model.')}
                
                model.load_state_dict(model_state, strict=False)
                ema_model.load_state_dict(ema_state, strict=False)
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Skip optimizer to prevent shape crash
                
                start_epoch = checkpoint.get("epoch", 0) + 1
            else:
                print("Modern DiT architecture detected! Resuming natively...")
                if ckpt_type == "pt":
                    model.load_state_dict(checkpoint.get('model_state_dict', {}), strict=False)
                    ema_model.load_state_dict(checkpoint.get('ema_model_state_dict', {}), strict=False)
                    if 'optimizer_state_dict' in checkpoint:
                        try:
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        except Exception as e:
                            print(f"Warning: Failed to load PyTorch optimizer state: {e}")
                else:
                    model_state = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
                    ema_state = {k.replace('ema_model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('ema_model.')}
                    model.load_state_dict(model_state, strict=False)
                    ema_model.load_state_dict(ema_state, strict=False)
                
                start_epoch = checkpoint.get("epoch", 0) + 1
                
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
            autocast_dtype = torch.float32 if device.type == "cpu" else torch.float16
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
                
            scheduler.step()
            
            # Update EMA model
            with torch.no_grad():
                for ema_v, model_v in zip(ema_model.parameters(), model.parameters()):
                    ema_v.copy_(ema_v * ema_decay + (1. - ema_decay) * model_v)
            
            # Logging
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Total Loss: {total_loss.item():.4f} | Diff: {diffloss.item():.4f} | JEPA: {jepa_loss.item():.4f} | Time/batch: {elapsed:.2f}s", flush=True)
                
        # --- VALIDATION LOOP ---
        if args.val:
            print(f"Running Validation for Epoch {epoch}...")
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    mel_specs, text_ids, _ = val_batch
                    mel_specs = mel_specs.to(device)
                    text_ids = text_ids.to(device)
                    
                    # Run validation in native FP32 to bypass PyTorch .eval() mixed precision bugs
                    ema_x = ema_model.forward_ema_encoder(mel_specs, text_ids)
                    diffloss, jepa_loss = model(mel_specs, text_ids, ema_x=ema_x)
                    val_loss = diffloss + jepa_loss
                    val_losses.append(val_loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
            print(f"Validation Loss: {avg_val_loss:.4f}")
        else:
            avg_val_loss = 0.0 # Default if validation is skipped
        
        # --- LIGHTNING CHECKPOINT SYNTHESIZER ---
        # Synthesize a PyTorch Lightning Checkpoint dictionary
        lightning_ckpt = {
            "epoch": epoch,
            "global_step": (epoch + 1) * len(train_loader),
            "state_dict": {},
            "pytorch-lightning_version": L.__version__
        }
        
        # Inject state dict with Lightning prefixes (strip DataParallel 'module.' prefix if present)
        model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        ema_state = ema_model.module.state_dict() if isinstance(ema_model, torch.nn.DataParallel) else ema_model.state_dict()
        
        for k, v in model_state.items():
            lightning_ckpt["state_dict"][f"model.{k}"] = v
        for k, v in ema_state.items():
            lightning_ckpt["state_dict"][f"ema_model.{k}"] = v
            
        # --- CHECKPOINT SAVING (Only every 10 epochs to prevent disk IO bottleneck) ---
        if epoch % 10 == 0:
            # 1. Save Last Epoch (Overwrites previous last-epoch)
            last_ckpt_path = os.path.join(log_dir, "overfit.ckpt")
            torch.save(lightning_ckpt, last_ckpt_path)
            print(f"Saved latest checkpoint: {last_ckpt_path}")
            
            # 1.5 Save Periodic Checkpoint
            periodic_ckpt_path = os.path.join(log_dir, f"overfit_epoch_{epoch:04d}.ckpt")
            torch.save(lightning_ckpt, periodic_ckpt_path)
            print(f"Saved periodic checkpoint: {periodic_ckpt_path}")
            
            # Clean up older periodic checkpoints (keep top 3)
            clean_old_checkpoints(log_dir, max_keep=3)
            
            # 2. Save Best Epoch if loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt_path = os.path.join(log_dir, "best-overfit.ckpt")
                torch.save(lightning_ckpt, best_ckpt_path)
                print(f"Saved best checkpoint: {best_ckpt_path}")

if __name__ == "__main__":
    main()
