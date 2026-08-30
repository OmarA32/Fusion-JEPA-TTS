import argparse
import os
import json
import sys
from huggingface_hub import HfApi, login

def get_config_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p1 = os.path.join(script_dir, filename)
    if os.path.exists(p1):
        return p1
    if os.path.exists(filename):
        return filename
    p2 = os.path.join("tools", filename)
    if os.path.exists(p2):
        return p2
    return None

def get_token():
    cfg = get_config_path("hf_config.json")
    if cfg and os.path.exists(cfg):
        with open(cfg, "r") as f:
            data = json.load(f)
            return data.get("HF_TOKEN", None)
    return None

def get_repo_id(lang):
    cfg = get_config_path("hf_repos.json")
    if cfg and os.path.exists(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                repos = json.load(f)
                if lang in repos:
                    return repos[lang]
        except Exception:
            pass
    return "KAST-JEPA-QUANTIZED/Arabic" if lang == "arabic" else "KAST-JEPA-QUANTIZED/English"

def upload_model(checkpoint_path, repo_id, hf_token, commit_message="Upload best JEPA-TTS model"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found!")
        return

    print(f"Logging into Hugging Face...")
    login(token=hf_token)
    
    api = HfApi()
    
    print(f"Ensuring repository '{repo_id}' exists...")
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    
    filename = os.path.basename(checkpoint_path)
    
    print(f"Uploading {checkpoint_path} to {repo_id}/{filename}...")
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message
    )
    print("Upload complete!")

def get_latest_checkpoint(log_dir):
    """Finds the most recent checkpoint recursively inside the log directory."""
    import re
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
                    except Exception:
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
                            max_ckpt_epoch = 99999999
                            latest_ckpt = filepath
                    except Exception:
                        pass

    if max_ckpt_epoch == -1 and max_pt_epoch == -1:
        return None, None, 0
        
    if max_ckpt_epoch >= max_pt_epoch:
        return latest_ckpt, "ckpt", max_ckpt_epoch
    else:
        return latest_pt, "pt", max_pt_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload latest JEPA-TTS weights to Hugging Face")
    parser.add_argument("--lang", type=str, default="arabic", choices=["arabic", "english"], help="Language model to upload")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token (optional if hf_config.json exists)")
    parser.add_argument("--ckpt", type=str, default=None, help="Explicit path to a .ckpt or .pt model checkpoint to upload")
    parser.add_argument("--repo", type=str, default=None, help="Explicit Hugging Face repo ID override")
    args = parser.parse_args()
    
    # 1. Resolve Token
    token = args.token or get_token()
    if not token:
        print("[ERROR] No Hugging Face token provided! Either pass --token or run the HF Auth notebook cell first.")
        sys.exit(1)
        
    # 2. Resolve Repo
    repo_id = args.repo or get_repo_id(args.lang)
    
    # 3. Resolve Checkpoint
    if args.ckpt and os.path.exists(args.ckpt):
        latest_ckpt = args.ckpt
        max_epoch = 99999999
    else:
        log_dir = os.path.join("training_logs", args.lang)
        latest_ckpt, ckpt_type, max_epoch = get_latest_checkpoint(log_dir)
    
    if not latest_ckpt or not os.path.exists(latest_ckpt):
        print(f"[ERROR] No valid checkpoints found to upload!")
        sys.exit(1)
        
    # Add 1 to max_epoch to convert PyTorch Lightning's 0-indexed epoch into a human-readable 1-indexed number
    epoch_str = f"Epoch {max_epoch + 1}"
    if max_epoch == 99999999 or "last.ckpt" in latest_ckpt:
        try:
            import torch
            ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            real_epoch = ckpt.get("epoch", None)
            if real_epoch is not None and isinstance(real_epoch, int):
                # PyTorch Lightning stores 0-indexed epoch, so add +1 for human-readable count
                epoch_str = f"Epoch {real_epoch + 1}"
            else:
                epoch_str = "Latest Epoch"
        except Exception:
            epoch_str = "Latest Epoch"
            
    commit_message = f"Manual upload from {epoch_str}"
    upload_model(latest_ckpt, repo_id, token, commit_message=commit_message)
