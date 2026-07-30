import argparse
import os
import json
import sys
from huggingface_hub import HfApi, login

def get_token():
    if os.path.exists("hf_config.json"):
        with open("hf_config.json", "r") as f:
            data = json.load(f)
            return data.get("HF_TOKEN", None)
    return None

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload latest JEPA-TTS weights to Hugging Face")
    parser.add_argument("--lang", type=str, default="arabic", choices=["arabic", "english"], help="Language model to upload")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token (optional if hf_config.json exists)")
    args = parser.parse_args()
    
    # 1. Resolve Token
    token = args.token or get_token()
    if not token:
        print("[ERROR] No Hugging Face token provided! Either pass --token or run the HF Auth notebook cell first.")
        sys.exit(1)
        
    # 2. Resolve Repo
    repo_id = "KAST-JEPA-QUANTIZED/Arabic" if args.lang == "arabic" else "KAST-JEPA-QUANTIZED/English"
    
    # 3. Resolve Checkpoint
    from train import get_latest_checkpoint
    log_dir = os.path.join("training_logs", args.lang)
    latest_ckpt, ckpt_type, max_epoch = get_latest_checkpoint(log_dir)
    
    if not latest_ckpt:
        print(f"[ERROR] No checkpoints found in {log_dir} to upload!")
        sys.exit(1)
        
    epoch_str = f"Epoch {max_epoch}"
    if max_epoch == 99999999: # last.ckpt indicator
        try:
            import torch
            ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            real_epoch = ckpt.get("epoch", "Latest")
            epoch_str = f"Epoch {real_epoch}"
        except:
            epoch_str = "Latest Epoch"
            
    commit_message = f"Manual upload from {epoch_str}"
    upload_model(latest_ckpt, repo_id, token, commit_message=commit_message)
