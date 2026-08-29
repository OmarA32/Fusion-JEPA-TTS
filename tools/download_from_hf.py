import argparse
import os
import json
import re
from huggingface_hub import HfApi, hf_hub_download

def get_config_path(filename):
    # Check directory of this script first, then current working directory
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

def main():
    parser = argparse.ArgumentParser(description="Download latest weights from Hugging Face")
    parser.add_argument("--lang", type=str, default="arabic", choices=["arabic", "english"], help="Language model to download")
    parser.add_argument("--repo", type=str, default=None, help="Explicit Hugging Face repo ID override")
    args = parser.parse_args()

    token = get_token()
    repo_id = args.repo or get_repo_id(args.lang)
    local_dir = os.path.join("training_logs", args.lang)

    print(f"Connecting to {repo_id}...")
    api = HfApi(token=token)
    
    try:
        files = api.list_repo_files(repo_id=repo_id)
    except Exception as e:
        print(f"Failed to access repository: {e}")
        return

    # Filter for checkpoint files
    ckpt_files = [f for f in files if f.endswith(".ckpt")]
    
    if not ckpt_files:
        print("No checkpoint files found in the repository!")
        return

    # Try to find the latest last.ckpt first, otherwise fallback to the highest best-epoch
    def extract_epoch(filename):
        match = re.search(r'epoch=(\d+)', filename)
        return int(match.group(1)) if match else -1

    # 1. Parse the LATEST commit message to see if it specifies an epoch
    try:
        latest_commit = api.list_repo_commits(repo_id=repo_id)[0]
        commit_msg = latest_commit.title
        match = re.search(r'(?i)epoch\s*(\d+)', commit_msg)
        if match:
            target_epoch = match.group(1)
            # Find a file that matches this exact epoch
            target_files = [f for f in ckpt_files if f"epoch={target_epoch}" in f]
            if target_files:
                latest_file = target_files[0] # Prefer the first match (e.g. best-epoch=129)
                print(f"Found explicit epoch {target_epoch} in latest commit message!")
    except Exception as e:
        print(f"Could not parse commit history: {e}")
        
    # 2. If the commit parsing didn't find a target, fallback to last.ckpt or highest epoch
    if 'latest_file' not in locals():
        if any(f.endswith("last.ckpt") for f in ckpt_files):
            latest_file = next(f for f in ckpt_files if f.endswith("last.ckpt"))
        else:
            latest_file = max(ckpt_files, key=extract_epoch)

    print(f"Found latest weights: {latest_file}")
    
    os.makedirs(local_dir, exist_ok=True)
    
    # Check if we already have it
    local_path = os.path.join(local_dir, latest_file.split('/')[-1])
    if os.path.exists(local_path):
        print(f"File {local_path} already exists. Skipping download.")
        return

    print(f"Downloading to {local_dir}...")
    try:
        hf_hub_download(repo_id=repo_id, filename=latest_file, local_dir=local_dir, local_dir_use_symlinks=False, token=token)
        print(f"Successfully downloaded to {local_dir}/{latest_file.split('/')[-1]}")
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    main()
