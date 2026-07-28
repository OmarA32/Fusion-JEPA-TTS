import argparse
import os
import json
import re
from huggingface_hub import HfApi, hf_hub_download

def get_token():
    if os.path.exists("hf_config.json"):
        with open("hf_config.json", "r") as f:
            data = json.load(f)
            return data.get("HF_TOKEN", None)
    return None

def main():
    parser = argparse.ArgumentParser(description="Download latest weights from Hugging Face")
    parser.add_argument("--lang", type=str, choices=["arabic", "english"], required=True, help="Language model to download")
    args = parser.parse_args()

    token = get_token()
    repo_id = "KAST-JEPA-QUANTIZED/Arabic" if args.lang == "arabic" else "KAST-JEPA-QUANTIZED/English"
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

    # Prefer last.ckpt for resuming
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
