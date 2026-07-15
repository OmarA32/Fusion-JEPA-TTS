import argparse
import os
from huggingface_hub import HfApi, login

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
    parser = argparse.ArgumentParser(description="Upload JEPA-TTS weights to Hugging Face")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face Repo ID")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Write Token")
    args = parser.parse_args()
    
    upload_model(args.ckpt, args.repo, args.token)
